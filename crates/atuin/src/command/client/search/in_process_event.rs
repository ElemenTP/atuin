//! Terminal event source for the in-process interactive TUI.
//!
//! The CLI uses `crossterm::event`, which on Unix lazily installs a
//! process-wide SIGWINCH handler through signal-hook. That is fine for a
//! short-lived binary, but in a `dlclose`d shell module it leaves a dangling
//! function pointer behind and can crash the host shell on the next resize.
//!
//! This module therefore supplies the small `poll`/`read` interface used by
//! the TUI render loop without touching `crossterm::event` on Unix. It parses
//! the subset of the terminal protocol the interactive search uses:
//!
//! * keys (including CSI/SS3 cursor keys and UTF-8 input);
//! * CSI-u / Kitty keyboard protocol keys (`ESC [ codepoint ; modifiers u`),
//!   which upstream enables through keyboard enhancement flags;
//! * SGR and X10 mouse reports (the TUI enables any-event tracking, so raw
//!   mouse *motion* sequences must be consumed rather than misread as `Esc`);
//! * bracketed paste (`ESC [ 200 ~ ... ESC [ 201 ~`), which upstream inserts
//!   into the query without triggering key bindings.
//!
//! Windows console input uses no signal handlers, so the normal crossterm
//! wrappers are safe there.

use std::io;
use std::time::Duration;

#[cfg(unix)]
pub use unix::{poll, read, reset};
#[cfg(windows)]
pub use windows::{poll, read, reset};

#[cfg(unix)]
mod unix {
    use super::*;
    use ratatui::crossterm::event::{
        Event, KeyCode, KeyEvent, KeyModifiers, MediaKeyCode, MouseButton, MouseEvent,
        MouseEventKind,
    };
    use std::fs::{File, OpenOptions};
    use std::io::Read;
    use std::os::fd::AsRawFd;
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    /// A fully parsed terminal event.
    #[derive(Debug)]
    enum Parsed {
        Key(KeyEvent),
        Mouse(MouseEvent),
        Paste(String),
    }

    struct Input {
        file: File,
        buf: Vec<u8>,
    }

    enum Parse {
        Event(Parsed, usize),
        NeedMore,
    }

    /// How long to keep collecting a bracketed-paste payload before yielding a
    /// partial paste. The terminal normally delivers the whole payload in one
    /// read; this only guards against a truncated/malformed sequence.
    const PASTE_TIMEOUT: Duration = Duration::from_millis(500);
    const PARTIAL_WAIT: Duration = Duration::from_millis(30);
    /// How long an incomplete escape/UTF-8 sequence is kept before it is
    /// dropped as an ignored key (a lone ESC is delivered after `PARTIAL_WAIT`).
    const INCOMPLETE_TIMEOUT: Duration = Duration::from_millis(500);
    /// Upper bound on a bracketed paste held in memory before it is delivered:
    /// an unterminated paste must not grow the buffer without limit.
    const MAX_PASTE: usize = 1 << 20;
    /// Upper bound on the parameter bytes of a single CSI/SS3 sequence.
    const MAX_SEQ: usize = 32;

    fn input() -> &'static Mutex<Option<Input>> {
        static INPUT: OnceLock<Mutex<Option<Input>>> = OnceLock::new();
        INPUT.get_or_init(|| Mutex::new(None))
    }

    fn with_input<T>(f: impl FnOnce(&mut Input) -> io::Result<T>) -> io::Result<T> {
        let mut guard = input().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.is_none() {
            *guard = Some(Input::open()?);
        }
        f(guard.as_mut().expect("input initialized above"))
    }

    /// Drop the cached terminal handle and any buffered input.
    ///
    /// The handle lives in a process-global `OnceLock`, so without this it
    /// would keep an open `/dev/tty` fd across sessions and library unloads,
    /// and unread bytes from a previous TUI run could leak into the next one.
    pub fn reset() {
        let mut guard = input().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = None;
    }

    pub fn poll(timeout: Duration) -> io::Result<bool> {
        with_input(|input| input.poll(timeout))
    }

    pub fn read() -> io::Result<Event> {
        with_input(|input| input.read_event())
    }

    /// A key event with an explicit modifier set, mirroring crossterm.
    fn key_event(code: KeyCode, modifiers: KeyModifiers) -> Parsed {
        Parsed::Key(KeyEvent::new(code, modifiers))
    }

    /// A character key event. crossterm marks uppercase ASCII as shifted.
    fn char_event(ch: char, modifiers: KeyModifiers) -> Parsed {
        let modifiers = if ch.is_ascii_uppercase() {
            modifiers | KeyModifiers::SHIFT
        } else {
            modifiers
        };
        Parsed::Key(KeyEvent::new(KeyCode::Char(ch), modifiers))
    }

    /// The ignored key used for unrecognized input: upstream drops
    /// `KeyCode::Null`, so unknown terminal reports cannot cancel the TUI.
    fn null_event() -> Parsed {
        Parsed::Key(KeyEvent::new(KeyCode::Null, KeyModifiers::NONE))
    }

    /// The replacement character for undecodable bytes.
    fn replacement_event() -> Parsed {
        Parsed::Key(KeyEvent::new(KeyCode::Char('\u{fffd}'), KeyModifiers::NONE))
    }

    fn parsed_event(parsed: Parsed) -> Event {
        match parsed {
            Parsed::Key(key) => Event::Key(key),
            Parsed::Mouse(mouse) => Event::Mouse(mouse),
            Parsed::Paste(text) => Event::Paste(text),
        }
    }

    impl Input {
        fn open() -> io::Result<Self> {
            let file = OpenOptions::new().read(true).open("/dev/tty")?;
            let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
            if flags < 0 {
                return Err(io::Error::last_os_error());
            }
            if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                file,
                buf: Vec::with_capacity(256),
            })
        }

        fn poll(&mut self, timeout: Duration) -> io::Result<bool> {
            if !self.buf.is_empty() {
                return Ok(true);
            }
            self.wait_fd(timeout)
        }

        fn read_event(&mut self) -> io::Result<Event> {
            let mut paste_deadline: Option<Instant> = None;
            // Tracks the last observed incomplete sequence (length, first seen)
            // so a stalled terminal cannot hold the loop forever.
            let mut incomplete: Option<(usize, Instant)> = None;

            loop {
                self.drain()?;
                if self.buf.is_empty() {
                    // Nothing buffered. Wait for input, but a readable fd that
                    // yields no bytes (EOF, hangup, or a spurious wakeup) must
                    // not spin this loop, so give up after one empty wait.
                    if !self.wait_fd(PARTIAL_WAIT)? || self.drain()? == 0 {
                        return Ok(parsed_event(null_event()));
                    }
                    continue;
                }

                // Bracketed paste is variable length and may span reads, so it
                // is collected here rather than in the stateless parser.
                if self.buf.starts_with(b"\x1b[200~") {
                    let deadline =
                        paste_deadline.get_or_insert_with(|| Instant::now() + PASTE_TIMEOUT);
                    if let Some(end) = find_subslice(&self.buf[6..], b"\x1b[201~") {
                        let text = String::from_utf8_lossy(&self.buf[6..6 + end]).into_owned();
                        self.buf.drain(..6 + end + 6);
                        return Ok(parsed_event(Parsed::Paste(text)));
                    }
                    // Never buffer an unbounded paste: a missing end marker
                    // must not be able to grow the buffer without limit.
                    if Instant::now() >= *deadline || self.buf.len() >= MAX_PASTE {
                        let text = String::from_utf8_lossy(&self.buf[6..]).into_owned();
                        self.buf.clear();
                        return Ok(parsed_event(Parsed::Paste(text)));
                    }
                    // Reset the timeout whenever more payload arrives, so a
                    // slow but healthy paste is not truncated.
                    if self.wait_fd(PARTIAL_WAIT)? {
                        *deadline = Instant::now() + PASTE_TIMEOUT;
                    }
                    continue;
                }
                paste_deadline = None;

                match parse(&self.buf) {
                    Parse::Event(parsed, len) => {
                        self.buf.drain(..len);
                        return Ok(parsed_event(parsed));
                    }
                    Parse::NeedMore => {
                        // A lone ESC is the Escape key: deliver it after the
                        // short partial wait that distinguishes it from the
                        // start of an escape sequence.
                        if self.buf.len() == 1 && self.buf[0] == 0x1b {
                            if !self.wait_fd(PARTIAL_WAIT)? {
                                self.buf.clear();
                                return Ok(parsed_event(key_event(
                                    KeyCode::Esc,
                                    KeyModifiers::NONE,
                                )));
                            }
                            continue;
                        }

                        // Any other incomplete sequence may still be completed
                        // by a slow terminal (ssh, a paste marker split across
                        // packets), so keep waiting for it. But never forever,
                        // and never turn it into Esc (which cancels the TUI) or
                        // into replacement text (which leaves garbage in the
                        // query): drop it as an ignored key instead.
                        let entry = incomplete.get_or_insert((self.buf.len(), Instant::now()));
                        if entry.0 != self.buf.len() {
                            *entry = (self.buf.len(), Instant::now());
                        }
                        if entry.1.elapsed() >= INCOMPLETE_TIMEOUT {
                            self.buf.clear();
                            return Ok(parsed_event(null_event()));
                        }
                        let _ = self.wait_fd(PARTIAL_WAIT)?;
                    }
                }
            }
        }

        fn wait_fd(&mut self, timeout: Duration) -> io::Result<bool> {
            let timeout_ms = timeout.as_millis().min(i32::MAX as u128) as i32;
            let mut pfd = libc::pollfd {
                fd: self.file.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            loop {
                let rc = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
                if rc >= 0 {
                    return Ok(rc > 0 && (pfd.revents & libc::POLLIN) != 0);
                }
                let err = io::Error::last_os_error();
                if err.kind() != io::ErrorKind::Interrupted {
                    return Err(err);
                }
            }
        }

        fn drain(&mut self) -> io::Result<usize> {
            let mut chunk = [0u8; 256];
            let mut total = 0;
            loop {
                match self.file.read(&mut chunk) {
                    Ok(0) => return Ok(total),
                    Ok(n) => {
                        self.buf.extend_from_slice(&chunk[..n]);
                        total += n;
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(total),
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            }
        }
    }

    fn parse(buf: &[u8]) -> Parse {
        let first = buf[0];
        match first {
            0x1b => parse_escape(buf),
            // 0x08 is Ctrl+H, not Backspace (0x7F); crossterm makes the same
            // distinction and `ctrl-h` is a bindable key.
            0x08 => Parse::Event(
                key_event(KeyCode::Char('h'), KeyModifiers::CONTROL),
                1,
            ),
            0x7f => Parse::Event(key_event(KeyCode::Backspace, KeyModifiers::NONE), 1),
            0x09 => Parse::Event(key_event(KeyCode::Tab, KeyModifiers::NONE), 1),
            0x0a => Parse::Event(key_event(KeyCode::Char('j'), KeyModifiers::CONTROL), 1),
            0x0d => Parse::Event(key_event(KeyCode::Enter, KeyModifiers::NONE), 1),
            0x00 => Parse::Event(key_event(KeyCode::Char(' '), KeyModifiers::CONTROL), 1),
            0x01..=0x1a => Parse::Event(
                key_event(
                    KeyCode::Char((b'a' + (first - 1)) as char),
                    KeyModifiers::CONTROL,
                ),
                1,
            ),
            // Ctrl+4..Ctrl+7 (0x1C..0x1F), same mapping as crossterm.
            0x1c..=0x1f => Parse::Event(
                key_event(
                    KeyCode::Char((b'4' + (first - 0x1c)) as char),
                    KeyModifiers::CONTROL,
                ),
                1,
            ),
            0x20..=0x7e => Parse::Event(char_event(first as char, KeyModifiers::NONE), 1),
            0x80..=0xff => parse_utf8(buf),
        }
    }

    /// Decode the Kitty/CSI-u modifier mask (`1` means "no modifiers").
    fn key_modifiers(mask: u8) -> KeyModifiers {
        let bits = mask.saturating_sub(1);
        let mut modifiers = KeyModifiers::NONE;
        if bits & 0b0000_0001 != 0 {
            modifiers |= KeyModifiers::SHIFT;
        }
        if bits & 0b0000_0010 != 0 {
            modifiers |= KeyModifiers::ALT;
        }
        if bits & 0b0000_0100 != 0 {
            modifiers |= KeyModifiers::CONTROL;
        }
        if bits & 0b0000_1000 != 0 {
            modifiers |= KeyModifiers::SUPER;
        }
        if bits & 0b0001_0000 != 0 {
            modifiers |= KeyModifiers::HYPER;
        }
        if bits & 0b0010_0000 != 0 {
            modifiers |= KeyModifiers::META;
        }
        modifiers
    }

    /// Modifiers from the second parameter of a CSI/SS3 list, used by
    /// `ESC [ 1 ; 5 A` and `ESC [ 3 ; 5 ~`. `1` means "no modifiers".
    fn csi_modifiers(seq: &[u8]) -> KeyModifiers {
        let Ok(text) = std::str::from_utf8(seq) else {
            return KeyModifiers::NONE;
        };
        let mut parts = text.split(';');
        let _first = parts.next();
        parts
            .next()
            .and_then(|second| second.split(':').next())
            .and_then(|value| value.parse::<u8>().ok())
            .map(key_modifiers)
            .unwrap_or(KeyModifiers::NONE)
    }

    /// Like [`csi_modifiers`], but also accepts the bare `ESC [ 5 A` spelling
    /// where the only parameter is the modifier mask (crossterm behavior).
    fn csi_modifiers_allow_bare(seq: &[u8]) -> KeyModifiers {
        let Ok(text) = std::str::from_utf8(seq) else {
            return KeyModifiers::NONE;
        };
        let mut parts = text.split(';');
        let first = parts.next().unwrap_or("");
        let mask = match parts.next() {
            Some(second) => second.split(':').next().and_then(|v| v.parse::<u8>().ok()),
            None => first.parse::<u8>().ok(),
        };
        mask.map(key_modifiers).unwrap_or(KeyModifiers::NONE)
    }

    fn parse_escape(buf: &[u8]) -> Parse {
        debug_assert_eq!(buf[0], 0x1b);
        if buf.len() < 2 {
            return Parse::NeedMore;
        }

        match buf[1] {
            b'[' => parse_csi(buf),
            b'O' => parse_ss3(buf),
            // A second ESC: report the first one, like crossterm.
            0x1b => Parse::Event(key_event(KeyCode::Esc, KeyModifiers::NONE), 1),
            // `ESC <key>` is the legacy "Alt/Meta sends escape" encoding, and
            // what terminals without the Kitty protocol send for Option/Alt:
            // decode the rest and mark it Alt. Without this, Alt+key looked
            // like Esc and cancelled the TUI.
            _ => match parse(&buf[1..]) {
                Parse::Event(Parsed::Key(event), len) => Parse::Event(
                    Parsed::Key(KeyEvent::new(
                        event.code,
                        event.modifiers | KeyModifiers::ALT,
                    )),
                    len + 1,
                ),
                Parse::Event(other, len) => Parse::Event(other, len + 1),
                Parse::NeedMore => Parse::NeedMore,
            },
        }
    }

    fn parse_csi(buf: &[u8]) -> Parse {
        debug_assert!(buf.starts_with(b"\x1b["));

        // X10 mouse: ESC [ M Cb Cx Cy.
        if buf.starts_with(b"\x1b[M") {
            if buf.len() < 6 {
                return Parse::NeedMore;
            }
            let cb = buf[3].saturating_sub(32);
            let (kind, modifiers) = parse_cb(cb);
            let column = u16::from(buf[4].saturating_sub(32)).saturating_sub(1);
            let row = u16::from(buf[5].saturating_sub(32)).saturating_sub(1);
            return Parse::Event(
                Parsed::Mouse(MouseEvent {
                    kind,
                    column,
                    row,
                    modifiers,
                }),
                6,
            );
        }

        // SGR mouse: ESC [ < Cb ; Cx ; Cy (M|m).
        if buf.starts_with(b"\x1b[<") {
            let Some(offset) = buf[3..]
                .iter()
                .take(MAX_SEQ)
                .position(|&b| b == b'M' || b == b'm')
            else {
                return if buf.len() >= 3 + MAX_SEQ {
                    // Malformed/oversized report: consume it instead of
                    // buffering without bound or leaking it as text.
                    Parse::Event(null_event(), 3 + MAX_SEQ)
                } else {
                    Parse::NeedMore
                };
            };
            let final_idx = offset + 3;
            let len = final_idx + 1;
            let Some(params) = std::str::from_utf8(&buf[3..final_idx]).ok() else {
                return Parse::Event(null_event(), len);
            };
            let release = buf[final_idx] == b'm';
            let mut parts = params.split(';');
            let (Some(cb), Some(column), Some(row)) = (
                parts.next().and_then(|v| v.parse::<u8>().ok()),
                parts.next().and_then(|v| v.parse::<u16>().ok()),
                parts.next().and_then(|v| v.parse::<u16>().ok()),
            ) else {
                return Parse::Event(null_event(), len);
            };

            let (kind, modifiers) = parse_cb(cb);
            let kind = if release {
                match kind {
                    MouseEventKind::Down(button) => MouseEventKind::Up(button),
                    other => other,
                }
            } else {
                kind
            };
            return Parse::Event(
                Parsed::Mouse(MouseEvent {
                    kind,
                    column: column.saturating_sub(1),
                    row: row.saturating_sub(1),
                    modifiers,
                }),
                len,
            );
        }

        // Legacy CSI: `ESC [ parameters final`.
        let Some(offset) = buf[2..]
            .iter()
            .take(MAX_SEQ)
            .position(|&b| (0x40..=0x7e).contains(&b))
        else {
            return if buf.len() >= 2 + MAX_SEQ {
                // Oversized sequence: consume it as one ignored key instead of
                // spilling `[` and the parameter bytes into the query.
                Parse::Event(null_event(), 2 + MAX_SEQ)
            } else {
                Parse::NeedMore
            };
        };

        let final_idx = offset + 2;
        let seq = &buf[2..final_idx];
        let len = final_idx + 1;

        // CSI-u / Kitty keyboard protocol.
        if buf[final_idx] == b'u' {
            let (code, modifiers) = parse_csi_u(seq);
            return Parse::Event(key_event(code, modifiers), len);
        }

        let code = match buf[final_idx] {
            b'A' => KeyCode::Up,
            b'B' => KeyCode::Down,
            b'C' => KeyCode::Right,
            b'D' => KeyCode::Left,
            b'H' => KeyCode::Home,
            b'F' => KeyCode::End,
            b'P' => KeyCode::F(1),
            b'Q' => KeyCode::F(2),
            b'S' => KeyCode::F(4),
            b'Z' => KeyCode::BackTab,
            b'~' => match csi_tilde_key(seq) {
                Some(code) => code,
                None => return Parse::Event(null_event(), len),
            },
            // Unknown but complete CSI sequence: consume it without treating
            // it as Esc (which would cancel the TUI).
            _ => KeyCode::Null,
        };
        // `~` sequences carry the key number first, so modifiers can only be
        // the second parameter (`ESC [ 11 ~` is F1, not "modifier 11"). The
        // letter forms also accept the bare `ESC [ 5 A` spelling.
        let modifiers = if buf[final_idx] == b'~' {
            csi_modifiers(seq)
        } else {
            csi_modifiers_allow_bare(seq)
        };
        Parse::Event(key_event(code, modifiers), len)
    }

    /// SS3 (`ESC O`) sequences: application cursor keys and F1-F4, optionally
    /// with `ESC O 1 ; modifiers final` on terminals that report modifiers.
    fn parse_ss3(buf: &[u8]) -> Parse {
        debug_assert!(buf.starts_with(b"\x1bO"));
        if buf.len() < 3 {
            return Parse::NeedMore;
        }

        let Some(offset) = buf[2..]
            .iter()
            .take(MAX_SEQ)
            .position(|&b| (0x40..=0x7e).contains(&b))
        else {
            return if buf.len() >= 2 + MAX_SEQ {
                Parse::Event(null_event(), 2 + MAX_SEQ)
            } else {
                Parse::NeedMore
            };
        };

        let final_idx = offset + 2;
        let seq = &buf[2..final_idx];
        let modifiers = csi_modifiers_allow_bare(seq);
        let code = match buf[final_idx] {
            b'A' => KeyCode::Up,
            b'B' => KeyCode::Down,
            b'C' => KeyCode::Right,
            b'D' => KeyCode::Left,
            b'H' => KeyCode::Home,
            b'F' => KeyCode::End,
            b'P' => KeyCode::F(1),
            b'Q' => KeyCode::F(2),
            b'R' => KeyCode::F(3),
            b'S' => KeyCode::F(4),
            _ => KeyCode::Null,
        };
        Parse::Event(key_event(code, modifiers), final_idx + 1)
    }

    /// Map the numeric parameter of a legacy `ESC [ n ~` sequence, mirroring
    /// crossterm (including the function-key codes).
    fn csi_tilde_key(seq: &[u8]) -> Option<KeyCode> {
        let first = std::str::from_utf8(seq)
            .ok()?
            .split(';')
            .next()?
            .parse::<u16>()
            .ok()?;
        Some(match first {
            1 | 7 => KeyCode::Home,
            2 => KeyCode::Insert,
            3 => KeyCode::Delete,
            4 | 8 => KeyCode::End,
            5 => KeyCode::PageUp,
            6 => KeyCode::PageDown,
            v @ 11..=15 => KeyCode::F((v - 10) as u8),
            v @ 17..=21 => KeyCode::F((v - 11) as u8),
            v @ 23..=26 => KeyCode::F((v - 12) as u8),
            v @ 28..=29 => KeyCode::F((v - 15) as u8),
            v @ 31..=34 => KeyCode::F((v - 17) as u8),
            _ => return None,
        })
    }

    fn parse_utf8(buf: &[u8]) -> Parse {
        let lead = buf[0];
        let expected = match lead {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            // Continuation bytes, overlong leads (0xC0/0xC1) and invalid leads
            // (0xF5..=0xFF) can never start a character.
            _ => return Parse::Event(replacement_event(), 1),
        };

        // Reject as soon as a byte that must be a continuation byte is not, so
        // malformed input does not stall waiting for bytes that will not come.
        for &byte in &buf[1..expected.min(buf.len())] {
            if byte & 0b1100_0000 != 0b1000_0000 {
                return Parse::Event(replacement_event(), 1);
            }
        }

        if buf.len() < expected {
            return Parse::NeedMore;
        }

        match std::str::from_utf8(&buf[..expected]) {
            Ok(text) => {
                let ch = text.chars().next().expect("valid UTF-8 is non-empty");
                Parse::Event(char_event(ch, KeyModifiers::NONE), expected)
            }
            Err(_) => Parse::Event(replacement_event(), 1),
        }
    }

    /// Decode a `CSI codepoint ; modifiers u` sequence (CSI-u / Kitty keyboard
    /// protocol). Upstream enables this protocol, so the legacy byte parser
    /// alone is not enough on terminals that support it.
    ///
    /// The decoder follows crossterm's `parse_csi_u_encoded_key_code` (the
    /// event model the upstream TUI is written against) and the functional-key
    /// codepoints from <https://sw.kovidgoyal.net/kitty/keyboard-protocol/#functional-key-definitions>
    /// (cross-checked against Ghostty's `src/input/kitty.zig`, itself ported
    /// from foot's `kitty-keymap.h`).
    fn parse_csi_u(seq: &[u8]) -> (KeyCode, KeyModifiers) {
        let ignored = (KeyCode::Null, KeyModifiers::NONE);
        let Ok(text) = std::str::from_utf8(seq) else {
            return ignored;
        };

        let mut fields = text.split(';');
        let mut codepoints = fields.next().unwrap_or("").split(':');
        let Some(codepoint) = codepoints.next().and_then(|value| value.parse::<u32>().ok()) else {
            return ignored;
        };
        let mut modifier_parts = fields.next().unwrap_or("").split(':');
        let modifier_mask = modifier_parts
            .next()
            .and_then(|value| value.parse::<u8>().ok())
            .unwrap_or(1);
        // Kitty sends key releases only when "report event types" is enabled
        // (flag 2, which upstream does not request). If a terminal sends one
        // anyway, the corresponding press was already handled, so drop it
        // instead of inserting the character a second time.
        let event_type = modifier_parts
            .next()
            .and_then(|value| value.parse::<u8>().ok())
            .unwrap_or(1);
        if event_type == 3 {
            return ignored;
        }

        let mut modifiers = key_modifiers(modifier_mask);
        let shift = modifiers.contains(KeyModifiers::SHIFT);
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);
        let alternate = codepoints
            .next()
            .and_then(|value| value.parse::<u32>().ok())
            .and_then(char::from_u32);

        let code = match codepoint {
            8 | 127 => KeyCode::Backspace,
            9 => {
                if shift {
                    KeyCode::BackTab
                } else {
                    KeyCode::Tab
                }
            }
            // Kitty reports Ctrl+J as codepoint 10 with no modifier; keep the
            // same normalized key the legacy parser produces for 0x0a.
            10 => {
                modifiers |= KeyModifiers::CONTROL;
                KeyCode::Char('j')
            }
            13 | 57414 => KeyCode::Enter,
            27 => KeyCode::Esc,
            57349 | 57426 => KeyCode::Delete,
            57350 | 57417 => KeyCode::Left,
            57351 | 57418 => KeyCode::Right,
            57352 | 57419 => KeyCode::Up,
            57353 | 57420 => KeyCode::Down,
            57354 | 57421 => KeyCode::PageUp,
            57355 | 57422 => KeyCode::PageDown,
            57356 | 57423 => KeyCode::Home,
            57357 | 57424 => KeyCode::End,
            57425 => KeyCode::Insert,
            // Keypad digits and symbols: crossterm turns these functional codes
            // into text, so a numeric keypad keeps working while
            // REPORT_ALL_KEYS_AS_ESCAPE_CODES is enabled.
            57399..=57408 => KeyCode::Char((b'0' + (codepoint - 57399) as u8) as char),
            57409 => KeyCode::Char('.'),
            57410 => KeyCode::Char('/'),
            57411 => KeyCode::Char('*'),
            57412 => KeyCode::Char('-'),
            57413 => KeyCode::Char('+'),
            57415 => KeyCode::Char('='),
            57416 => KeyCode::Char(','),
            // Function keys. crossterm only translates F13+, but kitty reports
            // F1-F12 through the same functional range once
            // REPORT_ALL_KEYS_AS_ESCAPE_CODES is on, and the keymap can bind
            // them. Mapping them beats emitting a private-use character.
            57364..=57398 => KeyCode::F((codepoint - 57363) as u8),
            57428..=57440 => KeyCode::Media(match codepoint {
                57428 => MediaKeyCode::Play,
                57429 => MediaKeyCode::Pause,
                57430 => MediaKeyCode::PlayPause,
                57431 => MediaKeyCode::Reverse,
                57432 => MediaKeyCode::Stop,
                57433 => MediaKeyCode::FastForward,
                57434 => MediaKeyCode::Rewind,
                57435 => MediaKeyCode::TrackNext,
                57436 => MediaKeyCode::TrackPrevious,
                57437 => MediaKeyCode::Record,
                57438 => MediaKeyCode::LowerVolume,
                57439 => MediaKeyCode::RaiseVolume,
                _ => MediaKeyCode::MuteVolume,
            }),
            // Remaining functional keys (bare modifier presses such as Shift,
            // CapsLock, ScrollLock, NumLock, PrintScreen, Pause, Menu, ...) are
            // reported as codepoints in the BMP private-use area. Those carry no
            // text: running them through `char::from_u32` inserted a stray
            // private-use character into the query, so pressing Shift before
            // typing an uppercase letter added U+E061. Upstream ignores
            // `KeyCode::Null`.
            0xE000..=0xF8FF => KeyCode::Null,
            _ => {
                // The alternate ("shifted") codepoint only applies while Shift
                // is held; crossterm ignores it otherwise.
                let mut ch = char::from_u32(codepoint);
                if shift {
                    if let Some(shifted) = alternate {
                        ch = Some(shifted);
                    } else if !ctrl
                        && let Some(original) = ch
                        && original.is_ascii_lowercase()
                    {
                        // Terminals that do not report alternate keys still need
                        // Shift+a to produce 'A'.
                        ch = Some(original.to_ascii_uppercase());
                    }
                }

                let Some(ch) = ch else {
                    return ignored;
                };
                if ch.is_ascii_uppercase() {
                    modifiers |= KeyModifiers::SHIFT;
                }
                KeyCode::Char(ch)
            }
        };
        (code, modifiers)
    }

    /// Cb is the mouse report byte: button number in the low and high bits,
    /// dragging/modifier flags in between. Mirrors crossterm's `parse_cb`.
    fn parse_cb(cb: u8) -> (MouseEventKind, KeyModifiers) {
        let button_number = (cb & 0b0000_0011) | ((cb & 0b1100_0000) >> 4);
        let dragging = cb & 0b0010_0000 == 0b0010_0000;

        let kind = match (button_number, dragging) {
            (0, false) => MouseEventKind::Down(MouseButton::Left),
            (1, false) => MouseEventKind::Down(MouseButton::Middle),
            (2, false) => MouseEventKind::Down(MouseButton::Right),
            (0, true) => MouseEventKind::Drag(MouseButton::Left),
            (1, true) => MouseEventKind::Drag(MouseButton::Middle),
            (2, true) => MouseEventKind::Drag(MouseButton::Right),
            (3, false) => MouseEventKind::Up(MouseButton::Left),
            (3, true) | (4, true) | (5, true) => MouseEventKind::Moved,
            (4, false) => MouseEventKind::ScrollUp,
            (5, false) => MouseEventKind::ScrollDown,
            (6, false) => MouseEventKind::ScrollLeft,
            (7, false) => MouseEventKind::ScrollRight,
            // Unsupported button: report a kind the TUI ignores instead of an
            // error that would drop the event stream.
            _ => MouseEventKind::Moved,
        };

        let mut modifiers = KeyModifiers::empty();
        if cb & 0b0000_0100 != 0 {
            modifiers |= KeyModifiers::SHIFT;
        }
        if cb & 0b0000_1000 != 0 {
            modifiers |= KeyModifiers::ALT;
        }
        if cb & 0b0001_0000 != 0 {
            modifiers |= KeyModifiers::CONTROL;
        }

        (kind, modifiers)
    }

    fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() || haystack.len() < needle.len() {
            return None;
        }
        haystack.windows(needle.len()).position(|window| window == needle)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const NONE: KeyModifiers = KeyModifiers::NONE;
        const SHIFT: KeyModifiers = KeyModifiers::SHIFT;
        const ALT: KeyModifiers = KeyModifiers::ALT;
        const CTRL: KeyModifiers = KeyModifiers::CONTROL;

        fn ev(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
            KeyEvent::new(code, modifiers)
        }

        fn parsed(bytes: &[u8]) -> (Parsed, usize) {
            match parse(bytes) {
                Parse::Event(event, len) => (event, len),
                Parse::NeedMore => panic!("expected a complete event for {bytes:?}"),
            }
        }

        fn key(bytes: &[u8]) -> KeyEvent {
            match parsed(bytes).0 {
                Parsed::Key(key) => key,
                _ => panic!("expected a key event for {bytes:?}"),
            }
        }

        fn mouse(bytes: &[u8]) -> MouseEvent {
            match parsed(bytes).0 {
                Parsed::Mouse(mouse) => mouse,
                _ => panic!("expected a mouse event for {bytes:?}"),
            }
        }

        // ---- control bytes and ASCII -------------------------------------

        #[test]
        fn decodes_control_bytes() {
            assert_eq!(key(b"\r"), ev(KeyCode::Enter, NONE));
            assert_eq!(key(b"\n"), ev(KeyCode::Char('j'), CTRL));
            assert_eq!(key(b"\t"), ev(KeyCode::Tab, NONE));
            assert_eq!(key(b"\x7f"), ev(KeyCode::Backspace, NONE));
            // 0x08 is Ctrl+H, not Backspace (0x7F), like crossterm.
            assert_eq!(key(b"\x08"), ev(KeyCode::Char('h'), CTRL));
            // NUL is Ctrl+Space, 0x1C..=0x1F are Ctrl+4..Ctrl+7.
            assert_eq!(key(b"\x00"), ev(KeyCode::Char(' '), CTRL));
            assert_eq!(key(b"\x1a"), ev(KeyCode::Char('z'), CTRL));
            assert_eq!(key(b"\x1c"), ev(KeyCode::Char('4'), CTRL));
            assert_eq!(key(b"\x1d"), ev(KeyCode::Char('5'), CTRL));
            assert_eq!(key(b"\x1e"), ev(KeyCode::Char('6'), CTRL));
            assert_eq!(key(b"\x1f"), ev(KeyCode::Char('7'), CTRL));
        }

        #[test]
        fn decodes_printable_ascii() {
            assert_eq!(key(b"a"), ev(KeyCode::Char('a'), NONE));
            assert_eq!(key(b" "), ev(KeyCode::Char(' '), NONE));
            assert_eq!(key(b"~"), ev(KeyCode::Char('~'), NONE));
            // crossterm marks uppercase characters as shifted.
            assert_eq!(key(b"A"), ev(KeyCode::Char('A'), SHIFT));
        }

        // ---- UTF-8 --------------------------------------------------------

        #[test]
        fn decodes_utf8_of_every_length() {
            assert_eq!(key("é".as_bytes()), ev(KeyCode::Char('é'), NONE));
            assert_eq!(key("中".as_bytes()), ev(KeyCode::Char('中'), NONE));
            assert_eq!(key("😀".as_bytes()), ev(KeyCode::Char('😀'), NONE));
        }

        #[test]
        fn incomplete_utf8_requests_more_data() {
            for prefix in [
                &b"\xc3"[..],
                &b"\xe4"[..],
                &b"\xe4\xb8"[..],
                &b"\xf0"[..],
                &b"\xf0\x9f"[..],
                &b"\xf0\x9f\x98"[..],
            ] {
                assert!(
                    matches!(parse(prefix), Parse::NeedMore),
                    "expected NeedMore for {prefix:?}"
                );
            }
        }

        #[test]
        fn malformed_utf8_becomes_one_replacement_char() {
            // Invalid continuation byte: consume exactly one byte.
            let (event, len) = parsed(b"\xc3\x28");
            assert_eq!(event_key(event), ev(KeyCode::Char('\u{fffd}'), NONE));
            assert_eq!(len, 1);

            // A continuation byte or an overlong/invalid lead is rejected at
            // once (no waiting for bytes that will never complete it).
            for invalid in [
                &b"\x80"[..],
                &b"\xc0\xaf"[..],
                &b"\xc1\xbf"[..],
                &b"\xf5\x80\x80\x80"[..],
                &b"\xf8\x88\x80\x80\x80"[..],
                &b"\xff"[..],
            ] {
                assert!(
                    matches!(parse(invalid), Parse::Event(_, 1)),
                    "expected an immediate 1-byte replacement for {invalid:?}"
                );
                assert_eq!(key(invalid), ev(KeyCode::Char('\u{fffd}'), NONE));
            }

            // A structurally valid but unencodable character (surrogate).
            assert_eq!(key(b"\xed\xa0\x80"), ev(KeyCode::Char('\u{fffd}'), NONE));
            // Out-of-range (beyond U+10FFFF).
            assert_eq!(key(b"\xf4\x90\x80\x80"), ev(KeyCode::Char('\u{fffd}'), NONE));
        }

        // ---- ESC / Alt ----------------------------------------------------

        #[test]
        fn lone_esc_requests_more_data() {
            assert!(matches!(parse(b"\x1b"), Parse::NeedMore));
        }

        #[test]
        fn legacy_alt_prefix_sets_alt_instead_of_cancelling() {
            // `ESC <key>` is what terminals without the Kitty protocol send for
            // Alt/Option. It must not be decoded as Escape (which exits the
            // search) followed by a stray character.
            assert_eq!(key(b"\x1bb"), ev(KeyCode::Char('b'), ALT));
            assert_eq!(parsed(b"\x1bb").1, 2);
            assert_eq!(key(b"\x1b1"), ev(KeyCode::Char('1'), ALT));
            assert_eq!(key(b"\x1b\x7f"), ev(KeyCode::Backspace, ALT));
            assert_eq!(key(b"\x1b\r"), ev(KeyCode::Enter, ALT));
            assert_eq!(key(b"\x1b\t"), ev(KeyCode::Tab, ALT));
            // Uppercase letters carry SHIFT as well.
            assert_eq!(key(b"\x1bH"), ev(KeyCode::Char('H'), ALT | SHIFT));
            // Alt + multi-byte UTF-8.
            assert_eq!(key("\x1bé".as_bytes()), ev(KeyCode::Char('é'), ALT));
            assert_eq!(parsed("\x1bé".as_bytes()).1, 3);
        }

        #[test]
        fn double_escape_reports_esc() {
            assert_eq!(key(b"\x1b\x1b"), ev(KeyCode::Esc, NONE));
            assert_eq!(parsed(b"\x1b\x1b").1, 1);
        }

        #[test]
        fn incomplete_alt_sequence_requests_more_data() {
            assert!(matches!(parse(b"\x1b\xc3"), Parse::NeedMore));
            assert!(matches!(parse(b"\x1b[<0;1"), Parse::NeedMore));
        }

        // ---- legacy CSI / SS3 --------------------------------------------

        #[test]
        fn decodes_legacy_csi_keys() {
            assert_eq!(key(b"\x1b[A"), ev(KeyCode::Up, NONE));
            assert_eq!(key(b"\x1b[B"), ev(KeyCode::Down, NONE));
            assert_eq!(key(b"\x1b[C"), ev(KeyCode::Right, NONE));
            assert_eq!(key(b"\x1b[D"), ev(KeyCode::Left, NONE));
            assert_eq!(key(b"\x1b[H"), ev(KeyCode::Home, NONE));
            assert_eq!(key(b"\x1b[F"), ev(KeyCode::End, NONE));
            assert_eq!(key(b"\x1b[Z"), ev(KeyCode::BackTab, NONE));
            assert_eq!(key(b"\x1b[2~"), ev(KeyCode::Insert, NONE));
            assert_eq!(key(b"\x1b[3~"), ev(KeyCode::Delete, NONE));
            assert_eq!(key(b"\x1b[5~"), ev(KeyCode::PageUp, NONE));
            assert_eq!(key(b"\x1b[6~"), ev(KeyCode::PageDown, NONE));
        }

        #[test]
        fn legacy_csi_modifiers_are_preserved() {
            // These back shipped default bindings (ctrl-left/right,
            // ctrl-delete); dropping the modifiers made them plain arrows.
            assert_eq!(key(b"\x1b[1;5A"), ev(KeyCode::Up, CTRL));
            assert_eq!(key(b"\x1b[1;2A"), ev(KeyCode::Up, SHIFT));
            assert_eq!(key(b"\x1b[1;3A"), ev(KeyCode::Up, ALT));
            assert_eq!(key(b"\x1b[1;6A"), ev(KeyCode::Up, SHIFT | CTRL));
            assert_eq!(key(b"\x1b[1;5D"), ev(KeyCode::Left, CTRL));
            assert_eq!(key(b"\x1b[1;5C"), ev(KeyCode::Right, CTRL));
            assert_eq!(key(b"\x1b[3;5~"), ev(KeyCode::Delete, CTRL));
            // Single-parameter form used by some terminals.
            assert_eq!(key(b"\x1b[5A"), ev(KeyCode::Up, CTRL));
        }

        #[test]
        fn decodes_function_keys() {
            assert_eq!(key(b"\x1bOP"), ev(KeyCode::F(1), NONE));
            assert_eq!(key(b"\x1bOQ"), ev(KeyCode::F(2), NONE));
            assert_eq!(key(b"\x1bOR"), ev(KeyCode::F(3), NONE));
            assert_eq!(key(b"\x1bOS"), ev(KeyCode::F(4), NONE));
            assert_eq!(key(b"\x1b[11~"), ev(KeyCode::F(1), NONE));
            assert_eq!(key(b"\x1b[15~"), ev(KeyCode::F(5), NONE));
            assert_eq!(key(b"\x1b[24~"), ev(KeyCode::F(12), NONE));
            assert_eq!(key(b"\x1b[15;2~"), ev(KeyCode::F(5), SHIFT));
        }

        #[test]
        fn decodes_ss3_keys() {
            assert_eq!(key(b"\x1bOA"), ev(KeyCode::Up, NONE));
            assert_eq!(key(b"\x1bOB"), ev(KeyCode::Down, NONE));
            assert_eq!(key(b"\x1bOC"), ev(KeyCode::Right, NONE));
            assert_eq!(key(b"\x1bOD"), ev(KeyCode::Left, NONE));
            assert_eq!(key(b"\x1bOH"), ev(KeyCode::Home, NONE));
            assert_eq!(key(b"\x1bOF"), ev(KeyCode::End, NONE));
        }

        #[test]
        fn unknown_csi_is_ignored_instead_of_esc() {
            assert_eq!(key(b"\x1b[I"), ev(KeyCode::Null, NONE)); // focus in
            assert_eq!(key(b"\x1b[O"), ev(KeyCode::Null, NONE)); // focus out
            assert_eq!(key(b"\x1b[?25h"), ev(KeyCode::Null, NONE)); // DEC reply
            assert_eq!(key(b"\x1b[?1u"), ev(KeyCode::Null, NONE)); // kitty flags reply
            assert_eq!(key(b"\x1b[>0u"), ev(KeyCode::Null, NONE));
            assert_eq!(key(b"\x1b[999~"), ev(KeyCode::Null, NONE));
        }

        #[test]
        fn incomplete_csi_requests_more_data() {
            for prefix in [
                &b"\x1b["[..],
                &b"\x1b[1"[..],
                &b"\x1b[1;"[..],
                &b"\x1b["[..],
                &b"\x1bO"[..],
                &b"\x1bO1;"[..],
            ] {
                assert!(
                    matches!(parse(prefix), Parse::NeedMore),
                    "expected NeedMore for {prefix:?}"
                );
            }
        }

        #[test]
        fn oversized_csi_is_consumed_as_one_ignored_key() {
            // Never spill `[` and the parameter bytes into the query as text.
            let mut seq = b"\x1b[".to_vec();
            seq.extend(std::iter::repeat_n(b'1', MAX_SEQ + 4));
            let (event, len) = parsed(&seq);
            assert_eq!(event_key(event), ev(KeyCode::Null, NONE));
            assert_eq!(len, 2 + MAX_SEQ);
        }

        // ---- CSI-u / Kitty keyboard protocol ------------------------------

        #[test]
        fn decodes_csi_u_characters() {
            assert_eq!(key(b"\x1b[97u"), ev(KeyCode::Char('a'), NONE));
            assert_eq!(key(b"\x1b[114;5u"), ev(KeyCode::Char('r'), CTRL));
            // The shipped default bindings: alt-1..9, alt-b/f/d/backspace.
            assert_eq!(key(b"\x1b[49;3u"), ev(KeyCode::Char('1'), ALT));
            assert_eq!(key(b"\x1b[98;3u"), ev(KeyCode::Char('b'), ALT));
            assert_eq!(key(b"\x1b[102;3u"), ev(KeyCode::Char('f'), ALT));
            assert_eq!(key(b"\x1b[100;3u"), ev(KeyCode::Char('d'), ALT));
            assert_eq!(key(b"\x1b[127;3u"), ev(KeyCode::Backspace, ALT));
            assert_eq!(key(b"\x1b[102;5u"), ev(KeyCode::Char('f'), CTRL));
            assert_eq!(key(b"\x1b[9u"), ev(KeyCode::Tab, NONE));
            assert_eq!(key(b"\x1b[9;2u"), ev(KeyCode::BackTab, SHIFT));
            assert_eq!(key(b"\x1b[13u"), ev(KeyCode::Enter, NONE));
            assert_eq!(key(b"\x1b[127u"), ev(KeyCode::Backspace, NONE));
            assert_eq!(key(b"\x1b[27u"), ev(KeyCode::Esc, NONE));
            // Ctrl+J normalization matches the legacy 0x0a path.
            assert_eq!(key(b"\x1b[10u"), ev(KeyCode::Char('j'), CTRL));
            // Super/Hyper/Meta are preserved (crossterm models them).
            assert_eq!(key(b"\x1b[97;9u"), ev(KeyCode::Char('a'), KeyModifiers::SUPER));
            assert_eq!(key(b"\x1b[97;17u"), ev(KeyCode::Char('a'), KeyModifiers::HYPER));
            assert_eq!(key(b"\x1b[97;33u"), ev(KeyCode::Char('a'), KeyModifiers::META));
        }

        #[test]
        fn csi_u_alternate_key_only_applies_with_shift() {
            // Kitty "report alternate keys": `97:65` is 'a' with shifted 'A'.
            assert_eq!(key(b"\x1b[97:65;2u"), ev(KeyCode::Char('A'), SHIFT));
            // Without SHIFT the alternate codepoint is ignored (crossterm).
            assert_eq!(key(b"\x1b[97:65;1u"), ev(KeyCode::Char('a'), NONE));
            // SHIFT without an alternate still produces the uppercase letter.
            assert_eq!(key(b"\x1b[97;2u"), ev(KeyCode::Char('A'), SHIFT));
            // Shifted symbol via the alternate.
            assert_eq!(key(b"\x1b[49:33;2u"), ev(KeyCode::Char('!'), SHIFT));
            // A shifted non-letter without an alternate keeps its modifiers.
            assert_eq!(key(b"\x1b[49;2u"), ev(KeyCode::Char('1'), SHIFT));
        }

        #[test]
        fn csi_u_special_keys_keep_modifiers() {
            assert_eq!(key(b"\x1b[57350;5u"), ev(KeyCode::Left, CTRL));
            assert_eq!(key(b"\x1b[57351;5u"), ev(KeyCode::Right, CTRL));
            assert_eq!(key(b"\x1b[57426;5u"), ev(KeyCode::Delete, CTRL));
            assert_eq!(key(b"\x1b[8;5u"), ev(KeyCode::Backspace, CTRL));
            assert_eq!(key(b"\x1b[57352u"), ev(KeyCode::Up, NONE));
            assert_eq!(key(b"\x1b[57425u"), ev(KeyCode::Insert, NONE));
            assert_eq!(key(b"\x1b[57423u"), ev(KeyCode::Home, NONE));
            assert_eq!(key(b"\x1b[57424u"), ev(KeyCode::End, NONE));
        }

        #[test]
        fn csi_u_ignores_bare_modifier_presses() {
            // Regression: these used to become private-use characters, so
            // pressing Shift inserted a stray glyph before an uppercase letter.
            for codepoint in 57441u32..=57454 {
                let seq = format!("\x1b[{codepoint}u");
                assert_eq!(
                    key(seq.as_bytes()),
                    ev(KeyCode::Null, NONE),
                    "codepoint {codepoint} must be ignored"
                );
            }
        }

        #[test]
        fn csi_u_ignores_locks_and_keypad_begin() {
            for codepoint in [57358u32, 57359, 57360, 57361, 57362, 57363, 57427] {
                let seq = format!("\x1b[{codepoint}u");
                assert_eq!(
                    key(seq.as_bytes()),
                    ev(KeyCode::Null, NONE),
                    "codepoint {codepoint} must be ignored"
                );
            }
        }

        #[test]
        fn csi_u_maps_keypad_function_and_media_keys() {
            assert_eq!(key(b"\x1b[57399u"), ev(KeyCode::Char('0'), NONE));
            assert_eq!(key(b"\x1b[57408u"), ev(KeyCode::Char('9'), NONE));
            assert_eq!(key(b"\x1b[57413u"), ev(KeyCode::Char('+'), NONE));
            assert_eq!(key(b"\x1b[57414u"), ev(KeyCode::Enter, NONE));
            assert_eq!(key(b"\x1b[57417u"), ev(KeyCode::Left, NONE));
            assert_eq!(key(b"\x1b[57364u"), ev(KeyCode::F(1), NONE));
            assert_eq!(key(b"\x1b[57375u"), ev(KeyCode::F(12), NONE));
            assert_eq!(key(b"\x1b[57376u"), ev(KeyCode::F(13), NONE));
            assert_eq!(key(b"\x1b[57398u"), ev(KeyCode::F(35), NONE));
            assert_eq!(
                key(b"\x1b[57428u"),
                ev(KeyCode::Media(MediaKeyCode::Play), NONE)
            );
            assert_eq!(
                key(b"\x1b[57440u"),
                ev(KeyCode::Media(MediaKeyCode::MuteVolume), NONE)
            );
        }

        #[test]
        fn csi_u_ignores_key_releases() {
            // Event types are not requested, but tolerate them if a terminal
            // sends them: a release must not insert the character again.
            assert_eq!(key(b"\x1b[97;1:3u"), ev(KeyCode::Null, NONE));
            assert_eq!(key(b"\x1b[57441;1:3u"), ev(KeyCode::Null, NONE));
            assert_eq!(key(b"\x1b[97;1:1u"), ev(KeyCode::Char('a'), NONE));
            assert_eq!(key(b"\x1b[97;1:2u"), ev(KeyCode::Char('a'), NONE));
        }

        #[test]
        fn csi_u_keeps_astral_plane_text() {
            // Emoji/CJK Extension B live above the BMP private-use area and are
            // still text, not functional keys.
            assert_eq!(key("\x1b[128512u".as_bytes()), ev(KeyCode::Char('😀'), NONE));
            assert_eq!(key("\x1b[131072u".as_bytes()), ev(KeyCode::Char('𠀀'), NONE));
        }

        #[test]
        fn malformed_csi_u_is_ignored() {
            for seq in [
                &b"\x1b[u"[..],
                &b"\x1b[;5u"[..],
                &b"\x1b[99999999999999999999u"[..],
                &b"\x1b[-1u"[..],
            ] {
                assert_eq!(key(seq), ev(KeyCode::Null, NONE), "sequence {seq:?}");
            }
        }

        // ---- mouse --------------------------------------------------------

        #[test]
        fn parses_sgr_mouse() {
            assert_eq!(
                mouse(b"\x1b[<0;1;2M").kind,
                MouseEventKind::Down(MouseButton::Left)
            );
            assert_eq!(
                mouse(b"\x1b[<0;1;2m").kind,
                MouseEventKind::Up(MouseButton::Left)
            );
            assert_eq!(
                mouse(b"\x1b[<32;5;6M").kind,
                MouseEventKind::Drag(MouseButton::Left)
            );
            assert_eq!(mouse(b"\x1b[<35;10;20M").kind, MouseEventKind::Moved);
            assert_eq!(mouse(b"\x1b[<64;1;2M").kind, MouseEventKind::ScrollUp);
            assert_eq!(mouse(b"\x1b[<65;1;2M").kind, MouseEventKind::ScrollDown);
            assert_eq!(mouse(b"\x1b[<66;1;2M").kind, MouseEventKind::ScrollLeft);
            assert_eq!(mouse(b"\x1b[<67;1;2M").kind, MouseEventKind::ScrollRight);
        }

        #[test]
        fn sgr_mouse_coordinates_are_one_based_and_unbounded() {
            let event = mouse(b"\x1b[<0;10;20M");
            assert_eq!((event.column, event.row), (9, 19));
            // SGR uses decimal numbers, so wide terminals work.
            let event = mouse(b"\x1b[<0;300;400M");
            assert_eq!((event.column, event.row), (299, 399));
        }

        #[test]
        fn sgr_mouse_modifiers_are_decoded() {
            // 4 = shift, 8 = alt, 16 = ctrl.
            assert_eq!(mouse(b"\x1b[<4;1;1M").modifiers, SHIFT);
            assert_eq!(mouse(b"\x1b[<8;1;1M").modifiers, ALT);
            assert_eq!(mouse(b"\x1b[<16;1;1M").modifiers, CTRL);
            assert_eq!(mouse(b"\x1b[<28;1;1M").modifiers, SHIFT | ALT | CTRL);
        }

        #[test]
        fn parses_x10_mouse() {
            // cb=0x20 -> left button, cx=0x21, cy=0x22 -> (0, 1).
            let event = mouse(b"\x1b[M\x20\x21\x22");
            assert_eq!(event.kind, MouseEventKind::Down(MouseButton::Left));
            assert_eq!((event.column, event.row), (0, 1));
            assert_eq!(parsed(b"\x1b[M\x20\x21\x22").1, 6);
        }

        #[test]
        fn incomplete_mouse_reports_request_more_data() {
            assert!(matches!(parse(b"\x1b[M\x20\x21"), Parse::NeedMore));
            assert!(matches!(parse(b"\x1b[<35;10"), Parse::NeedMore));
            assert!(matches!(parse(b"\x1b[<35;"), Parse::NeedMore));
        }

        #[test]
        fn malformed_mouse_reports_are_dropped() {
            for seq in [&b"\x1b[<x;1;2M"[..], &b"\x1b[<0;1M"[..]] {
                assert_eq!(key(seq), ev(KeyCode::Null, NONE), "sequence {seq:?}");
            }
            // Trailing parameters are ignored (crossterm does the same).
            assert_eq!(
                mouse(b"\x1b[<0;1;2;3M").kind,
                MouseEventKind::Down(MouseButton::Left)
            );
        }

        #[test]
        fn unsupported_mouse_buttons_do_not_drop_the_stream() {
            // crossterm rejects buttons it does not model; dropping the event
            // stream would be worse, so these map to `Moved` (ignored by the
            // TUI) instead.
            assert_eq!(mouse(b"\x1b[<128;1;1M").kind, MouseEventKind::Moved);
        }

        // ---- structural properties ---------------------------------------

        /// Every complete parse must consume at least one byte and never more
        /// than it was given.
        #[test]
        fn complete_sequences_consume_a_valid_prefix() {
            let corpus: &[&[u8]] = &[
                b"a",
                b"A",
                b"\r",
                b"\n",
                b"\t",
                b"\x00",
                b"\x1a",
                b"\x1bb",
                b"\xc3\xa9",
                b"\xe4\xb8\xad",
                b"\xf0\x9f\x98\x80",
                b"\xff",
                b"\x1b[A",
                b"\x1b[1;5A",
                b"\x1b[3~",
                b"\x1bOP",
                b"\x1bOA",
                b"\x1b[Z",
                b"\x1b[97u",
                b"\x1b[97;5u",
                b"\x1b[57441u",
                b"\x1b[<0;1;2M",
                b"\x1b[M\x20\x21\x22",
                b"\x1b[I",
                b"\x1b[?25h",
                b"\x1b[99999999999999999999u",
            ];
            for seq in corpus {
                match parse(seq) {
                    Parse::Event(_, len) => {
                        assert!(len >= 1 && len <= seq.len(), "bad len {len} for {seq:?}");
                    }
                    Parse::NeedMore => panic!("expected a complete parse for {seq:?}"),
                }
            }
        }

        /// For unambiguous sequences, no proper prefix may produce an event:
        /// this is what lets `read_event` wait for split reads safely.
        #[test]
        fn valid_sequences_never_emit_before_they_are_complete() {
            let sequences: &[&[u8]] = &[
                b"\xc3\xa9",
                b"\xe4\xb8\xad",
                b"\xf0\x9f\x98\x80",
                b"\x1b[A",
                b"\x1b[1;5A",
                b"\x1b[3~",
                b"\x1b[15;2~",
                b"\x1bOP",
                b"\x1bOA",
                b"\x1b[97u",
                b"\x1b[97;5u",
                b"\x1b[57441u",
                b"\x1b[<0;1;2M",
                b"\x1b[M\x20\x21\x22",
                b"\x1b\xc3\xa9",
                b"\x1bb",
            ];
            for seq in sequences {
                for end in 1..seq.len() {
                    let prefix = &seq[..end];
                    assert!(
                        matches!(parse(prefix), Parse::NeedMore),
                        "prefix {prefix:?} of {seq:?} parsed too early"
                    );
                }
                assert!(
                    matches!(parse(seq), Parse::Event(..)),
                    "full sequence {seq:?} did not parse"
                );
            }
        }

        /// The byte-level parser must never panic on arbitrary input and must
        /// never claim to consume more than it was given.
        #[test]
        fn fuzz_parser_never_panics_or_over_consumes() {
            // Deterministic xorshift so any failure is reproducible.
            let mut state = 0x2545_f491_4f6c_dd1du64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };

            for _ in 0..50_000 {
                let len = (next() % 32) as usize;
                let mut buf = Vec::with_capacity(len);
                for _ in 0..len {
                    buf.push((next() & 0xff) as u8);
                }
                if buf.is_empty() {
                    continue;
                }
                match parse(&buf) {
                    Parse::Event(_, consumed) => {
                        assert!(consumed >= 1, "consumed 0 bytes for {buf:?}");
                        assert!(consumed <= buf.len(), "over-consumed for {buf:?}");
                    }
                    Parse::NeedMore => {}
                }
            }
        }

        /// Feeding a valid sequence in arbitrarily split chunks (as `read()`
        /// does) must produce exactly the same event as one shot.
        #[test]
        fn split_reads_yield_the_same_event() {
            let sequences: &[&[u8]] = &[
                b"\xc3\xa9",
                b"\x1b[A",
                b"\x1b[1;5A",
                b"\x1b[97;5u",
                b"\x1b[57441u",
                b"\x1b[<0;1;2M",
                b"\x1b\xc3\xa9",
                b"\x1bb",
            ];
            for seq in sequences {
                let expected = match parse(seq) {
                    Parse::Event(parsed, len) => (format!("{parsed:?}"), len),
                    Parse::NeedMore => panic!("expected a complete parse for {seq:?}"),
                };

                for split in 1..seq.len() {
                    // The prefix on its own must ask for more data...
                    let mut buf = seq[..split].to_vec();
                    assert!(
                        matches!(parse(&buf), Parse::NeedMore),
                        "prefix of {seq:?} at {split} parsed early"
                    );
                    // ...and the joined buffer must decode identically.
                    buf.extend_from_slice(&seq[split..]);
                    match parse(&buf) {
                        Parse::Event(parsed, len) => {
                            assert_eq!((format!("{parsed:?}"), len), expected, "split {split}");
                        }
                        Parse::NeedMore => panic!("sequence {seq:?} did not parse when joined"),
                    }
                }
            }
        }

        fn event_key(event: Parsed) -> KeyEvent {
            match event {
                Parsed::Key(key) => key,
                _ => panic!("expected a key event"),
            }
        }
    }
}

#[cfg(windows)]
mod windows {
    use super::*;
    use ratatui::crossterm::event::{self, Event};

    pub fn poll(timeout: Duration) -> io::Result<bool> {
        event::poll(timeout)
    }

    pub fn read() -> io::Result<Event> {
        event::read()
    }

    /// No cached process-global terminal handle on Windows; nothing to drop.
    pub fn reset() {}
}
