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
        Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    };
    use std::fs::{File, OpenOptions};
    use std::io::Read;
    use std::os::fd::AsRawFd;
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    /// A private normalized key. Mirrors the small parser used by the FFI
    /// crate's previous TUI, kept here so no signal-handler-installing
    /// crossterm event code runs in the loaded library.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Key {
        Char(char),
        Up,
        Down,
        Left,
        Right,
        Home,
        End,
        PageUp,
        PageDown,
        Backspace,
        Delete,
        Tab,
        BackTab,
        Enter,
        Esc,
        /// A complete but unrecognized escape sequence. Upstream ignores
        /// `KeyCode::Null`, so unknown terminal reports (e.g. focus events or
        /// unsupported mouse buttons) cannot accidentally cancel the TUI.
        Null,
        Ctrl(char),
    }

    /// A fully parsed terminal event.
    enum Parsed {
        Key(Key),
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

    fn key_event(key: Key) -> KeyEvent {
        let (code, modifiers) = match key {
            Key::Char(c) => (KeyCode::Char(c), KeyModifiers::NONE),
            Key::Up => (KeyCode::Up, KeyModifiers::NONE),
            Key::Down => (KeyCode::Down, KeyModifiers::NONE),
            Key::Left => (KeyCode::Left, KeyModifiers::NONE),
            Key::Right => (KeyCode::Right, KeyModifiers::NONE),
            Key::Home => (KeyCode::Home, KeyModifiers::NONE),
            Key::End => (KeyCode::End, KeyModifiers::NONE),
            Key::PageUp => (KeyCode::PageUp, KeyModifiers::NONE),
            Key::PageDown => (KeyCode::PageDown, KeyModifiers::NONE),
            Key::Backspace => (KeyCode::Backspace, KeyModifiers::NONE),
            Key::Delete => (KeyCode::Delete, KeyModifiers::NONE),
            Key::Tab => (KeyCode::Tab, KeyModifiers::NONE),
            Key::BackTab => (KeyCode::BackTab, KeyModifiers::NONE),
            Key::Enter => (KeyCode::Enter, KeyModifiers::NONE),
            Key::Esc => (KeyCode::Esc, KeyModifiers::NONE),
            Key::Null => (KeyCode::Null, KeyModifiers::NONE),
            Key::Ctrl(c) => (KeyCode::Char(c), KeyModifiers::CONTROL),
        };
        KeyEvent::new(code, modifiers)
    }

    fn parsed_event(parsed: Parsed) -> Event {
        match parsed {
            Parsed::Key(key) => Event::Key(key_event(key)),
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

            loop {
                self.drain()?;
                if self.buf.is_empty() {
                    if !self.wait_fd(PARTIAL_WAIT)? {
                        return Ok(parsed_event(Parsed::Key(Key::Null)));
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
                    if Instant::now() >= *deadline {
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
                        if !self.wait_fd(PARTIAL_WAIT)? {
                            let fallback = if self.buf[0] == 0x1b {
                                Key::Esc
                            } else {
                                Key::Char('\u{fffd}')
                            };
                            self.buf.drain(..1);
                            return Ok(parsed_event(Parsed::Key(fallback)));
                        }
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

        fn drain(&mut self) -> io::Result<()> {
            let mut chunk = [0u8; 256];
            loop {
                match self.file.read(&mut chunk) {
                    Ok(0) => return Ok(()),
                    Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
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
            0x08 | 0x7f => Parse::Event(Parsed::Key(Key::Backspace), 1),
            0x09 => Parse::Event(Parsed::Key(Key::Tab), 1),
            0x0a => Parse::Event(Parsed::Key(Key::Ctrl('j')), 1),
            0x0d => Parse::Event(Parsed::Key(Key::Enter), 1),
            0x20..=0x7e => Parse::Event(Parsed::Key(Key::Char(first as char)), 1),
            0x01..=0x1a => {
                let letter = char::from(b'a' + (first - 1));
                Parse::Event(Parsed::Key(Key::Ctrl(letter)), 1)
            }
            0x80..=0xff => parse_utf8(buf),
            _ => Parse::Event(Parsed::Key(Key::Char('\u{fffd}')), 1),
        }
    }

    fn parse_escape(buf: &[u8]) -> Parse {
        debug_assert_eq!(buf[0], 0x1b);
        if buf.len() < 2 {
            return Parse::NeedMore;
        }

        match buf[1] {
            b'[' | b'O' => {
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
                    let Some(offset) = buf[3..].iter().position(|&b| b == b'M' || b == b'm')
                    else {
                        return Parse::NeedMore;
                    };
                    let final_idx = offset + 3;
                    let len = final_idx + 1;
                    let Some(params) = std::str::from_utf8(&buf[3..final_idx]).ok() else {
                        return Parse::Event(Parsed::Key(Key::Null), len);
                    };
                    let release = buf[final_idx] == b'm';
                    let mut parts = params.split(';');
                    let (Some(cb), Some(column), Some(row)) = (
                        parts.next().and_then(|v| v.parse::<u8>().ok()),
                        parts.next().and_then(|v| v.parse::<u16>().ok()),
                        parts.next().and_then(|v| v.parse::<u16>().ok()),
                    ) else {
                        return Parse::Event(Parsed::Key(Key::Null), len);
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

                const MAX_SEQ: usize = 32;
                let Some(offset) = buf[2..]
                    .iter()
                    .take(MAX_SEQ)
                    .position(|&b| (0x40..=0x7e).contains(&b))
                else {
                    return if buf.len() >= 2 + MAX_SEQ {
                        Parse::Event(Parsed::Key(Key::Esc), 1)
                    } else {
                        Parse::NeedMore
                    };
                };

                let final_idx = offset + 2;
                let seq = &buf[2..final_idx];

                // CSI-u / Kitty keyboard protocol. Upstream pushes
                // DISAMBIGUATE_ESCAPE_CODES | REPORT_ALL_KEYS_AS_ESCAPE_CODES,
                // so terminals supporting the protocol report every key this
                // way instead of as legacy bytes.
                if buf[final_idx] == b'u' {
                    return Parse::Event(Parsed::Key(parse_csi_u(seq)), final_idx + 1);
                }

                let key = match buf[final_idx] {
                    b'A' => Key::Up,
                    b'B' => Key::Down,
                    b'C' => Key::Right,
                    b'D' => Key::Left,
                    b'H' => Key::Home,
                    b'F' => Key::End,
                    b'Z' => Key::BackTab,
                    b'~' => {
                        let num = seq
                            .split(|&b| b == b';')
                            .next()
                            .and_then(|digits| std::str::from_utf8(digits).ok())
                            .and_then(|digits| digits.parse::<u32>().ok())
                            .unwrap_or(0);
                        match num {
                            1 | 7 => Key::Home,
                            3 => Key::Delete,
                            4 | 8 => Key::End,
                            5 => Key::PageUp,
                            6 => Key::PageDown,
                            _ => Key::Null,
                        }
                    }
                    // Unknown but complete CSI sequence: consume it without
                    // treating it as Esc (which would cancel the TUI).
                    _ => Key::Null,
                };
                Parse::Event(Parsed::Key(key), final_idx + 1)
            }
            _ => Parse::Event(Parsed::Key(Key::Esc), 1),
        }
    }

    fn parse_utf8(buf: &[u8]) -> Parse {
        let lead = buf[0];
        let expected = if lead >= 0xf0 {
            4
        } else if lead >= 0xe0 {
            3
        } else {
            2
        };

        if buf.len() < expected {
            return Parse::NeedMore;
        }

        match std::str::from_utf8(&buf[..expected]) {
            Ok(text) => {
                let ch = text.chars().next().expect("valid UTF-8 is non-empty");
                Parse::Event(Parsed::Key(Key::Char(ch)), expected)
            }
            Err(_) => Parse::Event(Parsed::Key(Key::Char('\u{fffd}')), 1),
        }
    }

    /// Decode a `CSI codepoint ; modifiers u` sequence (CSI-u / Kitty keyboard
    /// protocol). Upstream enables this protocol, so the legacy byte parser
    /// alone is not enough on terminals that support it.
    fn parse_csi_u(seq: &[u8]) -> Key {
        let Ok(text) = std::str::from_utf8(seq) else {
            return Key::Null;
        };

        let mut fields = text.split(';');
        let mut codepoints = fields.next().unwrap_or("").split(':');
        let Some(codepoint) = codepoints.next().and_then(|value| value.parse::<u32>().ok()) else {
            return Key::Null;
        };
        let modifier_mask = fields
            .next()
            .and_then(|field| field.split(':').next())
            .and_then(|value| value.parse::<u8>().ok())
            .unwrap_or(1);
        let modifier_bits = modifier_mask.saturating_sub(1);
        let shift = modifier_bits & 0b0000_0001 != 0;
        let ctrl = modifier_bits & 0b0000_0100 != 0;
        let alternate = codepoints
            .next()
            .and_then(|value| value.parse::<u32>().ok())
            .and_then(char::from_u32);

        match codepoint {
            8 | 127 => Key::Backspace,
            9 => {
                if shift {
                    Key::BackTab
                } else {
                    Key::Tab
                }
            }
            // Kitty reports Ctrl+J as codepoint 10 with no modifier; keep the
            // same normalized key the legacy parser produces for 0x0a.
            10 => Key::Ctrl('j'),
            13 => Key::Enter,
            27 => Key::Esc,
            57349 => Key::Delete,
            57350 => Key::Left,
            57351 => Key::Right,
            57352 => Key::Up,
            57353 => Key::Down,
            57354 => Key::PageUp,
            57355 => Key::PageDown,
            57356 => Key::Home,
            57357 => Key::End,
            _ => {
                let mut ch = alternate.or_else(|| char::from_u32(codepoint));
                if alternate.is_none()
                    && shift
                    && !ctrl
                    && let Some(original) = ch
                    && original.is_ascii_lowercase()
                {
                    ch = Some(original.to_ascii_uppercase());
                }

                let Some(ch) = ch else {
                    return Key::Null;
                };
                if ctrl {
                    Key::Ctrl(ch)
                } else {
                    Key::Char(ch)
                }
            }
        }
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

        fn mouse(parse: Parse) -> MouseEvent {
            match parse {
                Parse::Event(Parsed::Mouse(mouse), _) => mouse,
                _ => panic!("expected a mouse event"),
            }
        }

        fn key(parse: Parse) -> Key {
            match parse {
                Parse::Event(Parsed::Key(key), _) => key,
                _ => panic!("expected a key event"),
            }
        }

        #[test]
        fn parses_sgr_mouse_motion_as_moved() {
            let event = mouse(parse(b"\x1b[<35;10;20M"));
            assert_eq!(event.kind, MouseEventKind::Moved);
            assert_eq!(event.column, 9);
            assert_eq!(event.row, 19);
        }

        #[test]
        fn parses_sgr_scroll_and_release() {
            assert_eq!(mouse(parse(b"\x1b[<64;1;2M")).kind, MouseEventKind::ScrollUp);
            assert_eq!(mouse(parse(b"\x1b[<65;1;2M")).kind, MouseEventKind::ScrollDown);
            assert_eq!(
                mouse(parse(b"\x1b[<0;1;2m")).kind,
                MouseEventKind::Up(MouseButton::Left)
            );
        }

        #[test]
        fn parses_x10_mouse() {
            // cb=0x20 -> left button, cx=0x21, cy=0x22 -> (0, 1).
            let event = mouse(parse(b"\x1b[M\x20\x21\x22"));
            assert_eq!(event.kind, MouseEventKind::Down(MouseButton::Left));
            assert_eq!(event.column, 0);
            assert_eq!(event.row, 1);
        }

        #[test]
        fn unknown_csi_is_ignored_instead_of_esc() {
            // e.g. focus-in; must not cancel the TUI.
            assert_eq!(key(parse(b"\x1b[I")), Key::Null);
        }

        #[test]
        fn incomplete_mouse_report_requests_more_data() {
            assert!(matches!(parse(b"\x1b[<35;10"), Parse::NeedMore));
        }

        #[test]
        fn parses_csi_u_encoded_keys() {
            assert_eq!(key(parse(b"\x1b[97u")), Key::Char('a'));
            assert_eq!(key(parse(b"\x1b[114;5u")), Key::Ctrl('r'));
            assert_eq!(key(parse(b"\x1b[57352u")), Key::Up);
            assert_eq!(key(parse(b"\x1b[13u")), Key::Enter);
            assert_eq!(key(parse(b"\x1b[127u")), Key::Backspace);
        }

        #[test]
        fn csi_u_uses_alternate_key_for_shift() {
            // Kitty "report alternate keys": base 'a', alternate 'A'.
            assert_eq!(key(parse(b"\x1b[97:65;2u")), Key::Char('A'));
        }

        #[test]
        fn parses_shift_tab_as_backtab() {
            assert_eq!(key(parse(b"\x1b[Z")), Key::BackTab);
            assert_eq!(key(parse(b"\x1b[9;2u")), Key::BackTab);
            assert_eq!(key(parse(b"\x1b[9u")), Key::Tab);
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
