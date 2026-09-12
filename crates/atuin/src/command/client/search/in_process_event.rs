//! Terminal event source for the in-process interactive TUI.
//!
//! The CLI uses `crossterm::event`, which on Unix lazily installs a
//! process-wide SIGWINCH handler through signal-hook. That is fine for a
//! short-lived binary, but in a `dlclose`d shell module it leaves a dangling
//! function pointer behind and can crash the host shell on the next resize.
//!
//! This module therefore supplies the small `poll`/`read` interface used by
//! the TUI render loop without touching `crossterm::event` on Unix. Windows
//! console input uses no signal handlers, so the normal crossterm wrappers are
//! safe there.

use std::io;
use std::time::Duration;

#[cfg(unix)]
pub use unix::{poll, read};
#[cfg(windows)]
pub use windows::{poll, read};

#[cfg(unix)]
mod unix {
    use super::*;
    use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
    use std::fs::{File, OpenOptions};
    use std::io::Read;
    use std::os::fd::AsRawFd;
    use std::sync::{Mutex, OnceLock};

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
        Enter,
        Esc,
        Ctrl(char),
    }

    struct Input {
        file: File,
        buf: Vec<u8>,
    }

    enum Parse {
        Key(Key, usize),
        NeedMore,
    }

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

    pub fn poll(timeout: Duration) -> io::Result<bool> {
        with_input(|input| input.poll(timeout))
    }

    pub fn read() -> io::Result<Event> {
        with_input(|input| {
            loop {
                if let Some(key) = input.read_key()? {
                    return Ok(Event::Key(key_event(key)));
                }
                // `read` is only called after `poll` reported readiness. If a
                // partial sequence was consumed and no complete key is left,
                // wait briefly for the remainder.
                if !input.poll(Duration::from_millis(30))? {
                    return Ok(Event::Key(KeyEvent::new(KeyCode::Null, KeyModifiers::NONE)));
                }
            }
        })
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
            Key::Enter => (KeyCode::Enter, KeyModifiers::NONE),
            Key::Esc => (KeyCode::Esc, KeyModifiers::NONE),
            Key::Ctrl(c) => (KeyCode::Char(c), KeyModifiers::CONTROL),
        };
        KeyEvent::new(code, modifiers)
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

        fn read_key(&mut self) -> io::Result<Option<Key>> {
            loop {
                self.drain()?;
                if self.buf.is_empty() {
                    return Ok(None);
                }

                match parse(&self.buf) {
                    Parse::Key(key, len) => {
                        self.buf.drain(..len);
                        return Ok(Some(key));
                    }
                    Parse::NeedMore => {
                        if !self.wait_fd(Duration::from_millis(30))? {
                            let fallback = if self.buf[0] == 0x1b {
                                Key::Esc
                            } else {
                                Key::Char('\u{fffd}')
                            };
                            self.buf.drain(..1);
                            return Ok(Some(fallback));
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
            0x08 | 0x7f => Parse::Key(Key::Backspace, 1),
            0x09 => Parse::Key(Key::Tab, 1),
            0x0a => Parse::Key(Key::Ctrl('j'), 1),
            0x0d => Parse::Key(Key::Enter, 1),
            0x20..=0x7e => Parse::Key(Key::Char(first as char), 1),
            0x01..=0x1a => {
                let letter = char::from(b'a' + (first - 1));
                Parse::Key(Key::Ctrl(letter), 1)
            }
            0x80..=0xff => parse_utf8(buf),
            _ => Parse::Key(Key::Char('\u{fffd}'), 1),
        }
    }

    fn parse_escape(buf: &[u8]) -> Parse {
        debug_assert_eq!(buf[0], 0x1b);
        if buf.len() < 2 {
            return Parse::NeedMore;
        }

        match buf[1] {
            b'[' | b'O' => {
                const MAX_SEQ: usize = 32;
                let Some(offset) = buf[2..]
                    .iter()
                    .take(MAX_SEQ)
                    .position(|&b| (0x40..=0x7e).contains(&b))
                else {
                    return if buf.len() >= 2 + MAX_SEQ {
                        Parse::Key(Key::Esc, 1)
                    } else {
                        Parse::NeedMore
                    };
                };

                let final_idx = offset + 2;
                let seq = &buf[2..final_idx];
                let key = match buf[final_idx] {
                    b'A' => Key::Up,
                    b'B' => Key::Down,
                    b'C' => Key::Right,
                    b'D' => Key::Left,
                    b'H' => Key::Home,
                    b'F' => Key::End,
                    b'Z' => Key::Tab,
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
                            _ => Key::Esc,
                        }
                    }
                    _ => Key::Esc,
                };
                Parse::Key(key, final_idx + 1)
            }
            _ => Parse::Key(Key::Esc, 1),
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
                Parse::Key(Key::Char(ch), expected)
            }
            Err(_) => Parse::Key(Key::Char('\u{fffd}'), 1),
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
}
