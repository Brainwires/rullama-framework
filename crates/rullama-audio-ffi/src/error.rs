//! Return codes, the last-error slot, and buffer copying for the C ABI.

use std::any::Any;
use std::sync::Mutex;

use crate::engine::lock;

/// Success.
pub const OK: i32 = 0;
/// Failure; message via `rla_last_error`.
pub const ERROR: i32 = -1;
/// Called in the wrong state.
pub const WRONG_STATE: i32 = -2;
/// No speech model loaded.
pub const NO_MODEL: i32 = -3;
/// No audio device.
pub const NO_DEVICE: i32 = -4;

/// Holds the most recent error message.
#[derive(Default)]
pub struct ErrorSlot(Mutex<String>);

impl ErrorSlot {
    /// An empty slot, usable in a `static`.
    pub const fn new_const() -> Self {
        Self(Mutex::new(String::new()))
    }

    /// Replace the message.
    pub fn set(&self, msg: impl Into<String>) {
        *lock(&self.0) = msg.into();
    }

    /// The current message (empty if none).
    pub fn get(&self) -> String {
        lock(&self.0).clone()
    }
}

/// Copy `s` into `buf` (truncated to `cap - 1` bytes, always NUL-terminated
/// when `cap > 0`) and return the full byte length of `s`.
///
/// Truncation never splits a UTF-8 character, so the copied prefix is valid UTF-8.
///
/// # Safety
/// `buf` must be null or valid for `cap` bytes of writes.
pub unsafe fn copy_out(s: &str, buf: *mut u8, cap: usize) -> i32 {
    if !buf.is_null() && cap > 0 {
        let mut n = s.len().min(cap - 1);
        while !s.is_char_boundary(n) {
            n -= 1;
        }
        // SAFETY: `buf` is valid for `cap > n` bytes per the caller's contract,
        // and `s` cannot overlap a buffer the caller hands us for writing.
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), buf, n);
            *buf.add(n) = 0;
        }
    }
    i32::try_from(s.len()).unwrap_or(i32::MAX)
}

/// Best-effort text of a panic payload.
pub fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(s: &str, cap: usize) -> (i32, Vec<u8>) {
        let mut buf = vec![0xAAu8; cap];
        let n = unsafe { copy_out(s, buf.as_mut_ptr(), cap) };
        (n, buf)
    }

    #[test]
    fn copies_with_room_to_spare() {
        let (n, buf) = copy("abc", 8);
        assert_eq!(n, 3);
        assert_eq!(&buf[..4], b"abc\0");
        assert_eq!(buf[4], 0xAA);
    }

    #[test]
    fn truncates_and_terminates() {
        let (n, buf) = copy("abcdef", 4);
        assert_eq!(n, 6);
        assert_eq!(&buf, b"abc\0");
        let (n, buf) = copy("abc", 3);
        assert_eq!(n, 3);
        assert_eq!(&buf, b"ab\0");
        let (n, buf) = copy("abc", 1);
        assert_eq!(n, 3);
        assert_eq!(&buf, b"\0");
    }

    #[test]
    fn null_or_zero_cap_only_reports_length() {
        assert_eq!(unsafe { copy_out("hello", std::ptr::null_mut(), 100) }, 5);
        let (n, buf) = copy("hello", 0);
        assert_eq!((n, buf.len()), (5, 0));
        assert_eq!(copy("", 4).0, 0);
    }

    #[test]
    fn truncation_keeps_utf8_valid() {
        // "é" is two bytes; cap 3 leaves room for 2 bytes: "a" + half of "é".
        let (n, buf) = copy("aé", 3);
        assert_eq!(n, 3);
        assert_eq!(&buf, b"a\0\xAA");
    }

    #[test]
    fn panic_payloads() {
        let p: Box<dyn Any + Send> = Box::new("static");
        assert_eq!(panic_message(&p), "static");
        let p: Box<dyn Any + Send> = Box::new(String::from("owned"));
        assert_eq!(panic_message(&p), "owned");
        let p: Box<dyn Any + Send> = Box::new(7u8);
        assert_eq!(panic_message(&p), "unknown panic");
    }
}
