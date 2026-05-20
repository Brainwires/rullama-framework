//! Console output capture for TUI
//!
//! Captures stderr/stdout output and makes it available to the TUI

#[cfg(test)]
use std::io::Write;
use std::sync::{Arc, Mutex};

/// Shared console buffer
#[derive(Clone)]
pub struct ConsoleBuffer {
    messages: Arc<Mutex<Vec<String>>>,
}

impl ConsoleBuffer {
    /// Create a new console buffer
    pub fn new() -> Self {
        Self {
            messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add a message to the console
    pub fn add_message(&self, msg: String) {
        if let Ok(mut messages) = self.messages.lock() {
            messages.push(msg);
        }
    }

    /// Get all console messages
    pub fn get_messages(&self) -> Vec<String> {
        self.messages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Clear all messages
    pub fn clear(&self) {
        if let Ok(mut messages) = self.messages.lock() {
            messages.clear();
        }
    }
}

/// Custom writer that captures output to console buffer (only used in tests)
#[cfg(test)]
pub struct ConsoleWriter {
    buffer: ConsoleBuffer,
    line_buffer: Arc<Mutex<String>>,
}

#[cfg(test)]
impl ConsoleWriter {
    pub fn new(buffer: ConsoleBuffer) -> Self {
        Self {
            buffer,
            line_buffer: Arc::new(Mutex::new(String::new())),
        }
    }
}

#[cfg(test)]
impl Write for ConsoleWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let s = String::from_utf8_lossy(buf);
        if let Ok(mut line_buf) = self.line_buffer.lock() {
            line_buf.push_str(&s);

            // Split by newlines and add complete lines to messages
            while let Some(pos) = line_buf.find('\n') {
                let line = line_buf[..pos].to_string();
                self.buffer.add_message(line);
                line_buf.drain(..=pos);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Flush any remaining content in line buffer
        if let Ok(mut line_buf) = self.line_buffer.lock()
            && !line_buf.is_empty()
        {
            self.buffer.add_message(line_buf.clone());
            line_buf.clear();
        }
        Ok(())
    }
}

impl Default for ConsoleBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_console_buffer_new() {
        let buffer = ConsoleBuffer::new();
        assert!(buffer.get_messages().is_empty());
    }

    #[test]
    fn test_console_buffer_default() {
        let buffer = ConsoleBuffer::default();
        assert!(buffer.get_messages().is_empty());
    }

    #[test]
    fn test_add_message() {
        let buffer = ConsoleBuffer::new();
        buffer.add_message("Hello".to_string());
        buffer.add_message("World".to_string());

        let messages = buffer.get_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0], "Hello");
        assert_eq!(messages[1], "World");
    }

    #[test]
    fn test_clear() {
        let buffer = ConsoleBuffer::new();
        buffer.add_message("Message 1".to_string());
        buffer.add_message("Message 2".to_string());

        assert_eq!(buffer.get_messages().len(), 2);

        buffer.clear();
        assert!(buffer.get_messages().is_empty());
    }

    #[test]
    fn test_console_writer() {
        let buffer = ConsoleBuffer::new();
        let mut writer = ConsoleWriter::new(buffer.clone());

        // Write some data
        writer.write_all(b"Line 1\n").unwrap();
        writer.write_all(b"Line 2\n").unwrap();

        let messages = buffer.get_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0], "Line 1");
        assert_eq!(messages[1], "Line 2");
    }

    #[test]
    fn test_console_writer_flush() {
        let buffer = ConsoleBuffer::new();
        let mut writer = ConsoleWriter::new(buffer.clone());

        // Write without newline
        writer.write_all(b"Partial line").unwrap();
        assert!(buffer.get_messages().is_empty());

        // Flush should add the partial line
        writer.flush().unwrap();
        let messages = buffer.get_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], "Partial line");
    }

    #[test]
    fn test_console_writer_multiple_lines_in_one_write() {
        let buffer = ConsoleBuffer::new();
        let mut writer = ConsoleWriter::new(buffer.clone());

        // Write multiple lines at once
        writer.write_all(b"Line 1\nLine 2\nLine 3\n").unwrap();

        let messages = buffer.get_messages();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0], "Line 1");
        assert_eq!(messages[1], "Line 2");
        assert_eq!(messages[2], "Line 3");
    }

    #[test]
    fn test_console_buffer_clone() {
        let buffer1 = ConsoleBuffer::new();
        buffer1.add_message("Test".to_string());

        let buffer2 = buffer1.clone();
        let messages = buffer2.get_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], "Test");

        // Add to buffer2, should affect buffer1 too (shared state)
        buffer2.add_message("Another".to_string());
        assert_eq!(buffer1.get_messages().len(), 2);
    }
}
