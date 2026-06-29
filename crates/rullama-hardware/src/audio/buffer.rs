use crate::audio::types::AudioConfig;

/// A ring buffer for accumulating audio samples with a fixed capacity.
///
/// Useful for buffering streaming audio before processing (e.g., accumulating
/// enough audio for a STT inference pass).
pub struct AudioRingBuffer {
    data: Vec<u8>,
    capacity: usize,
    write_pos: usize,
    len: usize,
    config: AudioConfig,
}

impl AudioRingBuffer {
    /// Create a new ring buffer with capacity for `duration_secs` of audio.
    pub fn new(config: AudioConfig, duration_secs: f64) -> Self {
        let capacity =
            (config.sample_rate as f64 * duration_secs) as usize * config.bytes_per_frame();
        Self {
            data: vec![0u8; capacity],
            capacity,
            write_pos: 0,
            len: 0,
            config,
        }
    }

    /// Push raw PCM bytes into the buffer, overwriting oldest data if full.
    pub fn push(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.data[self.write_pos] = byte;
            self.write_pos = (self.write_pos + 1) % self.capacity;
            if self.len < self.capacity {
                self.len += 1;
            }
        }
    }

    /// Read all available data as a contiguous byte slice.
    ///
    /// Returns data in chronological order (oldest first).
    pub fn read_all(&self) -> Vec<u8> {
        if self.len < self.capacity {
            // Haven't wrapped yet
            self.data[..self.len].to_vec()
        } else {
            // Wrapped: read from write_pos to end, then from start to write_pos
            let mut result = Vec::with_capacity(self.capacity);
            result.extend_from_slice(&self.data[self.write_pos..]);
            result.extend_from_slice(&self.data[..self.write_pos]);
            result
        }
    }

    /// Duration of buffered audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        let frame_size = self.config.bytes_per_frame();
        if frame_size == 0 {
            return 0.0;
        }
        let num_frames = self.len / frame_size;
        num_frames as f64 / self.config.sample_rate as f64
    }

    /// Number of bytes currently buffered.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }

    /// Clear all buffered data.
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.len = 0;
    }

    /// Get the audio config for this buffer.
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let config = AudioConfig::speech(); // 16kHz mono i16 = 2 bytes/frame
        let mut buf = AudioRingBuffer::new(config, 0.001); // ~32 bytes

        assert!(buf.is_empty());
        assert_eq!(buf.duration_secs(), 0.0);

        buf.push(&[1, 2, 3, 4]);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.read_all(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let config = AudioConfig {
            sample_rate: 4,
            channels: 1,
            sample_format: crate::audio::types::SampleFormat::I16,
        };
        // 4 Hz * 1 sec * 2 bytes/frame = 8 bytes capacity
        let mut buf = AudioRingBuffer::new(config, 1.0);
        assert_eq!(buf.capacity(), 8);

        buf.push(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(buf.is_full());

        // Push more, should overwrite oldest
        buf.push(&[9, 10]);
        let data = buf.read_all();
        assert_eq!(data, vec![3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_ring_buffer_clear() {
        let config = AudioConfig::speech();
        let mut buf = AudioRingBuffer::new(config, 0.01);
        buf.push(&[1, 2, 3, 4]);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }
}
