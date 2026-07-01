use serde::{Deserialize, Serialize};

pub(crate) const SAMPLE_RATE_SPEECH: u32 = 16_000;
pub(crate) const SAMPLE_RATE_CD: u32 = 44_100;
pub(crate) const SAMPLE_RATE_HIGH_QUALITY: u32 = 48_000;

/// Supported audio sample formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleFormat {
    /// 16-bit signed integer PCM (most common for speech).
    I16,
    /// 32-bit floating point PCM.
    F32,
}

/// Audio stream configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Sample rate in Hz (e.g., 16000, 44100, 48000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Sample format.
    pub sample_format: SampleFormat,
}

impl AudioConfig {
    /// Standard speech config: 16kHz mono 16-bit (Whisper, most STT APIs).
    pub fn speech() -> Self {
        Self {
            sample_rate: SAMPLE_RATE_SPEECH,
            channels: 1,
            sample_format: SampleFormat::I16,
        }
    }

    /// CD quality: 44.1kHz stereo 16-bit.
    pub fn cd_quality() -> Self {
        Self {
            sample_rate: SAMPLE_RATE_CD,
            channels: 2,
            sample_format: SampleFormat::I16,
        }
    }

    /// High quality: 48kHz stereo float.
    pub fn high_quality() -> Self {
        Self {
            sample_rate: SAMPLE_RATE_HIGH_QUALITY,
            channels: 2,
            sample_format: SampleFormat::F32,
        }
    }

    /// Bytes per sample for this format.
    pub fn bytes_per_sample(&self) -> usize {
        match self.sample_format {
            SampleFormat::I16 => 2,
            SampleFormat::F32 => 4,
        }
    }

    /// Bytes per frame (one sample per channel).
    pub fn bytes_per_frame(&self) -> usize {
        self.bytes_per_sample() * self.channels as usize
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self::speech()
    }
}

/// A chunk of raw audio data with its format metadata.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Raw PCM sample data (little-endian).
    pub data: Vec<u8>,
    /// Audio configuration describing the format of `data`.
    pub config: AudioConfig,
}

impl AudioBuffer {
    /// Create a new empty buffer with the given config.
    pub fn new(config: AudioConfig) -> Self {
        Self {
            data: Vec::new(),
            config,
        }
    }

    /// Create a buffer from raw PCM bytes.
    pub fn from_pcm(data: Vec<u8>, config: AudioConfig) -> Self {
        Self { data, config }
    }

    /// Duration of the audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        let frame_size = self.config.bytes_per_frame();
        if frame_size == 0 {
            return 0.0;
        }
        let num_frames = self.data.len() / frame_size;
        num_frames as f64 / self.config.sample_rate as f64
    }

    /// Number of frames in this buffer.
    pub fn num_frames(&self) -> usize {
        let frame_size = self.config.bytes_per_frame();
        if frame_size == 0 {
            0
        } else {
            self.data.len() / frame_size
        }
    }

    /// Whether this buffer contains no audio data.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Voice identifier for TTS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    /// Provider-specific voice identifier (e.g., "alloy", "echo", "shimmer").
    pub id: String,
    /// Human-readable display name.
    pub name: Option<String>,
    /// Language code (e.g., "en-US").
    pub language: Option<String>,
}

impl Voice {
    /// Create a new voice with the given identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            language: None,
        }
    }
}

/// Output audio format for TTS.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OutputFormat {
    /// WAV format.
    Wav,
    /// MP3 format.
    Mp3,
    /// Raw PCM samples.
    Pcm,
    /// Opus compressed format.
    Opus,
    /// FLAC lossless compressed format.
    Flac,
}

/// Options for text-to-speech generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsOptions {
    /// Voice to use.
    pub voice: Voice,
    /// Playback speed multiplier (0.25 to 4.0, default 1.0).
    pub speed: Option<f32>,
    /// Output audio format.
    pub output_format: OutputFormat,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            voice: Voice::new("alloy"),
            speed: None,
            output_format: OutputFormat::Wav,
        }
    }
}

/// Options for speech-to-text transcription.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SttOptions {
    /// Language hint (ISO-639-1 code, e.g., "en").
    pub language: Option<String>,
    /// Whether to include word-level timestamps.
    pub timestamps: bool,
    /// Optional prompt to guide the model.
    pub prompt: Option<String>,
}

/// Result of a speech-to-text transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    /// The transcribed text.
    pub text: String,
    /// Language detected or used.
    pub language: Option<String>,
    /// Duration of the audio in seconds.
    pub duration_secs: Option<f64>,
    /// Word-level segments with timestamps (if requested).
    pub segments: Vec<TranscriptSegment>,
}

/// A timed segment within a transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Segment text.
    pub text: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_config_values() {
        let cfg = AudioConfig::speech();
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.channels, 1);
        assert_eq!(cfg.sample_format, SampleFormat::I16);
    }

    #[test]
    fn cd_quality_config_values() {
        let cfg = AudioConfig::cd_quality();
        assert_eq!(cfg.sample_rate, 44100);
        assert_eq!(cfg.channels, 2);
        assert_eq!(cfg.sample_format, SampleFormat::I16);
    }

    #[test]
    fn high_quality_config_values() {
        let cfg = AudioConfig::high_quality();
        assert_eq!(cfg.sample_rate, 48000);
        assert_eq!(cfg.channels, 2);
        assert_eq!(cfg.sample_format, SampleFormat::F32);
    }

    #[test]
    fn bytes_per_sample_i16() {
        let cfg = AudioConfig::speech(); // I16
        assert_eq!(cfg.bytes_per_sample(), 2);
    }

    #[test]
    fn bytes_per_sample_f32() {
        let cfg = AudioConfig::high_quality(); // F32
        assert_eq!(cfg.bytes_per_sample(), 4);
    }

    #[test]
    fn bytes_per_frame_mono_i16() {
        let cfg = AudioConfig::speech(); // 1 channel, I16
        assert_eq!(cfg.bytes_per_frame(), 2); // 2 * 1
    }

    #[test]
    fn bytes_per_frame_stereo_f32() {
        let cfg = AudioConfig::high_quality(); // 2 channels, F32
        assert_eq!(cfg.bytes_per_frame(), 8); // 4 * 2
    }

    #[test]
    fn audio_buffer_new_is_empty() {
        let buf = AudioBuffer::new(AudioConfig::speech());
        assert!(buf.is_empty());
        assert_eq!(buf.num_frames(), 0);
    }

    #[test]
    fn audio_buffer_from_pcm_stores_data() {
        let data = vec![0u8; 64];
        let cfg = AudioConfig::speech();
        let buf = AudioBuffer::from_pcm(data.clone(), cfg);
        assert_eq!(buf.data, data);
        assert_eq!(buf.config.sample_rate, 16000);
        assert!(!buf.is_empty());
    }

    #[test]
    fn audio_buffer_num_frames() {
        // 16kHz mono I16 => 2 bytes per frame
        // 100 bytes => 50 frames
        let buf = AudioBuffer::from_pcm(vec![0u8; 100], AudioConfig::speech());
        assert_eq!(buf.num_frames(), 50);
    }

    #[test]
    fn audio_buffer_duration_secs() {
        // 16kHz mono I16 => 2 bytes/frame
        // 32000 bytes => 16000 frames => 1.0 second
        let buf = AudioBuffer::from_pcm(vec![0u8; 32000], AudioConfig::speech());
        assert!((buf.duration_secs() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn voice_new_sets_id_and_defaults() {
        let v = Voice::new("shimmer");
        assert_eq!(v.id, "shimmer");
        assert!(v.name.is_none());
        assert!(v.language.is_none());
    }

    #[test]
    fn output_format_debug_is_reasonable() {
        let dbg = format!("{:?}", OutputFormat::Wav);
        assert_eq!(dbg, "Wav");
        let dbg_mp3 = format!("{:?}", OutputFormat::Mp3);
        assert_eq!(dbg_mp3, "Mp3");
    }
}
