//! FFI-safe mirror types for brainwires-hardware.
//!
//! These types are annotated with UniFFI derives so they can cross the Rust ↔ C#
//! (or Kotlin/Swift/Python) boundary. Each has `From` conversions to/from the
//! native brainwires-hardware equivalents.

use brainwires_hardware::{
    AudioBuffer, AudioConfig, AudioDevice, DeviceDirection, OutputFormat, SampleFormat, SttOptions,
    Transcript, TranscriptSegment, TtsOptions, Voice,
};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiSampleFormat {
    /// 16-bit signed integer.
    I16,
    /// 32-bit floating point.
    F32,
}

impl From<SampleFormat> for FfiSampleFormat {
    fn from(f: SampleFormat) -> Self {
        match f {
            SampleFormat::I16 => Self::I16,
            SampleFormat::F32 => Self::F32,
        }
    }
}

impl From<FfiSampleFormat> for SampleFormat {
    fn from(f: FfiSampleFormat) -> Self {
        match f {
            FfiSampleFormat::I16 => Self::I16,
            FfiSampleFormat::F32 => Self::F32,
        }
    }
}

/// Audio output format for TTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiOutputFormat {
    /// WAV container.
    Wav,
    /// MP3 compressed.
    Mp3,
    /// Raw PCM bytes.
    Pcm,
    /// Opus compressed.
    Opus,
    /// FLAC lossless.
    Flac,
}

impl From<OutputFormat> for FfiOutputFormat {
    fn from(f: OutputFormat) -> Self {
        match f {
            OutputFormat::Wav => Self::Wav,
            OutputFormat::Mp3 => Self::Mp3,
            OutputFormat::Pcm => Self::Pcm,
            OutputFormat::Opus => Self::Opus,
            OutputFormat::Flac => Self::Flac,
        }
    }
}

impl From<FfiOutputFormat> for OutputFormat {
    fn from(f: FfiOutputFormat) -> Self {
        match f {
            FfiOutputFormat::Wav => Self::Wav,
            FfiOutputFormat::Mp3 => Self::Mp3,
            FfiOutputFormat::Pcm => Self::Pcm,
            FfiOutputFormat::Opus => Self::Opus,
            FfiOutputFormat::Flac => Self::Flac,
        }
    }
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// Audio buffer — raw PCM data with metadata (flattened from AudioBuffer + AudioConfig).
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAudioBuffer {
    /// Raw audio bytes (PCM, little-endian).
    pub data: Vec<u8>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Sample format.
    pub sample_format: FfiSampleFormat,
}

impl From<AudioBuffer> for FfiAudioBuffer {
    fn from(b: AudioBuffer) -> Self {
        Self {
            data: b.data,
            sample_rate: b.config.sample_rate,
            channels: b.config.channels,
            sample_format: b.config.sample_format.into(),
        }
    }
}

impl From<FfiAudioBuffer> for AudioBuffer {
    fn from(b: FfiAudioBuffer) -> Self {
        Self {
            data: b.data,
            config: AudioConfig {
                sample_rate: b.sample_rate,
                channels: b.channels,
                sample_format: b.sample_format.into(),
            },
        }
    }
}

/// Voice identifier.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiVoice {
    /// Provider-specific voice ID.
    pub id: String,
    /// Human-readable name.
    pub name: Option<String>,
    /// ISO-639-1 language code.
    pub language: Option<String>,
}

impl From<Voice> for FfiVoice {
    fn from(v: Voice) -> Self {
        Self {
            id: v.id,
            name: v.name,
            language: v.language,
        }
    }
}

impl From<FfiVoice> for Voice {
    fn from(v: FfiVoice) -> Self {
        Self {
            id: v.id,
            name: v.name,
            language: v.language,
        }
    }
}

/// TTS synthesis options.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiTtsOptions {
    /// Voice ID to use.
    pub voice_id: String,
    /// Speech speed (0.25–4.0).
    pub speed: Option<f32>,
    /// Output audio format.
    pub output_format: FfiOutputFormat,
}

impl FfiTtsOptions {
    /// Convert to native TtsOptions.
    pub fn to_native(&self) -> TtsOptions {
        TtsOptions {
            voice: Voice {
                id: self.voice_id.clone(),
                name: None,
                language: None,
            },
            speed: self.speed,
            output_format: self.output_format.into(),
        }
    }
}

/// STT transcription options.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSttOptions {
    /// Language hint (ISO-639-1).
    pub language: Option<String>,
    /// Whether to include word-level timestamps.
    pub timestamps: bool,
    /// Prompt hint for the model.
    pub prompt: Option<String>,
}

impl From<FfiSttOptions> for SttOptions {
    fn from(o: FfiSttOptions) -> Self {
        Self {
            language: o.language,
            timestamps: o.timestamps,
            prompt: o.prompt,
        }
    }
}

/// Transcription segment with timestamps.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiTranscriptSegment {
    /// Segment text.
    pub text: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
}

impl From<TranscriptSegment> for FfiTranscriptSegment {
    fn from(s: TranscriptSegment) -> Self {
        Self {
            text: s.text,
            start: s.start,
            end: s.end,
        }
    }
}

/// Transcription result.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiTranscript {
    /// Full transcription text.
    pub text: String,
    /// Detected language.
    pub language: Option<String>,
    /// Audio duration in seconds.
    pub duration_secs: Option<f64>,
    /// Word-level segments (if timestamps requested).
    pub segments: Vec<FfiTranscriptSegment>,
}

impl From<Transcript> for FfiTranscript {
    fn from(t: Transcript) -> Self {
        Self {
            text: t.text,
            language: t.language,
            duration_secs: t.duration_secs,
            segments: t.segments.into_iter().map(Into::into).collect(),
        }
    }
}

/// Audio device descriptor.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAudioDevice {
    /// Platform-specific device ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Whether this is the default device.
    pub is_default: bool,
    /// Whether this is an input (capture) device.
    pub is_input: bool,
}

impl From<AudioDevice> for FfiAudioDevice {
    fn from(d: AudioDevice) -> Self {
        Self {
            id: d.id,
            name: d.name,
            is_default: d.is_default,
            is_input: matches!(d.direction, DeviceDirection::Input),
        }
    }
}

/// Provider info returned by `list_providers`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiProviderInfo {
    /// Provider identifier (e.g. "openai", "elevenlabs").
    pub name: String,
    /// Display name.
    pub display_name: String,
    /// Whether this provider supports TTS.
    pub has_tts: bool,
    /// Whether this provider supports STT.
    pub has_stt: bool,
    /// Whether this provider requires a `region` parameter.
    pub requires_region: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_hardware::{
        AudioBuffer, AudioConfig, AudioDevice, DeviceDirection, OutputFormat, SampleFormat,
        SttOptions, Transcript, TranscriptSegment, Voice,
    };

    // --- FfiSampleFormat ---

    #[test]
    fn sample_format_i16_roundtrip() {
        let native = SampleFormat::I16;
        let ffi: FfiSampleFormat = native.into();
        assert_eq!(ffi, FfiSampleFormat::I16);
        let back: SampleFormat = ffi.into();
        assert!(matches!(back, SampleFormat::I16));
    }

    #[test]
    fn sample_format_f32_roundtrip() {
        let ffi: FfiSampleFormat = SampleFormat::F32.into();
        assert_eq!(ffi, FfiSampleFormat::F32);
        let back: SampleFormat = ffi.into();
        assert!(matches!(back, SampleFormat::F32));
    }

    // --- FfiOutputFormat ---

    #[test]
    fn output_format_all_variants_roundtrip() {
        let pairs = [
            (OutputFormat::Wav, FfiOutputFormat::Wav),
            (OutputFormat::Mp3, FfiOutputFormat::Mp3),
            (OutputFormat::Pcm, FfiOutputFormat::Pcm),
            (OutputFormat::Opus, FfiOutputFormat::Opus),
            (OutputFormat::Flac, FfiOutputFormat::Flac),
        ];
        for (native, expected_ffi) in pairs {
            let ffi: FfiOutputFormat = native.into();
            assert_eq!(ffi, expected_ffi);
            let back: OutputFormat = ffi.into();
            let ffi2: FfiOutputFormat = back.into();
            assert_eq!(ffi2, expected_ffi);
        }
    }

    // --- FfiAudioBuffer ---

    #[test]
    fn audio_buffer_from_native() {
        let native = AudioBuffer {
            data: vec![1u8, 2, 3, 4],
            config: AudioConfig {
                sample_rate: 44100,
                channels: 2,
                sample_format: SampleFormat::I16,
            },
        };
        let ffi: FfiAudioBuffer = native.into();
        assert_eq!(ffi.data, vec![1u8, 2, 3, 4]);
        assert_eq!(ffi.sample_rate, 44100);
        assert_eq!(ffi.channels, 2);
        assert_eq!(ffi.sample_format, FfiSampleFormat::I16);
    }

    #[test]
    fn audio_buffer_roundtrip() {
        let original = FfiAudioBuffer {
            data: vec![0u8, 255, 128],
            sample_rate: 16000,
            channels: 1,
            sample_format: FfiSampleFormat::F32,
        };
        let native: AudioBuffer = original.clone().into();
        assert_eq!(native.data, original.data);
        assert_eq!(native.config.sample_rate, original.sample_rate);
        assert_eq!(native.config.channels, original.channels);
        assert!(matches!(native.config.sample_format, SampleFormat::F32));
    }

    // --- FfiVoice ---

    #[test]
    fn voice_from_native() {
        let native = Voice {
            id: "voice-001".to_string(),
            name: Some("Alice".to_string()),
            language: Some("en".to_string()),
        };
        let ffi: FfiVoice = native.into();
        assert_eq!(ffi.id, "voice-001");
        assert_eq!(ffi.name, Some("Alice".to_string()));
        assert_eq!(ffi.language, Some("en".to_string()));
    }

    #[test]
    fn voice_to_native() {
        let ffi = FfiVoice {
            id: "v-xyz".to_string(),
            name: None,
            language: Some("fr".to_string()),
        };
        let native: Voice = ffi.into();
        assert_eq!(native.id, "v-xyz");
        assert!(native.name.is_none());
        assert_eq!(native.language, Some("fr".to_string()));
    }

    // --- FfiTtsOptions ---

    #[test]
    fn tts_options_to_native() {
        let ffi = FfiTtsOptions {
            voice_id: "my-voice".to_string(),
            speed: Some(1.5),
            output_format: FfiOutputFormat::Mp3,
        };
        let native = ffi.to_native();
        assert_eq!(native.voice.id, "my-voice");
        assert_eq!(native.speed, Some(1.5));
        assert!(matches!(native.output_format, OutputFormat::Mp3));
    }

    #[test]
    fn tts_options_to_native_no_speed() {
        let ffi = FfiTtsOptions {
            voice_id: "v".to_string(),
            speed: None,
            output_format: FfiOutputFormat::Wav,
        };
        let native = ffi.to_native();
        assert!(native.speed.is_none());
        assert!(matches!(native.output_format, OutputFormat::Wav));
    }

    // --- FfiSttOptions ---

    #[test]
    fn stt_options_conversion() {
        let ffi = FfiSttOptions {
            language: Some("es".to_string()),
            timestamps: true,
            prompt: Some("transcript this".to_string()),
        };
        let native: SttOptions = ffi.into();
        assert_eq!(native.language, Some("es".to_string()));
        assert!(native.timestamps);
        assert_eq!(native.prompt, Some("transcript this".to_string()));
    }

    #[test]
    fn stt_options_defaults() {
        let ffi = FfiSttOptions {
            language: None,
            timestamps: false,
            prompt: None,
        };
        let native: SttOptions = ffi.into();
        assert!(native.language.is_none());
        assert!(!native.timestamps);
        assert!(native.prompt.is_none());
    }

    // --- FfiTranscriptSegment ---

    #[test]
    fn transcript_segment_from_native() {
        let seg = TranscriptSegment {
            text: "Hello".to_string(),
            start: 0.5,
            end: 1.2,
        };
        let ffi: FfiTranscriptSegment = seg.into();
        assert_eq!(ffi.text, "Hello");
        assert!((ffi.start - 0.5).abs() < 1e-9);
        assert!((ffi.end - 1.2).abs() < 1e-9);
    }

    // --- FfiTranscript ---

    #[test]
    fn transcript_from_native_with_segments() {
        let native = Transcript {
            text: "Hello world".to_string(),
            language: Some("en".to_string()),
            duration_secs: Some(3.0),
            segments: vec![
                TranscriptSegment {
                    text: "Hello".to_string(),
                    start: 0.0,
                    end: 0.5,
                },
                TranscriptSegment {
                    text: "world".to_string(),
                    start: 0.5,
                    end: 1.0,
                },
            ],
        };
        let ffi: FfiTranscript = native.into();
        assert_eq!(ffi.text, "Hello world");
        assert_eq!(ffi.language, Some("en".to_string()));
        assert_eq!(ffi.duration_secs, Some(3.0));
        assert_eq!(ffi.segments.len(), 2);
        assert_eq!(ffi.segments[0].text, "Hello");
        assert_eq!(ffi.segments[1].text, "world");
    }

    #[test]
    fn transcript_from_native_no_segments() {
        let native = Transcript {
            text: "x".to_string(),
            language: None,
            duration_secs: None,
            segments: Vec::new(),
        };
        let ffi: FfiTranscript = native.into();
        assert!(ffi.segments.is_empty());
        assert!(ffi.language.is_none());
        assert!(ffi.duration_secs.is_none());
    }

    // --- FfiAudioDevice ---

    #[test]
    fn audio_device_input_from_native() {
        let native = AudioDevice {
            id: "dev-001".to_string(),
            name: "Microphone".to_string(),
            is_default: true,
            direction: DeviceDirection::Input,
        };
        let ffi: FfiAudioDevice = native.into();
        assert_eq!(ffi.id, "dev-001");
        assert_eq!(ffi.name, "Microphone");
        assert!(ffi.is_default);
        assert!(ffi.is_input);
    }

    #[test]
    fn audio_device_output_from_native() {
        let native = AudioDevice {
            id: "dev-002".to_string(),
            name: "Speakers".to_string(),
            is_default: false,
            direction: DeviceDirection::Output,
        };
        let ffi: FfiAudioDevice = native.into();
        assert!(!ffi.is_input);
        assert!(!ffi.is_default);
    }
}
