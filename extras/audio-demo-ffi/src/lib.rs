//! UniFFI bindings for brainwires-hardware.
//!
//! Exposes TTS, STT, and hardware audio functions to C# (and Kotlin/Swift/Python)
//! via Mozilla's UniFFI binding generator.

mod bridge;
mod error;
pub mod types_ffi;

use std::sync::Arc;

use brainwires_hardware::*;
use brainwires_provider::openai_responses::ResponsesClient;

pub use error::FfiAudioError;
pub use types_ffi::*;

use bridge::{ProviderEntry, insert_provider, remove_provider};

// ---------------------------------------------------------------------------
// Provider factory
// ---------------------------------------------------------------------------

/// Create a provider and return an opaque handle.
///
/// # Provider names
///
/// | Name | TTS | STT |
/// |------|-----|-----|
/// | `openai` | Yes | Yes |
/// | `openai-responses` | Yes | Yes |
/// | `elevenlabs` | Yes | Yes |
/// | `deepgram` | Yes | Yes |
/// | `google` | Yes | No |
/// | `azure` | Yes | Yes |
/// | `fish` | Yes | Yes |
/// | `cartesia` | Yes | No |
/// | `murf` | Yes | No |
#[uniffi::export]
pub fn create_provider(
    name: String,
    api_key: String,
    region: Option<String>,
) -> Result<u64, FfiAudioError> {
    let entry = match name.as_str() {
        "openai" => ProviderEntry::Both {
            tts: Box::new(OpenAiTts::new(&api_key)),
            stt: Box::new(OpenAiStt::new(&api_key)),
        },
        "openai-responses" => {
            let client = Arc::new(ResponsesClient::new(api_key));
            ProviderEntry::Both {
                tts: Box::new(OpenAiResponsesTts::from_client(
                    Arc::clone(&client),
                    "gpt-4o-audio-preview",
                )),
                stt: Box::new(OpenAiResponsesStt::from_client(
                    client,
                    "gpt-4o-audio-preview",
                )),
            }
        }
        "elevenlabs" => ProviderEntry::Both {
            tts: Box::new(ElevenLabsTts::new(&api_key)),
            stt: Box::new(ElevenLabsStt::new(&api_key)),
        },
        "deepgram" => ProviderEntry::Both {
            tts: Box::new(DeepgramTts::new(&api_key)),
            stt: Box::new(DeepgramStt::new(&api_key)),
        },
        "google" => ProviderEntry::TtsOnly(Box::new(GoogleTts::new(&api_key))),
        "azure" => {
            let region = region.ok_or_else(|| FfiAudioError::Provider {
                message: "Azure requires a 'region' parameter".to_string(),
            })?;
            ProviderEntry::Both {
                tts: Box::new(AzureTts::new(&api_key, &region)),
                stt: Box::new(AzureStt::new(&api_key, &region)),
            }
        }
        "fish" => ProviderEntry::Both {
            tts: Box::new(FishTts::new(&api_key)),
            stt: Box::new(FishStt::new(&api_key)),
        },
        "cartesia" => ProviderEntry::TtsOnly(Box::new(CartesiaTts::new(&api_key))),
        "murf" => ProviderEntry::TtsOnly(Box::new(MurfTts::new(&api_key))),
        _ => {
            return Err(FfiAudioError::UnknownProvider {
                message: format!("unknown provider: {name}"),
            });
        }
    };
    Ok(insert_provider(entry))
}

/// Release a provider handle.
#[uniffi::export]
pub fn drop_provider(handle: u64) {
    remove_provider(handle);
}

/// List all supported providers with their capabilities.
#[uniffi::export]
fn list_providers() -> Vec<FfiProviderInfo> {
    vec![
        FfiProviderInfo {
            name: "openai".into(),
            display_name: "OpenAI (TTS-1 / Whisper)".into(),
            has_tts: true,
            has_stt: true,
            requires_region: false,
        },
        FfiProviderInfo {
            name: "openai-responses".into(),
            display_name: "OpenAI Responses API (GPT-4o Audio)".into(),
            has_tts: true,
            has_stt: true,
            requires_region: false,
        },
        FfiProviderInfo {
            name: "elevenlabs".into(),
            display_name: "ElevenLabs".into(),
            has_tts: true,
            has_stt: true,
            requires_region: false,
        },
        FfiProviderInfo {
            name: "deepgram".into(),
            display_name: "Deepgram (Aura / Nova)".into(),
            has_tts: true,
            has_stt: true,
            requires_region: false,
        },
        FfiProviderInfo {
            name: "google".into(),
            display_name: "Google Cloud TTS".into(),
            has_tts: true,
            has_stt: false,
            requires_region: false,
        },
        FfiProviderInfo {
            name: "azure".into(),
            display_name: "Azure Cognitive Services".into(),
            has_tts: true,
            has_stt: true,
            requires_region: true,
        },
        FfiProviderInfo {
            name: "fish".into(),
            display_name: "Fish Audio".into(),
            has_tts: true,
            has_stt: true,
            requires_region: false,
        },
        FfiProviderInfo {
            name: "cartesia".into(),
            display_name: "Cartesia (Sonic)".into(),
            has_tts: true,
            has_stt: false,
            requires_region: false,
        },
        FfiProviderInfo {
            name: "murf".into(),
            display_name: "Murf AI".into(),
            has_tts: true,
            has_stt: false,
            requires_region: false,
        },
    ]
}

// ---------------------------------------------------------------------------
// TTS exports
// ---------------------------------------------------------------------------

/// List available voices for a TTS provider.
#[uniffi::export]
pub fn tts_list_voices(handle: u64) -> Result<Vec<FfiVoice>, FfiAudioError> {
    bridge::tts_list_voices_sync(handle)
}

/// Synthesize text to audio.
#[uniffi::export]
pub fn tts_synthesize(
    handle: u64,
    text: String,
    options: FfiTtsOptions,
) -> Result<FfiAudioBuffer, FfiAudioError> {
    bridge::tts_synthesize_sync(handle, text, options)
}

// ---------------------------------------------------------------------------
// STT exports
// ---------------------------------------------------------------------------

/// Transcribe audio to text.
#[uniffi::export]
pub fn stt_transcribe(
    handle: u64,
    audio: FfiAudioBuffer,
    options: FfiSttOptions,
) -> Result<FfiTranscript, FfiAudioError> {
    bridge::stt_transcribe_sync(handle, audio, options)
}

// ---------------------------------------------------------------------------
// Hardware audio exports
// ---------------------------------------------------------------------------

/// List available audio input (microphone) devices.
#[uniffi::export]
fn audio_list_input_devices() -> Result<Vec<FfiAudioDevice>, FfiAudioError> {
    bridge::audio_list_input_devices_sync()
}

/// List available audio output (speaker) devices.
#[uniffi::export]
fn audio_list_output_devices() -> Result<Vec<FfiAudioDevice>, FfiAudioError> {
    bridge::audio_list_output_devices_sync()
}

/// Record audio from the default input device.
#[uniffi::export]
fn audio_record(
    device_id: Option<String>,
    duration_secs: f64,
) -> Result<FfiAudioBuffer, FfiAudioError> {
    bridge::audio_record_sync(device_id, duration_secs)
}

/// Play audio through the default output device.
#[uniffi::export]
fn audio_play(device_id: Option<String>, buffer: FfiAudioBuffer) -> Result<(), FfiAudioError> {
    bridge::audio_play_sync(device_id, buffer)
}

// ---------------------------------------------------------------------------
// UniFFI scaffolding
// ---------------------------------------------------------------------------

uniffi::setup_scaffolding!();
