//! Async-to-sync bridge and provider handle registry.
//!
//! Manages a static registry of live provider instances keyed by opaque `u64`
//! handles. All async audio operations are run on an internal Tokio runtime.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use brainwires_hardware::{
    AudioCapture, AudioConfig, AudioPlayback, CpalCapture, CpalPlayback, SpeechToText, TextToSpeech,
};

use crate::error::FfiAudioError;
use crate::types_ffi::*;

// ---------------------------------------------------------------------------
// Internal Tokio runtime
// ---------------------------------------------------------------------------

fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Runtime::new().expect("failed to create Tokio runtime for FFI bridge")
    })
}

// ---------------------------------------------------------------------------
// Provider registry
// ---------------------------------------------------------------------------

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

#[allow(dead_code)]
pub(crate) enum ProviderEntry {
    TtsOnly(Box<dyn TextToSpeech>),
    SttOnly(Box<dyn SpeechToText>),
    Both {
        tts: Box<dyn TextToSpeech>,
        stt: Box<dyn SpeechToText>,
    },
}

fn registry() -> &'static Mutex<HashMap<u64, ProviderEntry>> {
    static REG: OnceLock<Mutex<HashMap<u64, ProviderEntry>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn insert_provider(entry: ProviderEntry) -> u64 {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    registry().lock().unwrap().insert(handle, entry);
    handle
}

pub(crate) fn remove_provider(handle: u64) {
    registry().lock().unwrap().remove(&handle);
}

fn with_tts<F, R>(handle: u64, f: F) -> Result<R, FfiAudioError>
where
    F: FnOnce(&dyn TextToSpeech) -> Result<R, FfiAudioError>,
{
    let reg = registry().lock().unwrap();
    let entry = reg
        .get(&handle)
        .ok_or_else(|| FfiAudioError::InvalidHandle {
            message: format!("no provider with handle {handle}"),
        })?;
    match entry {
        ProviderEntry::TtsOnly(tts) | ProviderEntry::Both { tts, .. } => f(tts.as_ref()),
        ProviderEntry::SttOnly(_) => Err(FfiAudioError::Unsupported {
            message: "this provider does not support TTS".to_string(),
        }),
    }
}

fn with_stt<F, R>(handle: u64, f: F) -> Result<R, FfiAudioError>
where
    F: FnOnce(&dyn SpeechToText) -> Result<R, FfiAudioError>,
{
    let reg = registry().lock().unwrap();
    let entry = reg
        .get(&handle)
        .ok_or_else(|| FfiAudioError::InvalidHandle {
            message: format!("no provider with handle {handle}"),
        })?;
    match entry {
        ProviderEntry::SttOnly(stt) | ProviderEntry::Both { stt, .. } => f(stt.as_ref()),
        ProviderEntry::TtsOnly(_) => Err(FfiAudioError::Unsupported {
            message: "this provider does not support STT".to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// TTS bridge
// ---------------------------------------------------------------------------

pub(crate) fn tts_list_voices_sync(handle: u64) -> Result<Vec<FfiVoice>, FfiAudioError> {
    // Clone the provider reference pattern: we need to avoid holding the mutex
    // across an await point, so we grab what we need under the lock, then drop it.
    with_tts(handle, |tts| {
        runtime()
            .block_on(tts.list_voices())
            .map(|voices| voices.into_iter().map(Into::into).collect())
            .map_err(|e| FfiAudioError::Provider {
                message: e.to_string(),
            })
    })
}

pub(crate) fn tts_synthesize_sync(
    handle: u64,
    text: String,
    options: FfiTtsOptions,
) -> Result<FfiAudioBuffer, FfiAudioError> {
    with_tts(handle, |tts| {
        let native_opts = options.to_native();
        runtime()
            .block_on(tts.synthesize(&text, &native_opts))
            .map(Into::into)
            .map_err(|e| FfiAudioError::Provider {
                message: e.to_string(),
            })
    })
}

// ---------------------------------------------------------------------------
// STT bridge
// ---------------------------------------------------------------------------

pub(crate) fn stt_transcribe_sync(
    handle: u64,
    audio: FfiAudioBuffer,
    options: FfiSttOptions,
) -> Result<FfiTranscript, FfiAudioError> {
    with_stt(handle, |stt| {
        let native_audio = brainwires_hardware::AudioBuffer::from(audio);
        let native_opts = brainwires_hardware::SttOptions::from(options);
        runtime()
            .block_on(stt.transcribe(&native_audio, &native_opts))
            .map(Into::into)
            .map_err(|e| FfiAudioError::Provider {
                message: e.to_string(),
            })
    })
}

// ---------------------------------------------------------------------------
// Hardware audio bridge
// ---------------------------------------------------------------------------

pub(crate) fn audio_list_input_devices_sync() -> Result<Vec<FfiAudioDevice>, FfiAudioError> {
    let capture = CpalCapture::new();
    capture
        .list_devices()
        .map(|devs| devs.into_iter().map(Into::into).collect())
        .map_err(|e| FfiAudioError::Hardware {
            message: e.to_string(),
        })
}

pub(crate) fn audio_list_output_devices_sync() -> Result<Vec<FfiAudioDevice>, FfiAudioError> {
    let playback = CpalPlayback::new();
    playback
        .list_devices()
        .map(|devs| devs.into_iter().map(Into::into).collect())
        .map_err(|e| FfiAudioError::Hardware {
            message: e.to_string(),
        })
}

pub(crate) fn audio_record_sync(
    _device_id: Option<String>,
    duration_secs: f64,
) -> Result<FfiAudioBuffer, FfiAudioError> {
    let capture = CpalCapture::new();
    let config = AudioConfig::speech();
    runtime()
        .block_on(capture.record(None, &config, duration_secs))
        .map(Into::into)
        .map_err(|e| FfiAudioError::Hardware {
            message: e.to_string(),
        })
}

pub(crate) fn audio_play_sync(
    _device_id: Option<String>,
    buffer: FfiAudioBuffer,
) -> Result<(), FfiAudioError> {
    let playback = CpalPlayback::new();
    let native_buf = brainwires_hardware::AudioBuffer::from(buffer);
    runtime()
        .block_on(playback.play(None, &native_buf))
        .map_err(|e| FfiAudioError::Hardware {
            message: e.to_string(),
        })
}
