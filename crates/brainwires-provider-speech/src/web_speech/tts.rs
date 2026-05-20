//! Browser-native TTS via `window.speechSynthesis`.

#![cfg(all(target_arch = "wasm32", feature = "web-speech"))]

use wasm_bindgen::JsValue;
use web_sys::{SpeechSynthesis, SpeechSynthesisUtterance, SpeechSynthesisVoice};

use super::TtsSink;

/// Options for a [`WebSpeechTts::speak`] call.
///
/// All fields are optional; unset values fall back to the host's defaults
/// for the [`SpeechSynthesisUtterance`].
#[derive(Debug, Clone, Default)]
pub struct WebSpeechTtsOptions {
    /// Voice to use, identified by its [`SpeechSynthesisVoice::voice_uri`].
    /// If `None`, the browser picks a default voice for the language.
    pub voice_uri: Option<String>,
    /// BCP-47 language tag (e.g. `"en-US"`).
    pub lang: Option<String>,
    /// Speech rate (1.0 = normal). Browser-defined valid range, typically
    /// 0.1..=10.0.
    pub rate: Option<f32>,
    /// Pitch (1.0 = normal). Browser-defined valid range, typically 0.0..=2.0.
    pub pitch: Option<f32>,
    /// Volume (0.0..=1.0).
    pub volume: Option<f32>,
}

/// Lightweight description of a [`SpeechSynthesisVoice`] returned by
/// [`WebSpeechTts::voices`].
#[derive(Debug, Clone)]
pub struct VoiceInfo {
    /// Stable identifier — matches [`WebSpeechTtsOptions::voice_uri`].
    pub voice_uri: String,
    /// Human-readable voice name.
    pub name: String,
    /// BCP-47 language tag.
    pub lang: String,
    /// True if the voice runs locally (offline) rather than via a remote service.
    pub local_service: bool,
    /// True if this is the browser's default voice.
    pub default: bool,
}

/// Browser-native TTS, wrapping `window.speechSynthesis`.
///
/// TTS is a fire-and-forget sink — there is no audio output stream returned
/// to Rust. The browser plays audio directly through the user's output
/// device.
pub struct WebSpeechTts {
    synth: SpeechSynthesis,
}

impl WebSpeechTts {
    /// Create a new TTS handle from the current `window.speechSynthesis`.
    ///
    /// Returns an error if there is no `window` (e.g. when running inside a
    /// worker without a window) or `speechSynthesis` is unavailable.
    pub fn new() -> Result<Self, JsValue> {
        let window =
            web_sys::window().ok_or_else(|| JsValue::from_str("no global `window` available"))?;
        let synth = window.speech_synthesis().map_err(|e| {
            JsValue::from_str(&format!(
                "speechSynthesis unavailable: {}",
                e.as_string().unwrap_or_default()
            ))
        })?;
        Ok(Self { synth })
    }

    /// Construct from an existing [`SpeechSynthesis`] handle.
    pub fn from_synthesis(synth: SpeechSynthesis) -> Self {
        Self { synth }
    }

    /// Queue an utterance for playback. Returns immediately; speech happens
    /// asynchronously.
    pub fn speak(&self, text: &str, opts: WebSpeechTtsOptions) -> Result<(), JsValue> {
        let utter = SpeechSynthesisUtterance::new_with_text(text)?;

        if let Some(rate) = opts.rate {
            utter.set_rate(rate);
        }
        if let Some(pitch) = opts.pitch {
            utter.set_pitch(pitch);
        }
        if let Some(volume) = opts.volume {
            utter.set_volume(volume);
        }
        if let Some(lang) = opts.lang.as_deref() {
            utter.set_lang(lang);
        }
        if let Some(voice_uri) = opts.voice_uri.as_deref()
            && let Some(voice) = self.find_voice_by_uri(voice_uri)
        {
            utter.set_voice(Some(&voice));
        }

        self.synth.speak(&utter);
        Ok(())
    }

    /// Cancel any pending and currently-spoken utterances.
    pub fn cancel(&self) {
        self.synth.cancel();
    }

    /// Pause the current utterance.
    pub fn pause(&self) {
        self.synth.pause();
    }

    /// Resume a paused utterance.
    pub fn resume(&self) {
        self.synth.resume();
    }

    /// Snapshot of currently-available voices.
    ///
    /// Note: in many browsers the voice list is populated asynchronously and
    /// may be empty on the first call. Listen for the
    /// `voiceschanged` event on `speechSynthesis` if you need to wait for it.
    pub fn voices(&self) -> Vec<VoiceInfo> {
        let arr = self.synth.get_voices();
        let len = arr.length();
        let mut out = Vec::with_capacity(len as usize);
        for i in 0..len {
            let v = arr.get(i);
            let voice: SpeechSynthesisVoice = v.unchecked_into();
            out.push(VoiceInfo {
                voice_uri: voice.voice_uri(),
                name: voice.name(),
                lang: voice.lang(),
                local_service: voice.local_service(),
                default: voice.default(),
            });
        }
        out
    }

    /// True if the synthesizer is currently speaking.
    pub fn is_speaking(&self) -> bool {
        self.synth.speaking()
    }

    /// True if the synthesizer is currently paused.
    pub fn is_paused(&self) -> bool {
        self.synth.paused()
    }

    fn find_voice_by_uri(&self, voice_uri: &str) -> Option<SpeechSynthesisVoice> {
        let arr = self.synth.get_voices();
        let len = arr.length();
        for i in 0..len {
            let v = arr.get(i);
            let voice: SpeechSynthesisVoice = v.unchecked_into();
            if voice.voice_uri() == voice_uri {
                return Some(voice);
            }
        }
        None
    }
}

// Bring `unchecked_into` into scope for the `JsValue` -> `SpeechSynthesisVoice` casts above.
use wasm_bindgen::JsCast;

impl TtsSink for WebSpeechTts {
    type Options = WebSpeechTtsOptions;
    type Error = JsValue;

    fn speak(&self, text: &str, opts: Self::Options) -> Result<(), Self::Error> {
        WebSpeechTts::speak(self, text, opts)
    }

    fn cancel(&self) {
        WebSpeechTts::cancel(self);
    }

    fn pause(&self) {
        WebSpeechTts::pause(self);
    }

    fn resume(&self) {
        WebSpeechTts::resume(self);
    }
}
