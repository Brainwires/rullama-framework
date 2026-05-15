//! Browser-native STT via `SpeechRecognition` / `webkitSpeechRecognition`.

#![cfg(all(target_arch = "wasm32", feature = "web-speech"))]

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Function, Reflect};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{SpeechRecognition, SpeechRecognitionEvent};

use super::SttSource;

/// Options for [`WebSpeechStt::start`].
#[derive(Debug, Clone, Default)]
pub struct WebSpeechSttOptions {
    /// BCP-47 language tag (e.g. `"en-US"`). If unset, the browser default is
    /// used (typically the document `lang` or system locale).
    pub lang: Option<String>,
    /// If true, the recognizer keeps listening across pauses until [`WebSpeechStt::stop`]
    /// or [`WebSpeechStt::abort`] is called.
    pub continuous: bool,
    /// If true, partial (non-final) results are dispatched as they arrive.
    pub interim_results: bool,
    /// Maximum number of alternatives per result; defaults to browser default
    /// (usually 1) when `None`.
    pub max_alternatives: Option<u32>,
}

/// A single recognition result delivered to the result callback.
#[derive(Debug, Clone)]
pub struct WebSpeechSttResult {
    /// Recognized transcript for this result (best alternative).
    pub text: String,
    /// True if this is a final result; false for interim results.
    pub is_final: bool,
    /// Confidence score (0.0..=1.0) reported by the recognizer.
    pub confidence: f32,
}

/// Errors that can be surfaced to the error callback.
#[derive(Debug, Clone)]
pub struct WebSpeechSttError {
    /// Error code/string from the underlying recognition error event.
    /// Common values: `"no-speech"`, `"aborted"`, `"audio-capture"`,
    /// `"network"`, `"not-allowed"`, `"service-not-allowed"`,
    /// `"bad-grammar"`, `"language-not-supported"`.
    pub error: String,
    /// Optional human-readable message.
    pub message: Option<String>,
}

/// Closures kept alive for the lifetime of the recognition session. Dropping
/// these would unhook the JS callbacks mid-flight.
struct StoredCallbacks {
    on_result: Option<Closure<dyn FnMut(JsValue)>>,
    on_error: Option<Closure<dyn FnMut(JsValue)>>,
    on_end: Option<Closure<dyn FnMut(JsValue)>>,
}

/// Browser-native STT, wrapping `SpeechRecognition` (with a runtime fallback
/// to `webkitSpeechRecognition` for Safari / older Chromium-based browsers).
pub struct WebSpeechStt {
    recognition: SpeechRecognition,
    callbacks: Rc<RefCell<StoredCallbacks>>,
}

impl WebSpeechStt {
    /// Create a new STT handle, preferring the standard `SpeechRecognition`
    /// constructor and falling back to `webkitSpeechRecognition`.
    ///
    /// Returns an error if neither constructor is present on `globalThis`
    /// (e.g. Firefox, where the API is unavailable at the time of writing).
    pub fn new() -> Result<Self, JsValue> {
        let recognition = construct_speech_recognition()?;
        Ok(Self {
            recognition,
            callbacks: Rc::new(RefCell::new(StoredCallbacks {
                on_result: None,
                on_error: None,
                on_end: None,
            })),
        })
    }

    /// Register a callback for incoming results (interim or final).
    ///
    /// The callback is stored on the handle to keep its [`Closure`] alive.
    /// Calling this again replaces the previous callback.
    pub fn on_result<F>(&self, mut cb: F)
    where
        F: FnMut(WebSpeechSttResult) + 'static,
    {
        let closure = Closure::wrap(Box::new(move |evt: JsValue| {
            let event: SpeechRecognitionEvent = match evt.dyn_into() {
                Ok(e) => e,
                Err(_) => return,
            };
            let result_index = event.result_index();
            let Some(results) = event.results() else {
                return;
            };
            let total = results.length();
            for i in result_index..total {
                let Some(result) = results.get(i) else {
                    continue;
                };
                let Some(alt) = result.get(0) else { continue };
                cb(WebSpeechSttResult {
                    text: alt.transcript(),
                    is_final: result.is_final(),
                    confidence: alt.confidence(),
                });
            }
        }) as Box<dyn FnMut(JsValue)>);

        self.recognition
            .set_onresult(Some(closure.as_ref().unchecked_ref::<Function>()));
        self.callbacks.borrow_mut().on_result = Some(closure);
    }

    /// Register a callback for recognition errors.
    pub fn on_error<F>(&self, mut cb: F)
    where
        F: FnMut(WebSpeechSttError) + 'static,
    {
        let closure = Closure::wrap(Box::new(move |evt: JsValue| {
            // The error event in current browsers exposes `error` (string) and
            // optionally `message` (string). We pull them via `Reflect` to
            // stay portable across the SpeechRecognitionError /
            // SpeechRecognitionErrorEvent split between web-sys versions.
            let error = Reflect::get(&evt, &JsValue::from_str("error"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_else(|| "unknown".to_string());
            let message = Reflect::get(&evt, &JsValue::from_str("message"))
                .ok()
                .and_then(|v| v.as_string());
            cb(WebSpeechSttError { error, message });
        }) as Box<dyn FnMut(JsValue)>);

        self.recognition
            .set_onerror(Some(closure.as_ref().unchecked_ref::<Function>()));
        self.callbacks.borrow_mut().on_error = Some(closure);
    }

    /// Register a callback fired when recognition ends.
    pub fn on_end<F>(&self, mut cb: F)
    where
        F: FnMut() + 'static,
    {
        let closure = Closure::wrap(Box::new(move |_evt: JsValue| {
            cb();
        }) as Box<dyn FnMut(JsValue)>);

        self.recognition
            .set_onend(Some(closure.as_ref().unchecked_ref::<Function>()));
        self.callbacks.borrow_mut().on_end = Some(closure);
    }

    /// Apply options and start recognition.
    pub fn start(&self, opts: WebSpeechSttOptions) -> Result<(), JsValue> {
        if let Some(lang) = opts.lang.as_deref() {
            self.recognition.set_lang(lang);
        }
        // continuous can throw on some browsers if unsupported; ignore the
        // error so we degrade gracefully.
        let _ = self.recognition.set_continuous(opts.continuous);
        self.recognition.set_interim_results(opts.interim_results);
        if let Some(n) = opts.max_alternatives {
            self.recognition.set_max_alternatives(n);
        }

        self.recognition.start()
    }

    /// Stop recognition gracefully (the recognizer may still emit a final
    /// result before the `end` event fires).
    pub fn stop(&self) {
        self.recognition.stop();
    }

    /// Abort recognition immediately without emitting any pending result.
    pub fn abort(&self) {
        self.recognition.abort();
    }
}

impl SttSource for WebSpeechStt {
    type Options = WebSpeechSttOptions;
    type Result = WebSpeechSttResult;
    type Error = JsValue;

    fn start(&self, opts: Self::Options) -> Result<(), Self::Error> {
        WebSpeechStt::start(self, opts)
    }

    fn stop(&self) {
        WebSpeechStt::stop(self);
    }

    fn abort(&self) {
        WebSpeechStt::abort(self);
    }
}

/// Try `SpeechRecognition`, then fall back to `webkitSpeechRecognition`.
fn construct_speech_recognition() -> Result<SpeechRecognition, JsValue> {
    // First, try the standard constructor via web-sys.
    if let Ok(r) = SpeechRecognition::new() {
        return Ok(r);
    }

    // Fall back to webkitSpeechRecognition on the global object.
    let global = js_sys::global();
    let ctor = Reflect::get(&global, &JsValue::from_str("webkitSpeechRecognition"))?;
    if ctor.is_undefined() || ctor.is_null() {
        return Err(JsValue::from_str(
            "SpeechRecognition is not available in this environment",
        ));
    }
    let ctor: Function = ctor
        .dyn_into()
        .map_err(|_| JsValue::from_str("webkitSpeechRecognition is not a constructor function"))?;
    let instance = Reflect::construct(&ctor, &js_sys::Array::new())?;
    instance.dyn_into::<SpeechRecognition>().map_err(|_| {
        JsValue::from_str("webkitSpeechRecognition() did not return a SpeechRecognition")
    })
}
