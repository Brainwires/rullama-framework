//! Cooperative cancellation for the async TTS synths (Kokoro + StyleTTS2 clone).
//!
//! The synths run as wasm-bindgen `async` methods that borrow their model
//! (`&mut self`) for the whole call, so JS cannot call a method on that object to
//! cancel it mid-flight — wasm-bindgen would panic on the re-entrant borrow.
//! Instead a module-global flag is flipped by a wasm-bindgen **free** function
//! ([`request_cancel_js`], `ttsRequestCancel` in JS), which touches no object, and
//! polled by the forward at each stage boundary, where it bails out with an empty
//! buffer. The synth yields to the worker event loop at every GPU readback, so a
//! `cancel` message posted while a synth is running is handled at the next yield and
//! takes effect at the next stage boundary. No worker teardown, model stays loaded.
//!
//! wasm is single-threaded, so `Relaxed` ordering is sufficient.

use std::sync::atomic::{AtomicBool, Ordering};

static CANCEL: AtomicBool = AtomicBool::new(false);

/// Request cancellation of the in-flight synthesis (native callers / the wasm fn).
pub fn request() {
    CANCEL.store(true, Ordering::Relaxed);
}

/// Clear the flag. Called at the start of every synth so a stale request from a
/// previous run can't abort the next one.
pub fn clear() {
    CANCEL.store(false, Ordering::Relaxed);
}

/// Whether a cancel has been requested since the last [`clear`]. The forward polls
/// this at stage boundaries.
pub fn requested() -> bool {
    CANCEL.load(Ordering::Relaxed)
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// JS-facing: request cancellation of the in-flight TTS synthesis. Safe to call
/// while a synth is running — it does not touch the model object, so there is no
/// borrow conflict. The running synth aborts at its next stage boundary and
/// resolves with an empty buffer (the client treats empty as "cancelled").
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = ttsRequestCancel)]
pub fn request_cancel_js() {
    request();
}
