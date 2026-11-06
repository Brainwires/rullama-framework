//! The capture → transcribe state machine behind the C ABI.
//!
//! Devices and the speech model sit behind [`AudioSource`] and
//! [`Transcriber`], so the state machine is testable without a microphone or
//! a whisper model.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::dsp;
use crate::error::{self, ErrorSlot};

/// `rla_state` codes.
pub const STATE_IDLE: i32 = 0;
/// Capturing audio.
pub const STATE_CAPTURING: i32 = 1;
/// Transcription running on a worker thread.
pub const STATE_TRANSCRIBING: i32 = 2;
/// Transcript ready.
pub const STATE_DONE: i32 = 3;
/// Transcription failed (see `rla_last_error`).
pub const STATE_ERROR: i32 = 4;

/// A capture stops buffering after this many seconds (what was captured is kept).
pub const MAX_CAPTURE_SECS: u32 = 120;
/// Window for `rla_level`.
const LEVEL_WINDOW_MS: u32 = 100;

/// Lock a mutex, recovering from poisoning (a panic elsewhere must not wedge the ABI).
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why a source could not start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceError {
    /// No input device (maps to `-4`).
    NoDevice(String),
    /// Any other failure (maps to `-1`).
    Failed(String),
}

/// A running capture. Dropping it stops the capture.
pub type CaptureHandle = Box<dyn Send>;

/// Something that can deliver microphone audio into a [`CaptureSink`].
pub trait AudioSource: Send + Sync {
    /// Start delivering audio. Must call [`CaptureSink::configure`] before the
    /// first [`CaptureSink::push`].
    fn open(&self, sink: Arc<CaptureSink>) -> Result<CaptureHandle, SourceError>;
}

/// Speech-to-text over 16 kHz mono `f32`.
pub trait Transcriber: Send + Sync {
    /// Transcribe; the returned text is final (trimmed, no timestamps).
    ///
    /// Implementations should poll `cancel` and give up (with any `Err`) soon
    /// after it becomes `true`; the result is discarded then.
    fn transcribe(&self, samples_16k: &[f32], cancel: &AtomicBool) -> Result<String, String>;
}

/// Receives mono samples from a capture source (any thread).
#[derive(Default)]
pub struct CaptureSink {
    inner: Mutex<SinkInner>,
}

#[derive(Default)]
struct SinkInner {
    rate: u32,
    samples: Vec<f32>,
    max_samples: usize,
    recent: VecDeque<f32>,
    recent_cap: usize,
}

impl CaptureSink {
    /// Set the rate of the mono samples that will be pushed.
    pub fn configure(&self, rate: u32) {
        let mut s = lock(&self.inner);
        s.rate = rate;
        s.max_samples = rate as usize * MAX_CAPTURE_SECS as usize;
        s.recent_cap = ((rate * LEVEL_WINDOW_MS / 1000) as usize).max(1);
        s.samples.reserve(rate as usize * 10);
    }

    /// Append mono samples; buffering stops at [`MAX_CAPTURE_SECS`].
    pub fn push(&self, mono: &[f32]) {
        let mut s = lock(&self.inner);
        let room = s.max_samples.saturating_sub(s.samples.len());
        let n = room.min(mono.len());
        s.samples.extend_from_slice(&mono[..n]);
        let cap = s.recent_cap;
        let keep = mono.len().min(cap);
        let excess = (s.recent.len() + keep).saturating_sub(cap);
        s.recent.drain(..excess);
        s.recent.extend(&mono[mono.len() - keep..]);
    }

    /// RMS of the last ~100 ms.
    pub fn level(&self) -> f32 {
        let s = lock(&self.inner);
        let (a, b) = s.recent.as_slices();
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = a
            .iter()
            .chain(b)
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        ((sum_sq / s.recent.len() as f64).sqrt() as f32).clamp(0.0, 1.0)
    }

    /// ms buffered so far.
    pub fn captured_ms(&self) -> u32 {
        let s = lock(&self.inner);
        dsp::samples_to_ms(s.samples.len(), s.rate)
    }

    fn take(&self) -> (Vec<f32>, u32) {
        let mut s = lock(&self.inner);
        (std::mem::take(&mut s.samples), s.rate)
    }
}

enum Phase {
    Idle,
    Capturing {
        sink: Arc<CaptureSink>,
        // Held only to keep the capture running; dropped to stop it.
        _handle: CaptureHandle,
    },
    Transcribing {
        job: u64,
        // Set to abort the inference when the job is abandoned.
        cancel: Arc<AtomicBool>,
    },
    Done(String),
    Failed(String),
}

/// A transcription worker thread, kept so it can be aborted and joined.
struct Worker {
    cancel: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

struct Inner {
    phase: Phase,
    transcriber: Option<Arc<dyn Transcriber>>,
    job: u64,
    last_captured_ms: u32,
    /// Worker threads that may still be running (finished ones are pruned).
    workers: Vec<Worker>,
    /// Cancel flags of `transcribe_now` calls in flight on caller threads.
    sync_jobs: Vec<Arc<AtomicBool>>,
}

/// Removes a `transcribe_now` cancel flag from the engine when the call ends,
/// even if the transcriber panics.
struct SyncJob<'a> {
    inner: &'a Mutex<Inner>,
    cancel: Arc<AtomicBool>,
}

impl Drop for SyncJob<'_> {
    fn drop(&mut self) {
        lock(self.inner)
            .sync_jobs
            .retain(|c| !Arc::ptr_eq(c, &self.cancel));
    }
}

/// The state machine. One global instance backs the C ABI.
pub struct Engine {
    inner: Arc<Mutex<Inner>>,
    source: Box<dyn AudioSource>,
    errors: &'static ErrorSlot,
}

/// Transcribe already-prepared audio: empty when there is too little speech.
fn run_transcriber(
    t: &dyn Transcriber,
    prepared: &dsp::Trimmed,
    cancel: &AtomicBool,
) -> Result<String, String> {
    if prepared.is_too_short() {
        return Ok(String::new());
    }
    t.transcribe(&prepared.samples, cancel)
}

impl Engine {
    /// A new idle engine capturing from `source` and reporting into `errors`.
    pub fn new(source: Box<dyn AudioSource>, errors: &'static ErrorSlot) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                phase: Phase::Idle,
                transcriber: None,
                job: 0,
                last_captured_ms: 0,
                workers: Vec::new(),
                sync_jobs: Vec::new(),
            })),
            source,
            errors,
        }
    }

    /// Install or replace the speech model.
    pub fn set_transcriber(&self, t: Arc<dyn Transcriber>) {
        lock(&self.inner).transcriber = Some(t);
    }

    /// Whether a model is installed.
    pub fn has_transcriber(&self) -> bool {
        lock(&self.inner).transcriber.is_some()
    }

    fn fail(&self, code: i32, msg: impl Into<String>) -> i32 {
        self.errors.set(msg);
        code
    }

    /// `rla_capture_start`.
    pub fn capture_start(&self) -> i32 {
        let mut inner = lock(&self.inner);
        if !matches!(inner.phase, Phase::Idle | Phase::Done(_)) {
            return self.fail(error::WRONG_STATE, "capture_start: not idle");
        }
        let sink = Arc::new(CaptureSink::default());
        match self.source.open(sink.clone()) {
            Ok(handle) => {
                inner.phase = Phase::Capturing {
                    sink,
                    _handle: handle,
                };
                inner.last_captured_ms = 0;
                error::OK
            }
            Err(SourceError::NoDevice(m)) => self.fail(error::NO_DEVICE, m),
            Err(SourceError::Failed(m)) => self.fail(error::ERROR, m),
        }
    }

    /// Stop the running capture and return its samples and rate.
    fn stop_capture(inner: &mut Inner) -> Option<(Vec<f32>, u32)> {
        let Phase::Capturing { sink, .. } = std::mem::replace(&mut inner.phase, Phase::Idle) else {
            return None;
        };
        // The handle was dropped by the replace above, so the source has stopped.
        inner.last_captured_ms = sink.captured_ms();
        Some(sink.take())
    }

    /// `rla_capture_stop`: stop and transcribe on a worker thread.
    ///
    /// Without a model the capture keeps running (so the caller can load one
    /// and stop again, or cancel) and `-3` is returned.
    pub fn capture_stop(&self) -> i32 {
        let mut inner = lock(&self.inner);
        if !matches!(inner.phase, Phase::Capturing { .. }) {
            return self.fail(error::WRONG_STATE, "capture_stop: not capturing");
        }
        let Some(transcriber) = inner.transcriber.clone() else {
            return self.fail(
                error::NO_MODEL,
                "no speech model loaded (call rla_stt_load)",
            );
        };
        let Some((samples, rate)) = Self::stop_capture(&mut inner) else {
            return error::WRONG_STATE;
        };
        inner.job += 1;
        let job = inner.job;
        let cancel = Arc::new(AtomicBool::new(false));
        inner.phase = Phase::Transcribing {
            job,
            cancel: Arc::clone(&cancel),
        };
        // Threads of abandoned jobs that have since finished need no tracking.
        inner.workers.retain(|w| !w.thread.is_finished());

        let shared = Arc::clone(&self.inner);
        let errors = self.errors;
        let flag = Arc::clone(&cancel);
        let spawned = std::thread::Builder::new()
            .name("rla-transcribe".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_transcriber(transcriber.as_ref(), &dsp::prepare(&samples, rate), &flag)
                }))
                .unwrap_or_else(|p| {
                    Err(format!(
                        "transcription panicked: {}",
                        error::panic_message(&p)
                    ))
                });
                let mut inner = lock(&shared);
                if matches!(inner.phase, Phase::Transcribing { job: j, .. } if j == job) {
                    inner.phase = match result {
                        Ok(text) => Phase::Done(text),
                        Err(e) => {
                            errors.set(e.clone());
                            Phase::Failed(e)
                        }
                    };
                }
            });
        match spawned {
            Ok(thread) => {
                inner.workers.push(Worker { cancel, thread });
                error::OK
            }
            Err(e) => {
                let msg = format!("cannot start transcription thread: {e}");
                inner.phase = Phase::Failed(msg.clone());
                self.fail(error::ERROR, msg)
            }
        }
    }

    /// `rla_capture_stop_wav`: stop without transcribing, write the trimmed
    /// audio as 16 kHz mono 16-bit WAV, return its duration in ms.
    pub fn capture_stop_wav(&self, path: &Path) -> i32 {
        let mut inner = lock(&self.inner);
        let Some((samples, rate)) = Self::stop_capture(&mut inner) else {
            return self.fail(error::WRONG_STATE, "capture_stop_wav: not capturing");
        };
        drop(inner);
        let trimmed = dsp::prepare(&samples, rate);
        match dsp::write_wav_16k(path, &trimmed.samples) {
            Ok(()) => i32::try_from(dsp::samples_to_ms(trimmed.samples.len(), dsp::TARGET_RATE))
                .unwrap_or(i32::MAX),
            Err(e) => self.fail(error::ERROR, e),
        }
    }

    /// `rla_cancel`: stop a capture (its stream is closed before this
    /// returns) or abort a transcription in flight (its worker exits soon
    /// after; the result is dropped either way).
    pub fn cancel(&self) -> i32 {
        let mut inner = lock(&self.inner);
        if let Phase::Transcribing { cancel, .. } = &inner.phase {
            cancel.store(true, Ordering::SeqCst);
        }
        inner.phase = Phase::Idle;
        inner.last_captured_ms = 0;
        error::OK
    }

    /// `rla_shutdown`: cancel everything, unload the model, abort synchronous
    /// transcriptions on other threads, and join every worker thread, so no
    /// thread of this engine is running when it returns (synchronous
    /// `transcribe_now` callers return soon after with an error).
    pub fn shutdown(&self) -> i32 {
        let workers = {
            let mut inner = lock(&self.inner);
            for w in &inner.workers {
                w.cancel.store(true, Ordering::SeqCst);
            }
            for c in &inner.sync_jobs {
                c.store(true, Ordering::SeqCst);
            }
            // Dropping a capture handle closes the stream and joins its thread.
            inner.phase = Phase::Idle;
            inner.transcriber = None;
            inner.last_captured_ms = 0;
            std::mem::take(&mut inner.workers)
        };
        // The lock is released: workers take it once to publish (and drop) their result.
        for w in workers {
            let _ = w.thread.join();
        }
        error::OK
    }

    /// `rla_state`.
    pub fn state(&self) -> i32 {
        match lock(&self.inner).phase {
            Phase::Idle => STATE_IDLE,
            Phase::Capturing { .. } => STATE_CAPTURING,
            Phase::Transcribing { .. } => STATE_TRANSCRIBING,
            Phase::Done(_) => STATE_DONE,
            Phase::Failed(_) => STATE_ERROR,
        }
    }

    /// `rla_level`.
    pub fn level(&self) -> f32 {
        match &lock(&self.inner).phase {
            Phase::Capturing { sink, .. } => sink.level(),
            _ => 0.0,
        }
    }

    /// `rla_captured_ms`: live while capturing, then the final length of the
    /// last capture until the next start or cancel.
    pub fn captured_ms(&self) -> u32 {
        let inner = lock(&self.inner);
        match &inner.phase {
            Phase::Capturing { sink, .. } => sink.captured_ms(),
            _ => inner.last_captured_ms,
        }
    }

    /// `rla_take_transcript`.
    ///
    /// # Safety
    /// `buf` must be null or valid for `cap` bytes of writes.
    pub unsafe fn take_transcript(&self, buf: *mut u8, cap: usize) -> i32 {
        let mut inner = lock(&self.inner);
        match &inner.phase {
            Phase::Done(text) => {
                // SAFETY: forwarded from the caller's contract.
                let len = unsafe { error::copy_out(text, buf, cap) };
                if text.len() < cap {
                    inner.phase = Phase::Idle;
                }
                len
            }
            Phase::Failed(msg) => {
                // Re-publish: another call may have replaced the last error since.
                self.errors.set(msg.clone());
                inner.phase = Phase::Idle;
                error::ERROR
            }
            _ => self.fail(error::WRONG_STATE, "take_transcript: no transcript ready"),
        }
    }

    /// Synchronous transcription of 16 kHz-resampled-and-trimmed audio, without
    /// touching the capture state. `Err` carries the return code.
    ///
    /// [`Engine::shutdown`] aborts it (it then returns `Err(-1)`).
    pub fn transcribe_now(&self, mono: &[f32], rate: u32) -> Result<String, i32> {
        let job = SyncJob {
            inner: &self.inner,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        let t = {
            let mut inner = lock(&self.inner);
            let Some(t) = inner.transcriber.clone() else {
                return Err(self.fail(
                    error::NO_MODEL,
                    "no speech model loaded (call rla_stt_load)",
                ));
            };
            inner.sync_jobs.push(Arc::clone(&job.cancel));
            t
        };
        run_transcriber(t.as_ref(), &dsp::prepare(mono, rate), &job.cancel)
            .map_err(|e| self.fail(error::ERROR, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::tests::phrase_16k;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn errors() -> &'static ErrorSlot {
        Box::leak(Box::new(ErrorSlot::default()))
    }

    /// Pushes a fixed clip on open; records whether it was stopped.
    struct FakeSource {
        clip: Vec<f32>,
        rate: u32,
        fail: Option<SourceError>,
        stopped: Arc<AtomicUsize>,
    }

    struct FakeHandle(Arc<AtomicUsize>);
    impl Drop for FakeHandle {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl AudioSource for FakeSource {
        fn open(&self, sink: Arc<CaptureSink>) -> Result<CaptureHandle, SourceError> {
            if let Some(e) = &self.fail {
                return Err(e.clone());
            }
            sink.configure(self.rate);
            sink.push(&self.clip);
            Ok(Box::new(FakeHandle(self.stopped.clone())))
        }
    }

    fn source(clip: Vec<f32>, rate: u32) -> (Box<FakeSource>, Arc<AtomicUsize>) {
        let stopped = Arc::new(AtomicUsize::new(0));
        (
            Box::new(FakeSource {
                clip,
                rate,
                fail: None,
                stopped: stopped.clone(),
            }),
            stopped,
        )
    }

    /// Returns `text`, optionally waiting for a release signal first.
    struct FakeTranscriber {
        text: Result<String, String>,
        gate: Option<Mutex<mpsc::Receiver<()>>>,
        calls: AtomicUsize,
        aborted: AtomicUsize,
    }

    impl FakeTranscriber {
        fn ok(text: &str) -> Arc<Self> {
            Arc::new(Self {
                text: Ok(text.into()),
                gate: None,
                calls: AtomicUsize::new(0),
                aborted: AtomicUsize::new(0),
            })
        }
    }

    impl Transcriber for FakeTranscriber {
        fn transcribe(&self, samples: &[f32], cancel: &AtomicBool) -> Result<String, String> {
            assert!(!samples.is_empty());
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(g) = &self.gate {
                // Wait for the release signal or for cancellation, like
                // whisper.cpp's abort callback.
                let gate = lock(g);
                loop {
                    match gate.recv_timeout(Duration::from_millis(5)) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    if cancel.load(Ordering::SeqCst) {
                        self.aborted.fetch_add(1, Ordering::SeqCst);
                        return Err("cancelled".into());
                    }
                }
            }
            self.text.clone()
        }
    }

    fn wait_for(engine: &Engine, state: i32) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while engine.state() != state {
            assert!(
                Instant::now() < deadline,
                "stuck in state {}",
                engine.state()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn take(engine: &Engine, cap: usize) -> (i32, Vec<u8>) {
        let mut buf = vec![0xAAu8; cap];
        let n = unsafe { engine.take_transcript(buf.as_mut_ptr(), cap) };
        (n, buf)
    }

    #[test]
    fn full_cycle_with_truncated_then_full_take() {
        let (src, stopped) = source(phrase_16k(), 16_000);
        let engine = Engine::new(src, errors());
        assert_eq!(engine.state(), STATE_IDLE);
        assert_eq!(take(&engine, 8).0, error::WRONG_STATE);
        assert_eq!(engine.capture_stop(), error::WRONG_STATE);

        assert_eq!(engine.capture_start(), 0);
        assert_eq!(engine.state(), STATE_CAPTURING);
        assert_eq!(engine.capture_start(), error::WRONG_STATE);
        assert!(engine.level() > 0.0);
        assert!(engine.captured_ms() > 1_000);

        // No model: -3 and the capture keeps running.
        assert_eq!(engine.capture_stop(), error::NO_MODEL);
        assert_eq!(engine.state(), STATE_CAPTURING);
        assert_eq!(stopped.load(Ordering::SeqCst), 0);

        let t = FakeTranscriber::ok("hello world");
        engine.set_transcriber(t.clone());
        assert_eq!(engine.capture_stop(), 0);
        assert_eq!(stopped.load(Ordering::SeqCst), 1);
        assert_eq!(engine.level(), 0.0);
        wait_for(&engine, STATE_DONE);
        assert_eq!(t.calls.load(Ordering::SeqCst), 1);

        // Too small (len == cap): truncated, NUL-terminated, not consumed.
        let (n, buf) = take(&engine, 11);
        assert_eq!(n, 11);
        assert_eq!(&buf, b"hello worl\0");
        assert_eq!(engine.state(), STATE_DONE);
        // Size query with a null buffer.
        assert_eq!(
            unsafe { engine.take_transcript(std::ptr::null_mut(), 0) },
            11
        );
        assert_eq!(engine.state(), STATE_DONE);

        let (n, buf) = take(&engine, 12);
        assert_eq!(n, 11);
        assert_eq!(&buf, b"hello world\0");
        assert_eq!(engine.state(), STATE_IDLE);
        assert_eq!(take(&engine, 12).0, error::WRONG_STATE);
    }

    #[test]
    fn start_from_done_discards_transcript() {
        let (src, _) = source(phrase_16k(), 16_000);
        let engine = Engine::new(src, errors());
        engine.set_transcriber(FakeTranscriber::ok("x"));
        assert_eq!(engine.capture_start(), 0);
        assert_eq!(engine.capture_stop(), 0);
        wait_for(&engine, STATE_DONE);
        assert_eq!(engine.capture_start(), 0);
        assert_eq!(engine.state(), STATE_CAPTURING);
        assert_eq!(engine.cancel(), 0);
        assert_eq!(engine.state(), STATE_IDLE);
        assert_eq!(engine.captured_ms(), 0);
    }

    #[test]
    fn short_or_silent_capture_gives_empty_transcript() {
        let (src, _) = source(vec![0.0; 48_000], 48_000);
        let engine = Engine::new(src, errors());
        let t = FakeTranscriber::ok("should not be used");
        engine.set_transcriber(t.clone());
        assert_eq!(engine.capture_start(), 0);
        assert_eq!(engine.captured_ms(), 1_000);
        assert_eq!(engine.capture_stop(), 0);
        wait_for(&engine, STATE_DONE);
        assert_eq!(engine.captured_ms(), 1_000);
        assert_eq!(t.calls.load(Ordering::SeqCst), 0);
        let (n, buf) = take(&engine, 4);
        assert_eq!((n, buf[0]), (0, 0));
        assert_eq!(engine.state(), STATE_IDLE);
    }

    #[test]
    fn no_device_and_source_failure_codes() {
        let errs = errors();
        let mut src = source(vec![], 16_000).0;
        src.fail = Some(SourceError::NoDevice("no default input device".into()));
        let engine = Engine::new(src, errs);
        assert_eq!(engine.capture_start(), error::NO_DEVICE);
        assert_eq!(engine.state(), STATE_IDLE);
        assert_eq!(errs.get(), "no default input device");

        let mut src = source(vec![], 16_000).0;
        src.fail = Some(SourceError::Failed("boom".into()));
        let engine = Engine::new(src, errs);
        assert_eq!(engine.capture_start(), error::ERROR);
        assert_eq!(errs.get(), "boom");
    }

    #[test]
    fn transcriber_error_goes_to_state_4_then_resets() {
        let errs = errors();
        let (src, _) = source(phrase_16k(), 16_000);
        let engine = Engine::new(src, errs);
        engine.set_transcriber(Arc::new(FakeTranscriber {
            text: Err("model exploded".into()),
            gate: None,
            calls: AtomicUsize::new(0),
            aborted: AtomicUsize::new(0),
        }));
        assert_eq!(engine.capture_start(), 0);
        assert_eq!(engine.capture_stop(), 0);
        wait_for(&engine, STATE_ERROR);
        assert_eq!(errs.get(), "model exploded");
        assert_eq!(engine.capture_start(), error::WRONG_STATE);
        // The take re-publishes the job's error over the one set since.
        assert_eq!(errs.get(), "capture_start: not idle");
        assert_eq!(take(&engine, 16).0, error::ERROR);
        assert_eq!(errs.get(), "model exploded");
        assert_eq!(engine.state(), STATE_IDLE);
    }

    fn gated(text: &str) -> (Arc<FakeTranscriber>, mpsc::Sender<()>) {
        let (release, gate) = mpsc::channel();
        let t = Arc::new(FakeTranscriber {
            text: Ok(text.into()),
            gate: Some(Mutex::new(gate)),
            calls: AtomicUsize::new(0),
            aborted: AtomicUsize::new(0),
        });
        (t, release)
    }

    fn wait_until(what: &str, cond: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn cancel_aborts_transcription_in_flight() {
        let (src, _) = source(phrase_16k(), 16_000);
        let engine = Engine::new(src, errors());
        let (t, _release) = gated("late");
        engine.set_transcriber(t.clone());
        assert_eq!(engine.capture_start(), 0);
        assert_eq!(engine.capture_stop(), 0);
        assert_eq!(engine.state(), STATE_TRANSCRIBING);
        assert_eq!(engine.capture_start(), error::WRONG_STATE);
        wait_until("the job to start", || t.calls.load(Ordering::SeqCst) == 1);
        assert_eq!(engine.cancel(), 0);
        assert_eq!(engine.state(), STATE_IDLE);
        // A new capture starts while the old worker is still winding down.
        assert_eq!(engine.capture_start(), 0);
        // The cancel flag reached the transcriber, which gave up.
        wait_until("the abort", || t.aborted.load(Ordering::SeqCst) == 1);
        std::thread::sleep(Duration::from_millis(50));
        // The stale (failed) result was dropped.
        assert_eq!(engine.state(), STATE_CAPTURING);
        assert_eq!(engine.cancel(), 0);
    }

    #[test]
    fn stale_result_after_cancel_is_dropped() {
        // A transcriber that ignores the cancel flag and returns late.
        struct Stubborn(Mutex<mpsc::Receiver<()>>);
        impl Transcriber for Stubborn {
            fn transcribe(&self, _: &[f32], _: &AtomicBool) -> Result<String, String> {
                let _ = lock(&self.0).recv();
                Ok("late".into())
            }
        }
        let (src, _) = source(phrase_16k(), 16_000);
        let engine = Engine::new(src, errors());
        let (release, gate) = mpsc::channel();
        engine.set_transcriber(Arc::new(Stubborn(Mutex::new(gate))));
        assert_eq!(engine.capture_start(), 0);
        assert_eq!(engine.capture_stop(), 0);
        assert_eq!(engine.cancel(), 0);
        assert_eq!(engine.capture_start(), 0);
        release.send(()).unwrap();
        wait_until("the worker to exit", || {
            lock(&engine.inner)
                .workers
                .iter()
                .all(|w| w.thread.is_finished())
        });
        assert_eq!(engine.state(), STATE_CAPTURING);
        assert_eq!(engine.cancel(), 0);
    }

    #[test]
    fn shutdown_stops_capture_aborts_and_joins_workers_and_unloads() {
        let (src, stopped) = source(phrase_16k(), 16_000);
        let engine = Engine::new(src, errors());
        let mut opened = 0;
        for round in 0..4 {
            let (t, _release) = gated("x");
            engine.set_transcriber(t.clone());
            assert_eq!(engine.capture_start(), 0);
            opened += 1;
            assert_eq!(engine.capture_stop(), 0);
            wait_until("the job to start", || t.calls.load(Ordering::SeqCst) == 1);
            if round % 2 == 0 {
                // Abandon the job and leave a new capture running.
                assert_eq!(engine.cancel(), 0);
                assert_eq!(engine.capture_start(), 0);
                opened += 1;
            } else {
                // Shut down mid-transcription.
                assert_eq!(engine.state(), STATE_TRANSCRIBING);
            }

            assert_eq!(engine.shutdown(), 0);
            assert_eq!(engine.state(), STATE_IDLE);
            assert!(!engine.has_transcriber());
            assert_eq!(t.aborted.load(Ordering::SeqCst), 1);
            assert!(lock(&engine.inner).workers.is_empty());
            // Every capture opened so far was closed.
            assert_eq!(stopped.load(Ordering::SeqCst), opened);
        }
        // Idempotent when idle.
        assert_eq!(engine.shutdown(), 0);
    }

    #[test]
    fn shutdown_aborts_synchronous_transcription() {
        let (src, _) = source(vec![], 16_000);
        let engine = Arc::new(Engine::new(src, errors()));
        let (t, _release) = gated("never");
        engine.set_transcriber(t.clone());
        let e2 = Arc::clone(&engine);
        let call = std::thread::spawn(move || e2.transcribe_now(&phrase_16k(), 16_000));
        wait_until("the call to start", || t.calls.load(Ordering::SeqCst) == 1);
        assert_eq!(engine.shutdown(), 0);
        assert_eq!(call.join().unwrap(), Err(error::ERROR));
        assert!(lock(&engine.inner).sync_jobs.is_empty());
    }

    #[test]
    fn capture_stop_wav_writes_trimmed_16k_audio() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cap.wav");
        let mut clip = vec![0.0f32; 48_000];
        clip.extend(dsp::resample(&phrase_16k(), 16_000, 48_000));
        clip.extend(vec![0.0f32; 48_000]);
        let (src, _) = source(clip, 48_000);
        let engine = Engine::new(src, errors());
        assert_eq!(engine.capture_stop_wav(&path), error::WRONG_STATE);
        assert_eq!(engine.capture_start(), 0);
        let total_ms = engine.captured_ms();
        let ms = engine.capture_stop_wav(&path);
        assert!(
            ms > 1_000 && (ms as u32) < total_ms - 1_000,
            "{ms} of {total_ms}"
        );
        assert_eq!(engine.state(), STATE_IDLE);
        let spec = hound::WavReader::open(&path).unwrap().spec();
        assert_eq!(
            (spec.sample_rate, spec.channels, spec.bits_per_sample),
            (16_000, 1, 16)
        );
        let pcm = dsp::read_wav_pcm(&path).unwrap();
        assert_eq!(dsp::samples_to_ms(pcm.samples.len(), 16_000) as i32, ms);
    }

    #[test]
    fn capture_stop_wav_reports_write_errors() {
        let errs = errors();
        let (src, _) = source(phrase_16k(), 16_000);
        let engine = Engine::new(src, errs);
        assert_eq!(engine.capture_start(), 0);
        let bad = Path::new("/nonexistent-dir-rla/x.wav");
        assert_eq!(engine.capture_stop_wav(bad), error::ERROR);
        assert!(errs.get().contains("nonexistent-dir-rla"), "{}", errs.get());
        assert_eq!(engine.state(), STATE_IDLE);
    }

    #[test]
    fn transcribe_now_needs_a_model_and_leaves_state() {
        let (src, _) = source(vec![], 16_000);
        let engine = Engine::new(src, errors());
        assert_eq!(
            engine.transcribe_now(&phrase_16k(), 16_000),
            Err(error::NO_MODEL)
        );
        engine.set_transcriber(FakeTranscriber::ok("text"));
        assert_eq!(
            engine.transcribe_now(&phrase_16k(), 16_000).as_deref(),
            Ok("text")
        );
        assert_eq!(
            engine.transcribe_now(&[0.0; 8_000], 16_000).as_deref(),
            Ok("")
        );
        assert_eq!(engine.state(), STATE_IDLE);
    }

    #[test]
    fn sink_caps_at_120_s_and_meters_level() {
        let sink = CaptureSink::default();
        sink.configure(1_000);
        assert_eq!(sink.level(), 0.0);
        for _ in 0..130 {
            sink.push(&[0.5; 1_000]);
        }
        assert_eq!(sink.captured_ms(), 120_000);
        assert!((sink.level() - 0.5).abs() < 1e-6);
        // Level tracks only the last 100 ms (100 samples at 1 kHz).
        sink.push(&[0.0; 50]);
        assert!((sink.level() - (0.125f32).sqrt()).abs() < 1e-4);
        sink.push(&[0.0; 500]);
        assert_eq!(sink.level(), 0.0);
        assert_eq!(sink.captured_ms(), 120_000);
    }
}
