use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU8, Ordering},
};
use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use crate::audio::vad::{VoiceActivityDetector, energy::EnergyVad};
use crate::audio::{
    buffer::AudioRingBuffer,
    capture::AudioCapture,
    device::AudioDevice,
    error::{AudioError, AudioResult},
    playback::AudioPlayback,
    stt::SpeechToText,
    tts::TextToSpeech,
    types::{AudioBuffer, AudioConfig, SampleFormat, SttOptions, TtsOptions},
};

#[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
use crate::audio::wake_word::{WakeWordDetection, WakeWordDetector};

// ── State enum ────────────────────────────────────────────────────────────────

const STATE_IDLE: u8 = 0;
const STATE_LISTENING: u8 = 1;
const STATE_PROCESSING: u8 = 2;
const STATE_SPEAKING: u8 = 3;

/// Current operational state of the voice assistant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistantState {
    /// Waiting for a wake word (or for `listen_once` to be called).
    Idle,
    /// Capturing user speech.
    Listening,
    /// Running STT + calling the handler.
    Processing,
    /// Playing back a TTS response.
    Speaking,
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the voice assistant pipeline.
#[derive(Debug, Clone)]
pub struct VoiceAssistantConfig {
    /// Capture format. Default: `AudioConfig::speech()` (16 kHz mono i16).
    pub capture_config: AudioConfig,
    /// Energy threshold in dBFS below which audio is considered silent.
    /// Default: -40 dB.
    pub silence_threshold_db: f32,
    /// How many milliseconds of silence ends an utterance. Default: 800 ms.
    pub silence_duration_ms: u32,
    /// Maximum recording duration safety ceiling (seconds). Default: 30 s.
    pub max_record_secs: f64,
    /// STT options forwarded to the speech-to-text backend.
    pub stt_options: SttOptions,
    /// TTS options. `None` means no spoken response.
    pub tts_options: Option<TtsOptions>,
    /// Microphone device. `None` uses the system default.
    pub microphone: Option<AudioDevice>,
    /// Speaker device. `None` uses the system default.
    pub speaker: Option<AudioDevice>,
    /// How long to listen before entering the Idle/wake-word state again when
    /// no speech is detected (seconds). Default: 10 s.
    pub listen_timeout_secs: f64,
}

impl Default for VoiceAssistantConfig {
    fn default() -> Self {
        Self {
            capture_config: AudioConfig::speech(),
            silence_threshold_db: -40.0,
            silence_duration_ms: 800,
            max_record_secs: 30.0,
            stt_options: SttOptions::default(),
            tts_options: None,
            microphone: None,
            speaker: None,
            listen_timeout_secs: 10.0,
        }
    }
}

// ── Handler trait ─────────────────────────────────────────────────────────────

/// Callbacks invoked by [`VoiceAssistant::run`] during the pipeline.
#[async_trait]
pub trait VoiceAssistantHandler: Send + Sync {
    /// Called when a wake word fires (before listening begins).
    /// Override to provide feedback (e.g. a chime sound or LED flash).
    #[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
    async fn on_wake_word(&self, _detection: &WakeWordDetection) {}

    /// Called with the completed transcript.
    ///
    /// Return `Some(text)` to have the assistant speak a response via TTS.
    /// Return `None` for a silent acknowledgement (e.g. action-only commands).
    async fn on_speech(&self, transcript: &crate::audio::types::Transcript) -> Option<String>;

    /// Called on non-fatal errors (capture glitches, STT failures, etc.).
    async fn on_error(&self, _error: &AudioError) {}
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Builder for [`VoiceAssistant`].
pub struct VoiceAssistantBuilder {
    capture: Arc<dyn AudioCapture>,
    stt: Arc<dyn SpeechToText>,
    playback: Option<Arc<dyn AudioPlayback>>,
    tts: Option<Arc<dyn TextToSpeech>>,
    #[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
    wake_word: Option<Box<dyn WakeWordDetector>>,
    vad: Option<Box<dyn VoiceActivityDetector>>,
    config: VoiceAssistantConfig,
}

impl VoiceAssistantBuilder {
    /// Start a new builder with the minimum required components.
    pub fn new(capture: Arc<dyn AudioCapture>, stt: Arc<dyn SpeechToText>) -> Self {
        Self {
            capture,
            stt,
            playback: None,
            tts: None,
            #[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
            wake_word: None,
            vad: None,
            config: VoiceAssistantConfig::default(),
        }
    }

    /// Set the audio playback backend for TTS output.
    pub fn with_playback(mut self, p: Arc<dyn AudioPlayback>) -> Self {
        self.playback = Some(p);
        self
    }

    /// Set the text-to-speech backend for spoken responses.
    pub fn with_tts(mut self, tts: Arc<dyn TextToSpeech>) -> Self {
        self.tts = Some(tts);
        self
    }

    /// Set the wake word detector used to activate the listening phase.
    #[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
    pub fn with_wake_word(mut self, detector: Box<dyn WakeWordDetector>) -> Self {
        self.wake_word = Some(detector);
        self
    }

    /// Override the default `EnergyVad` with a custom VAD implementation.
    pub fn with_vad(mut self, vad: Box<dyn VoiceActivityDetector>) -> Self {
        self.vad = Some(vad);
        self
    }

    /// Replace the default [`VoiceAssistantConfig`].
    pub fn with_config(mut self, config: VoiceAssistantConfig) -> Self {
        self.config = config;
        self
    }

    /// Consume the builder and produce a [`VoiceAssistant`].
    pub fn build(self) -> VoiceAssistant {
        let vad: Box<dyn VoiceActivityDetector> = self
            .vad
            .unwrap_or_else(|| Box::new(EnergyVad::new(self.config.silence_threshold_db)));

        VoiceAssistant {
            config: self.config,
            capture: self.capture,
            playback: self.playback,
            stt: self.stt,
            tts: self.tts,
            #[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
            wake_word: self.wake_word,
            vad,
            state: Arc::new(AtomicU8::new(STATE_IDLE)),
            stop_flag: Arc::new(AtomicBool::new(false)),
            _stop_tx: None,
        }
    }
}

// ── VoiceAssistant ────────────────────────────────────────────────────────────

/// A voice assistant that orchestrates the full pipeline:
/// listen → (wake word) → VAD-gated capture → STT → handler → TTS → playback.
///
/// Create via [`VoiceAssistant::builder`] (or [`VoiceAssistantBuilder::new`]).
pub struct VoiceAssistant {
    config: VoiceAssistantConfig,
    capture: Arc<dyn AudioCapture>,
    playback: Option<Arc<dyn AudioPlayback>>,
    stt: Arc<dyn SpeechToText>,
    tts: Option<Arc<dyn TextToSpeech>>,
    #[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
    wake_word: Option<Box<dyn WakeWordDetector>>,
    vad: Box<dyn VoiceActivityDetector>,
    state: Arc<AtomicU8>,
    stop_flag: Arc<AtomicBool>,
    _stop_tx: Option<oneshot::Sender<()>>,
}

impl VoiceAssistant {
    /// Create a new builder with the required capture and STT backends.
    pub fn builder(
        capture: Arc<dyn AudioCapture>,
        stt: Arc<dyn SpeechToText>,
    ) -> VoiceAssistantBuilder {
        VoiceAssistantBuilder::new(capture, stt)
    }

    /// Current operational state.
    pub fn state(&self) -> AssistantState {
        match self.state.load(Ordering::Relaxed) {
            STATE_IDLE => AssistantState::Idle,
            STATE_LISTENING => AssistantState::Listening,
            STATE_PROCESSING => AssistantState::Processing,
            STATE_SPEAKING => AssistantState::Speaking,
            _ => AssistantState::Idle,
        }
    }

    /// Signal the running loop to stop after the current utterance completes.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    // ── listen_once ───────────────────────────────────────────────────────────

    /// Capture a single utterance (VAD-gated or timed) and return the
    /// transcript. Does **not** invoke wake word detection or handler callbacks.
    pub async fn listen_once(&mut self) -> AudioResult<crate::audio::types::Transcript> {
        let captured = self.capture_utterance().await?;
        self.state.store(STATE_PROCESSING, Ordering::Relaxed);
        let transcript = self
            .stt
            .transcribe(&captured, &self.config.stt_options)
            .await?;
        self.state.store(STATE_IDLE, Ordering::Relaxed);
        Ok(transcript)
    }

    // ── run ───────────────────────────────────────────────────────────────────

    /// Run the full assistant event loop indefinitely.
    ///
    /// The loop:
    /// 1. Listens for a wake word (if configured), then enters Listening state.
    /// 2. In Listening state, accumulates speech via VAD into an `AudioRingBuffer`.
    /// 3. Calls `handler.on_speech()` with the completed transcript.
    /// 4. If the handler returns text and TTS is configured, speaks the reply.
    /// 5. Loops back to step 1.
    ///
    /// Call [`stop`][Self::stop] to terminate cleanly after the current cycle.
    pub async fn run<H: VoiceAssistantHandler>(&mut self, handler: &H) -> AudioResult<()> {
        self.stop_flag.store(false, Ordering::Relaxed);
        info!("VoiceAssistant started");

        loop {
            if self.stop_flag.load(Ordering::Relaxed) {
                info!("VoiceAssistant stopping");
                break;
            }

            // ── Wake word phase ───────────────────────────────────────────────
            #[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
            {
                // Take detector out to avoid simultaneous &self borrow conflict.
                let mut detector = self.wake_word.take();
                let wake_result = if let Some(ref mut det) = detector {
                    Some(
                        Self::wait_for_wake_word_inner(
                            &self.capture,
                            &self.config,
                            &self.stop_flag,
                            det,
                        )
                        .await,
                    )
                } else {
                    None
                };
                self.wake_word = detector;

                match wake_result {
                    Some(Ok(det)) => {
                        info!(keyword = %det.keyword, score = det.score, "Wake word detected");
                        handler.on_wake_word(&det).await;
                    }
                    Some(Err(e)) => {
                        warn!("Wake word error: {e}");
                        handler.on_error(&e).await;
                        continue;
                    }
                    None => {}
                }
            }

            if self.stop_flag.load(Ordering::Relaxed) {
                break;
            }

            // ── Capture utterance ─────────────────────────────────────────────
            self.state.store(STATE_LISTENING, Ordering::Relaxed);
            let captured = match self.capture_utterance().await {
                Ok(buf) => buf,
                Err(e) => {
                    handler.on_error(&e).await;
                    self.state.store(STATE_IDLE, Ordering::Relaxed);
                    continue;
                }
            };

            if captured.is_empty() {
                debug!("No speech captured — returning to idle");
                self.state.store(STATE_IDLE, Ordering::Relaxed);
                continue;
            }

            // ── STT ───────────────────────────────────────────────────────────
            self.state.store(STATE_PROCESSING, Ordering::Relaxed);
            let transcript = match self
                .stt
                .transcribe(&captured, &self.config.stt_options)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    handler.on_error(&e).await;
                    self.state.store(STATE_IDLE, Ordering::Relaxed);
                    continue;
                }
            };

            if transcript.text.trim().is_empty() {
                debug!("STT returned empty transcript — returning to idle");
                self.state.store(STATE_IDLE, Ordering::Relaxed);
                continue;
            }

            info!(text = %transcript.text, "Transcript received");

            // ── Handler callback ──────────────────────────────────────────────
            let reply = handler.on_speech(&transcript).await;

            // ── TTS + Playback ────────────────────────────────────────────────
            if let (Some(text), Some(tts), Some(playback)) =
                (reply, self.tts.as_ref(), self.playback.as_ref())
            {
                self.state.store(STATE_SPEAKING, Ordering::Relaxed);
                let opts = self.config.tts_options.clone().unwrap_or_default();
                match tts.synthesize(&text, &opts).await {
                    Ok(audio) => {
                        if let Err(e) = playback.play(self.config.speaker.as_ref(), &audio).await {
                            handler.on_error(&e).await;
                        }
                    }
                    Err(e) => handler.on_error(&e).await,
                }
            }

            self.state.store(STATE_IDLE, Ordering::Relaxed);
        }

        Ok(())
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Capture one utterance from the microphone, gated by VAD.
    async fn capture_utterance(&mut self) -> AudioResult<AudioBuffer> {
        let config = &self.config.capture_config;
        let mut stream = self
            .capture
            .start_capture(self.config.microphone.as_ref(), config)?;

        let sr = config.sample_rate;
        let bytes_per_sample = match config.sample_format {
            SampleFormat::I16 => 2usize,
            SampleFormat::F32 => 4,
        };
        // 20 ms per VAD chunk
        let chunk_samples = (sr as usize * 20 / 1000) * config.channels as usize;
        let chunk_bytes = chunk_samples * bytes_per_sample;

        let mut ring = AudioRingBuffer::new(config.clone(), self.config.max_record_secs);
        let mut silence_ms: u32 = 0;
        let mut total_ms: u64;
        let mut started_speaking = false;
        let silence_limit_ms = self.config.silence_duration_ms;
        let listen_timeout_ms = (self.config.listen_timeout_secs * 1000.0) as u64;

        let mut pending: Vec<u8> = Vec::new();

        let start = Instant::now();

        while let Some(result) = stream.next().await {
            let buf = match result {
                Ok(b) => b,
                Err(e) => return Err(e),
            };

            total_ms = start.elapsed().as_millis() as u64;

            // Timeout if we haven't heard speech yet
            if !started_speaking && total_ms > listen_timeout_ms {
                debug!("Listen timeout — no speech detected");
                break;
            }

            pending.extend_from_slice(&buf.data);

            // Process complete 20 ms chunks
            while pending.len() >= chunk_bytes {
                let chunk_data: Vec<u8> = pending.drain(..chunk_bytes).collect();
                let chunk_buf = AudioBuffer {
                    data: chunk_data,
                    config: config.clone(),
                };
                let is_speech = self.vad.is_speech(&chunk_buf);

                if is_speech {
                    started_speaking = true;
                    silence_ms = 0;
                    ring.push(&chunk_buf.data);
                } else if started_speaking {
                    silence_ms += 20;
                    ring.push(&chunk_buf.data); // include trailing silence
                    if silence_ms >= silence_limit_ms {
                        debug!("End of utterance (silence={silence_ms}ms)");
                        break;
                    }
                }
            }

            // Check ceiling
            if ring.duration_secs() >= self.config.max_record_secs {
                debug!("Max record duration reached");
                break;
            }

            // Break on silence limit (inner loop may have set this)
            if started_speaking && silence_ms >= silence_limit_ms {
                break;
            }
        }

        if ring.is_empty() {
            return Ok(AudioBuffer::new(config.clone()));
        }

        Ok(AudioBuffer::from_pcm(ring.read_all(), config.clone()))
    }

    /// Block until the wake word fires, returning the detection.
    /// Static to avoid borrow conflicts when `wake_word` is taken out of self.
    #[cfg(any(feature = "wake-word", feature = "wake-word-rustpotter"))]
    async fn wait_for_wake_word_inner(
        capture: &Arc<dyn AudioCapture>,
        cfg: &VoiceAssistantConfig,
        stop_flag: &Arc<AtomicBool>,
        detector: &mut Box<dyn WakeWordDetector>,
    ) -> AudioResult<WakeWordDetection> {
        use crate::audio::vad::pcm_to_i16_mono;

        let config = &cfg.capture_config;
        let mut stream = capture.start_capture(cfg.microphone.as_ref(), config)?;

        let frame_size = detector.frame_size();

        let mut sample_buf: Vec<i16> = Vec::new();

        while let Some(result) = stream.next().await {
            let buf = match result {
                Ok(b) => b,
                Err(e) => return Err(e),
            };

            let mono = pcm_to_i16_mono(&buf);
            sample_buf.extend_from_slice(&mono);

            while sample_buf.len() >= frame_size {
                let frame: Vec<i16> = sample_buf.drain(..frame_size).collect();
                if let Some(det) = detector.process_frame(&frame) {
                    return Ok(det);
                }
            }

            if stop_flag.load(Ordering::Relaxed) {
                return Err(AudioError::StreamClosed("assistant stopped".into()));
            }
        }

        Err(AudioError::StreamClosed("mic stream ended".into()))
    }
}
