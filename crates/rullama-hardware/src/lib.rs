#![deny(missing_docs)]

//! # rullama-hardware
//!
//! Hardware I/O for the Brainwires Agent Framework.
//!
//! Provides a unified hardware abstraction layer covering:
//!
//! | Module | Feature flag | Description |
//! |--------|-------------|-------------|
//! | [`audio`] | `audio` | Audio capture/playback, STT, TTS (16 cloud providers + local Whisper) |
//! | [`gpio`] | `gpio` | GPIO pin management with safety allow-lists and PWM (Linux) |
//! | [`bluetooth`] | `bluetooth` | BLE advertisement scanning and adapter enumeration |
//! | [`camera`] | `camera` | Webcam/camera frame capture (V4L2/AVFoundation/MSMF) |
//! | [`usb`] | `usb` | Raw USB device enumeration and bulk/control/interrupt transfers |
//!
//! Home automation protocols (Matter, Zigbee, Z-Wave, Thread) live in the
//! standalone `future/home-automation/rullama-homeauto` workspace.
//!
//! ## Feature flags
//!
//! ```toml
//! [dependencies]
//! rullama-hardware = { version = "0.11", features = ["audio", "gpio", "bluetooth", "camera"] }
//! # or enable everything:
//! rullama-hardware = { version = "0.11", features = ["full"] }
//! ```
//!
//! ### Audio
//! The `audio` feature enables hardware audio capture/playback via CPAL and
//! 16 cloud STT/TTS provider integrations. Add `local-stt` for offline Whisper
//! inference and `flac` for FLAC codec support.
//!
//! ### GPIO (Linux)
//! The `gpio` feature exposes safe GPIO pin access using the Linux character
//! device API (`gpio-cdev`) with an explicit allow-list safety policy.
//!
//! ### Bluetooth
//! The `bluetooth` feature uses [`btleplug`](https://crates.io/crates/btleplug)
//! for cross-platform BLE scanning (Linux/BlueZ, macOS CoreBluetooth, Windows WinRT).
//!
//! ### Camera
//! The `camera` feature enables video frame capture using [`nokhwa`](https://crates.io/crates/nokhwa):
//! V4L2 on Linux, AVFoundation on macOS, Media Foundation on Windows.
//!
//! ### USB
//! The `usb` feature provides raw USB device enumeration and transfers via
//! [`nusb`](https://crates.io/crates/nusb) — a pure-Rust async USB library
//! with no `libusb` system dependency.

/// Audio capture, playback, STT, and TTS.
#[cfg(feature = "audio")]
pub mod audio;

/// GPIO hardware access (Linux).
#[cfg(feature = "gpio")]
pub mod gpio;

/// Bluetooth discovery and scanning.
#[cfg(feature = "bluetooth")]
pub mod bluetooth;

/// Camera and webcam frame capture.
#[cfg(feature = "camera")]
pub mod camera;

/// Raw USB device access and transfers.
#[cfg(feature = "usb")]
pub mod usb;

// ── Convenience re-exports: mirrors the old rullama-audio public API ──────

#[cfg(feature = "audio")]
pub use audio::{
    AudioBuffer, AudioCapture, AudioConfig, AudioDevice, AudioError, AudioPlayback, AudioResult,
    AudioRingBuffer, DeviceDirection, OutputFormat, SampleFormat, SpeechToText, SttOptions,
    TextToSpeech, Transcript, TranscriptSegment, TtsOptions, Voice,
};

#[cfg(feature = "audio")]
pub use audio::{decode_wav, encode_wav};

#[cfg(feature = "audio")]
pub use audio::{
    AzureStt, AzureTts, CartesiaTts, DeepgramStt, DeepgramTts, ElevenLabsStt, ElevenLabsTts,
    FishStt, FishTts, GoogleTts, MurfTts, OpenAiResponsesStt, OpenAiResponsesTts, OpenAiStt,
    OpenAiTts,
};

#[cfg(feature = "audio")]
pub use audio::{CpalCapture, CpalPlayback};

#[cfg(all(feature = "audio", feature = "flac"))]
pub use audio::{decode_flac, encode_flac};

#[cfg(all(feature = "audio", feature = "local-stt"))]
pub use audio::WhisperStt;

// ── GPIO re-exports ───────────────────────────────────────────────────────────

#[cfg(feature = "gpio")]
pub use gpio::{GpioChipInfo, GpioLineInfo, GpioPin, GpioPinManager, GpioSafetyPolicy};

// ── Camera re-exports ─────────────────────────────────────────────────────────

#[cfg(feature = "camera")]
pub use camera::{
    CameraCapture, CameraDevice, CameraError, CameraFormat, CameraFrame, FrameRate, NokhwaCapture,
    PixelFormat, Resolution,
};

// ── USB re-exports ────────────────────────────────────────────────────────────

#[cfg(feature = "usb")]
pub use usb::{UsbClass, UsbDevice, UsbError, UsbHandle, UsbSpeed};

// ── VAD re-exports ────────────────────────────────────────────────────────────

#[cfg(feature = "audio")]
pub use audio::{EnergyVad, SpeechSegment, VoiceActivityDetector};
#[cfg(feature = "vad")]
pub use audio::{VadMode, WebRtcVad};

// ── Wake word re-exports ──────────────────────────────────────────────────────

#[cfg(feature = "wake-word-dtw")]
pub use audio::{DtwWakeWordDetector, MfccExtractor};
#[cfg(any(feature = "wake-word", feature = "wake-word-dtw"))]
pub use audio::{EnergyTriggerDetector, WakeWordDetection, WakeWordDetector};

// ── Voice assistant re-exports ────────────────────────────────────────────────

#[cfg(feature = "voice-assistant")]
pub use audio::{
    AssistantState, VoiceAssistant, VoiceAssistantBuilder, VoiceAssistantConfig,
    VoiceAssistantHandler,
};
