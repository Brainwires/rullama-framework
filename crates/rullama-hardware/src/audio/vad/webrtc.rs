use webrtc_vad::{Vad, VadMode as WrtcMode};

use crate::audio::types::AudioBuffer;
use crate::audio::vad::{SpeechSegment, VoiceActivityDetector, pcm_to_i16_mono};

/// WebRTC VAD aggressiveness mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadMode {
    /// Least aggressive — best recall, more false positives.
    Quality,
    /// Low-bitrate mode — balanced.
    LowBitrate,
    /// Aggressive mode — fewer false positives.
    Aggressive,
    /// Most aggressive — fewest false positives, may miss soft speech.
    VeryAggressive,
}

impl From<VadMode> for WrtcMode {
    fn from(m: VadMode) -> Self {
        match m {
            VadMode::Quality => WrtcMode::Quality,
            VadMode::LowBitrate => WrtcMode::LowBitrate,
            VadMode::Aggressive => WrtcMode::Aggressive,
            VadMode::VeryAggressive => WrtcMode::VeryAggressive,
        }
    }
}

/// Voice Activity Detector backed by the WebRTC VAD algorithm.
///
/// More accurate than energy-based detection, especially in noisy environments.
/// Supports 8 kHz, 16 kHz, 32 kHz, and 48 kHz sample rates with 10, 20, or
/// 30 ms frames.
///
/// Feature: `vad`
pub struct WebRtcVad {
    /// Aggressiveness mode controlling the speech/silence threshold.
    pub mode: VadMode,
}

impl Default for WebRtcVad {
    fn default() -> Self {
        Self {
            mode: VadMode::Aggressive,
        }
    }
}

impl WebRtcVad {
    /// Create a new WebRTC VAD with the given aggressiveness mode.
    pub fn new(mode: VadMode) -> Self {
        Self { mode }
    }
}

impl VoiceActivityDetector for WebRtcVad {
    fn is_speech(&self, audio: &AudioBuffer) -> bool {
        let sr = audio.config.sample_rate;
        if !matches!(sr, 8000 | 16000 | 32000 | 48000) {
            // Unsupported sample rate — fall back to energy check
            let energy_vad = crate::audio::vad::energy::EnergyVad::default();
            return energy_vad.is_speech(audio);
        }

        let mut vad = Vad::new_with_rate_and_mode(sr_to_wrtc(sr), WrtcMode::from(self.mode));

        let samples = pcm_to_i16_mono(audio);
        // WebRTC requires exactly 10, 20, or 30 ms frames
        let frame_size = (sr / 100) as usize; // 10 ms
        let mut any_speech = false;
        for frame in samples.chunks(frame_size) {
            if frame.len() < frame_size {
                break;
            }
            if vad.is_voice_segment(frame).unwrap_or(false) {
                any_speech = true;
                break;
            }
        }
        any_speech
    }

    fn detect_segments(&self, audio: &AudioBuffer, frame_ms: u32) -> Vec<SpeechSegment> {
        let frame_ms = frame_ms.clamp(10, 30);
        let sr = audio.config.sample_rate;
        if !matches!(sr, 8000 | 16000 | 32000 | 48000) {
            return crate::audio::vad::energy::EnergyVad::default()
                .detect_segments(audio, frame_ms);
        }

        let mut vad = Vad::new_with_rate_and_mode(sr_to_wrtc(sr), WrtcMode::from(self.mode));

        let samples = pcm_to_i16_mono(audio);
        let frame_size = (sr * frame_ms / 1000) as usize;
        let mut segments: Vec<SpeechSegment> = Vec::new();

        for (i, frame) in samples.chunks(frame_size).enumerate() {
            if frame.len() < frame_size {
                break;
            }
            let is_speech = vad.is_voice_segment(frame).unwrap_or(false);
            let start = i * frame_size;
            let end = start + frame_size;

            match segments.last_mut() {
                Some(last) if last.is_speech == is_speech => {
                    last.end_sample = end;
                }
                _ => segments.push(SpeechSegment {
                    is_speech,
                    start_sample: start,
                    end_sample: end,
                }),
            }
        }

        segments
    }
}

fn sr_to_wrtc(sr: u32) -> webrtc_vad::SampleRate {
    match sr {
        8000 => webrtc_vad::SampleRate::Rate8kHz,
        16000 => webrtc_vad::SampleRate::Rate16kHz,
        32000 => webrtc_vad::SampleRate::Rate32kHz,
        _ => webrtc_vad::SampleRate::Rate48kHz,
    }
}
