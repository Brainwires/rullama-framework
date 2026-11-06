//! Sample conversion, resampling, WAV I/O and VAD trimming.
//!
//! Everything here is synchronous and allocation-only (no devices), so it is
//! shared by the capture path, `rla_transcribe_wav` and the tests.

use std::path::Path;

use rullama_hardware::audio::types::{AudioBuffer, AudioConfig, SampleFormat};
use rullama_hardware::audio::vad::VoiceActivityDetector;

/// Sample rate whisper.cpp expects (and the rate capture is resampled to).
pub const TARGET_RATE: u32 = 16_000;
/// Silence kept on each side of the detected speech so trimming never clips it.
pub const PAD_MS: u32 = 250;
/// Less detected speech than this yields an empty transcript.
pub const MIN_SPEECH_MS: u32 = 300;
/// VAD analysis frame.
const VAD_FRAME_MS: u32 = 30;

/// Interleaved PCM as `f32` in `[-1, 1]` plus its layout.
#[derive(Debug, Clone)]
pub struct Pcm {
    /// Interleaved samples.
    pub samples: Vec<f32>,
    /// Channel count (>= 1).
    pub channels: usize,
    /// Sample rate in Hz.
    pub rate: u32,
}

/// Convert a decoded [`AudioBuffer`] (16-bit int or 32-bit float) to `f32`.
pub fn buffer_to_pcm(buffer: &AudioBuffer) -> Pcm {
    let samples = match buffer.config.sample_format {
        SampleFormat::I16 => buffer
            .data
            .chunks_exact(2)
            .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
            .collect(),
        SampleFormat::F32 => buffer
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    };
    Pcm {
        samples,
        channels: usize::from(buffer.config.channels.max(1)),
        rate: buffer.config.sample_rate,
    }
}

/// Decode WAV bytes (any rate, any channel count, 16-bit int or 32-bit float).
pub fn decode_wav_pcm(bytes: &[u8]) -> Result<Pcm, String> {
    let buffer = rullama_hardware::audio::decode_wav(bytes).map_err(|e| e.to_string())?;
    if buffer.config.sample_rate == 0 {
        return Err("WAV has a sample rate of 0".to_string());
    }
    Ok(buffer_to_pcm(&buffer))
}

/// Read and decode a WAV file.
pub fn read_wav_pcm(path: &Path) -> Result<Pcm, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    decode_wav_pcm(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Average interleaved channels down to mono.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample mono audio with linear interpolation. When downsampling, a box
/// filter as wide as the decimation ratio runs first as a cheap anti-alias.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() || from == 0 || to == 0 {
        return input.to_vec();
    }
    let filtered;
    let src: &[f32] = if from > to {
        let width = (from as f64 / to as f64).round() as usize;
        if width > 1 {
            filtered = box_filter(input, width);
            &filtered
        } else {
            input
        }
    } else {
        input
    };
    let out_len = (src.len() as u64 * u64::from(to) / u64::from(from)) as usize;
    let step = f64::from(from) / f64::from(to);
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * step;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = src[idx.min(src.len() - 1)];
            let b = src.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

/// Centered moving average of `width` samples.
fn box_filter(input: &[f32], width: usize) -> Vec<f32> {
    let half = width / 2;
    let mut prefix = Vec::with_capacity(input.len() + 1);
    prefix.push(0.0f64);
    let mut acc = 0.0f64;
    for &s in input {
        acc += f64::from(s);
        prefix.push(acc);
    }
    (0..input.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + width - half).min(input.len());
            ((prefix[hi] - prefix[lo]) / (hi - lo) as f64) as f32
        })
        .collect()
}

/// Mono 16 kHz from any [`Pcm`].
pub fn to_mono_16k(pcm: &Pcm) -> Vec<f32> {
    resample(&downmix(&pcm.samples, pcm.channels), pcm.rate, TARGET_RATE)
}

/// RMS of `samples`, clamped to `0..=1`.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_sq = samples
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / samples.len() as f64;
    (mean_sq.sqrt() as f32).clamp(0.0, 1.0)
}

fn f32_to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// The speech-bearing part of a 16 kHz mono recording.
#[derive(Debug, Clone, PartialEq)]
pub struct Trimmed {
    /// Samples from (first speech - pad) to (last speech + pad).
    pub samples: Vec<f32>,
    /// Total ms the VAD classified as speech.
    pub speech_ms: u32,
}

impl Trimmed {
    /// Too little speech to be worth transcribing.
    pub fn is_too_short(&self) -> bool {
        self.speech_ms < MIN_SPEECH_MS
    }
}

#[cfg(feature = "vad")]
fn detector() -> impl VoiceActivityDetector {
    // Least aggressive mode: the goal is to drop silence, never speech.
    rullama_hardware::audio::vad::WebRtcVad::new(rullama_hardware::audio::vad::VadMode::Quality)
}

#[cfg(not(feature = "vad"))]
fn detector() -> impl VoiceActivityDetector {
    rullama_hardware::audio::vad::EnergyVad::default()
}

/// Trim leading and trailing silence from 16 kHz mono audio, keeping
/// [`PAD_MS`] on each side of the outermost speech frames.
pub fn trim_silence(samples: &[f32]) -> Trimmed {
    let buffer = AudioBuffer::from_pcm(
        samples
            .iter()
            .flat_map(|&s| f32_to_i16(s).to_le_bytes())
            .collect(),
        AudioConfig::speech(),
    );
    let segments = detector().detect_segments(&buffer, VAD_FRAME_MS);
    let speech: Vec<_> = segments.iter().filter(|s| s.is_speech).collect();
    let (Some(first), Some(last)) = (speech.first(), speech.last()) else {
        return Trimmed {
            samples: Vec::new(),
            speech_ms: 0,
        };
    };
    let speech_samples: usize = speech.iter().map(|s| s.len()).sum();
    let pad = (TARGET_RATE * PAD_MS / 1000) as usize;
    let start = first.start_sample.saturating_sub(pad);
    let end = (last.end_sample + pad).min(samples.len());
    Trimmed {
        samples: samples[start..end].to_vec(),
        speech_ms: samples_to_ms(speech_samples, TARGET_RATE),
    }
}

/// Resample to 16 kHz mono and trim silence.
pub fn prepare(mono: &[f32], rate: u32) -> Trimmed {
    trim_silence(&resample(mono, rate, TARGET_RATE))
}

/// Duration of `n` samples at `rate`, in whole ms.
pub fn samples_to_ms(n: usize, rate: u32) -> u32 {
    if rate == 0 {
        return 0;
    }
    u32::try_from(n as u64 * 1000 / u64::from(rate)).unwrap_or(u32::MAX)
}

/// Write 16 kHz mono 16-bit PCM.
pub fn write_wav_16k(path: &Path, samples: &[f32]) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let err = |e: hound::Error| format!("cannot write {}: {e}", path.display());
    let mut writer = hound::WavWriter::create(path, spec).map_err(err)?;
    for &s in samples {
        writer.write_sample(f32_to_i16(s)).map_err(err)?;
    }
    writer.finalize().map_err(err)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The committed fixture: macOS `say` of "Run the tests in the second tile."
    pub(crate) const PHRASE_WAV: &[u8] = include_bytes!("../tests/fixtures/phrase.wav");

    pub(crate) fn phrase_16k() -> Vec<f32> {
        let pcm = decode_wav_pcm(PHRASE_WAV).expect("fixture decodes");
        assert_eq!((pcm.rate, pcm.channels), (16_000, 1));
        pcm.samples
    }

    fn sine(freq: f32, rate: u32, secs: f32, amp: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    fn zero_crossings(s: &[f32]) -> usize {
        s.windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count()
    }

    fn wav_bytes(spec: hound::WavSpec, interleaved: &[f32]) -> Vec<u8> {
        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut w = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for &s in interleaved {
            match spec.sample_format {
                hound::SampleFormat::Int => w.write_sample(f32_to_i16(s)).unwrap(),
                hound::SampleFormat::Float => w.write_sample(s).unwrap(),
            }
        }
        w.finalize().unwrap();
        cursor.into_inner()
    }

    #[test]
    fn resample_48k_to_16k_keeps_duration_and_pitch() {
        let input = sine(440.0, 48_000, 1.0, 0.5);
        let out = resample(&input, 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
        // 440 Hz for 1 s = 880 zero crossings.
        let zc = zero_crossings(&out) as i64;
        assert!((zc - 880).abs() <= 2, "zero crossings {zc}");
        let r = rms(&out);
        assert!((r - 0.5 / 2f32.sqrt()).abs() < 0.02, "rms {r}");
    }

    #[test]
    fn resample_upsamples_and_identity_is_noop() {
        let input = sine(300.0, 8_000, 0.5, 0.3);
        assert_eq!(resample(&input, 8_000, 8_000), input);
        let out = resample(&input, 8_000, 16_000);
        assert_eq!(out.len(), 8_000);
        let zc = zero_crossings(&out) as i64;
        assert!((zc - 300).abs() <= 2, "zero crossings {zc}");
    }

    #[test]
    fn decode_stereo_44k_i16_to_16k_mono() {
        let rate = 44_100;
        let left = sine(500.0, rate, 0.5, 0.4);
        // Right channel is the inverse of a different tone; mixing averages.
        let right: Vec<f32> = left.iter().map(|&s| s * 0.5).collect();
        let interleaved: Vec<f32> = left
            .iter()
            .zip(&right)
            .flat_map(|(&l, &r)| [l, r])
            .collect();
        let bytes = wav_bytes(
            hound::WavSpec {
                channels: 2,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
            &interleaved,
        );
        let pcm = decode_wav_pcm(&bytes).unwrap();
        assert_eq!((pcm.channels, pcm.rate), (2, rate));
        let mono = to_mono_16k(&pcm);
        assert!((mono.len() as i64 - 8_000).abs() <= 1, "len {}", mono.len());
        // Mixed amplitude = (0.4 + 0.2) / 2 = 0.3
        let r = rms(&mono);
        assert!((r - 0.3 / 2f32.sqrt()).abs() < 0.02, "rms {r}");
    }

    #[test]
    fn decode_mono_22k_f32_to_16k() {
        let input = sine(250.0, 22_050, 2.0, 0.25);
        let bytes = wav_bytes(
            hound::WavSpec {
                channels: 1,
                sample_rate: 22_050,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
            &input,
        );
        let mono = to_mono_16k(&decode_wav_pcm(&bytes).unwrap());
        assert!(
            (mono.len() as i64 - 32_000).abs() <= 1,
            "len {}",
            mono.len()
        );
        let zc = zero_crossings(&mono) as i64;
        assert!((zc - 1000).abs() <= 2, "zero crossings {zc}");
    }

    #[test]
    fn decode_rejects_24_bit_and_garbage() {
        let mut cursor = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::new(&mut cursor, spec).unwrap();
        w.write_sample(0i32).unwrap();
        w.finalize().unwrap();
        assert!(decode_wav_pcm(&cursor.into_inner()).is_err());
        assert!(decode_wav_pcm(b"not a wav file").is_err());
    }

    #[test]
    fn wav_write_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.wav");
        let input = sine(200.0, 16_000, 0.25, 0.5);
        write_wav_16k(&path, &input).unwrap();
        let pcm = read_wav_pcm(&path).unwrap();
        assert_eq!(
            (pcm.rate, pcm.channels, pcm.samples.len()),
            (16_000, 1, 4_000)
        );
        let max_err = input
            .iter()
            .zip(&pcm.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "max error {max_err}");
    }

    #[test]
    fn vad_trims_surrounding_silence_without_cutting_speech() {
        let speech = phrase_16k();
        let silence = vec![0.0f32; 16_000];
        let mut padded = silence.clone();
        padded.extend_from_slice(&speech);
        padded.extend_from_slice(&silence);

        let trimmed = trim_silence(&padded);
        assert!(
            trimmed.speech_ms >= 1_000,
            "speech {} ms",
            trimmed.speech_ms
        );
        assert!(!trimmed.is_too_short());
        // Most of the added 2 s of silence is gone...
        assert!(
            trimmed.samples.len() < speech.len() + 16_000,
            "trimmed {} vs speech {}",
            trimmed.samples.len(),
            speech.len()
        );
        // ...and every loud (speech) sample survived.
        let loud = |s: &[f32]| s.iter().filter(|x| x.abs() > 0.05).count();
        assert_eq!(loud(&trimmed.samples), loud(&speech));
    }

    #[test]
    fn vad_on_silence_or_a_blip_is_too_short() {
        let silent = trim_silence(&vec![0.0f32; 32_000]);
        assert!(silent.samples.is_empty());
        assert_eq!(silent.speech_ms, 0);
        assert!(silent.is_too_short());

        // 150 ms of speech inside silence.
        let speech = phrase_16k();
        let loud_at = speech.iter().position(|s| s.abs() > 0.1).unwrap();
        let mut blip = vec![0.0f32; 16_000];
        blip.extend_from_slice(&speech[loud_at..loud_at + 2_400]);
        blip.extend(vec![0.0f32; 16_000]);
        assert!(trim_silence(&blip).is_too_short());
    }

    #[test]
    fn prepare_resamples_before_trimming() {
        let speech = phrase_16k();
        let at_48k = resample(&speech, 16_000, 48_000);
        let t = prepare(&at_48k, 48_000);
        assert!(!t.is_too_short());
        assert!(t.samples.len() <= speech.len() + 1);
    }
}
