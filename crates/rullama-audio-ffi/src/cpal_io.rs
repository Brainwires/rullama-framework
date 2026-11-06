//! cpal capture and playback on the default devices.
//!
//! rullama-hardware's `CpalCapture` / `CpalPlayback` open the device at the
//! caller's requested rate and format (16 kHz i16 is often rejected by
//! CoreAudio / ALSA), report stream-build failures only asynchronously, and
//! playback cannot be stopped mid-clip. The C ABI needs synchronous
//! "no device" errors, the device's native format, a level meter and
//! stoppable playback, so this module drives cpal directly. A `cpal::Stream`
//! is `!Send`, so each stream lives on its own thread.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};

use crate::dsp::{self, Pcm};
use crate::engine::{AudioSource, CaptureHandle, CaptureSink, SourceError, lock};
use crate::error;

/// How long to wait for a device to open before giving up.
const OPEN_TIMEOUT: Duration = Duration::from_secs(5);
/// After the last sample is handed to the device, let it drain this long.
const DRAIN: Duration = Duration::from_millis(150);

/// Stops a stream thread when dropped (and waits for it).
struct StreamThread {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for StreamThread {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ── Capture ─────────────────────────────────────────────────────────────────

/// Captures the default input device in its native format, mixing to mono.
pub struct CpalSource;

fn build_input<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sink: Arc<CaptureSink>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels.max(1));
    let mut mono = Vec::new();
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            mono.clear();
            mono.extend(data.chunks(channels).map(|frame| {
                frame.iter().map(|&s| s.to_sample::<f32>()).sum::<f32>() / frame.len() as f32
            }));
            sink.push(&mono);
        },
        |_err| {},
        None,
    )
}

fn open_input(sink: Arc<CaptureSink>) -> Result<cpal::Stream, SourceError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| SourceError::NoDevice("no default audio input device".into()))?;
    let supported = device
        .default_input_config()
        .map_err(|e| SourceError::NoDevice(format!("input device unavailable: {e}")))?;
    sink.configure(supported.sample_rate().0);
    let config = supported.config();
    let built = match supported.sample_format() {
        cpal::SampleFormat::F32 => build_input::<f32>(&device, &config, sink),
        cpal::SampleFormat::F64 => build_input::<f64>(&device, &config, sink),
        cpal::SampleFormat::I8 => build_input::<i8>(&device, &config, sink),
        cpal::SampleFormat::I16 => build_input::<i16>(&device, &config, sink),
        cpal::SampleFormat::I32 => build_input::<i32>(&device, &config, sink),
        cpal::SampleFormat::U8 => build_input::<u8>(&device, &config, sink),
        cpal::SampleFormat::U16 => build_input::<u16>(&device, &config, sink),
        cpal::SampleFormat::U32 => build_input::<u32>(&device, &config, sink),
        other => {
            return Err(SourceError::Failed(format!(
                "unsupported input sample format {other:?}"
            )));
        }
    };
    let stream = built.map_err(|e| match e {
        cpal::BuildStreamError::DeviceNotAvailable => {
            SourceError::NoDevice("audio input device not available".into())
        }
        e => SourceError::Failed(format!("cannot open audio input: {e}")),
    })?;
    stream
        .play()
        .map_err(|e| SourceError::Failed(format!("cannot start audio input: {e}")))?;
    Ok(stream)
}

impl AudioSource for CpalSource {
    fn open(&self, sink: Arc<CaptureSink>) -> Result<CaptureHandle, SourceError> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("rla-capture".into())
            .spawn(move || match open_input(sink) {
                Ok(stream) => {
                    let _ = ready_tx.send(Ok(()));
                    // Runs until the handle sends stop or is dropped.
                    let _ = stop_rx.recv();
                    drop(stream);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| SourceError::Failed(format!("cannot start capture thread: {e}")))?;
        let handle = StreamThread {
            stop: Some(stop_tx),
            thread: Some(thread),
        };
        match ready_rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok(())) => Ok(Box::new(handle)),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(SourceError::Failed("timed out opening audio input".into())),
        }
    }
}

// ── Playback ────────────────────────────────────────────────────────────────

/// Plays one clip at a time on the default output device.
#[derive(Default)]
pub struct Player {
    current: Mutex<Option<Playback>>,
}

struct Playback {
    playing: Arc<AtomicBool>,
    // Dropping it stops playback.
    _thread: StreamThread,
}

/// Convert `pcm` to the device's rate and channel count (interleaved).
fn adapt(pcm: &Pcm, rate: u32, channels: usize) -> Vec<f32> {
    let src_ch = pcm.channels.max(1);
    let per_channel: Vec<Vec<f32>> = if channels == 1 {
        vec![dsp::resample(
            &dsp::downmix(&pcm.samples, src_ch),
            pcm.rate,
            rate,
        )]
    } else {
        (0..src_ch)
            .map(|c| {
                let ch: Vec<f32> = pcm
                    .samples
                    .iter()
                    .skip(c)
                    .step_by(src_ch)
                    .copied()
                    .collect();
                dsp::resample(&ch, pcm.rate, rate)
            })
            .collect()
    };
    let frames = per_channel.iter().map(Vec::len).min().unwrap_or(0);
    let mut out = Vec::with_capacity(frames * channels);
    let last = per_channel.len() - 1;
    let sources: Vec<&[f32]> = (0..channels)
        .map(|c| &per_channel[c.min(last)][..frames])
        .collect();
    for f in 0..frames {
        out.extend(sources.iter().map(|src| src[f]));
    }
    out
}

fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    data: Vec<f32>,
    done: mpsc::Sender<()>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample + FromSample<f32>,
{
    let mut pos = 0usize;
    let mut signalled = false;
    device.build_output_stream(
        config,
        move |out: &mut [T], _: &cpal::OutputCallbackInfo| {
            for s in out.iter_mut() {
                *s = match data.get(pos) {
                    Some(&v) => {
                        pos += 1;
                        v.to_sample::<T>()
                    }
                    None => T::EQUILIBRIUM,
                };
            }
            if pos >= data.len() && !signalled {
                signalled = true;
                let _ = done.send(());
            }
        },
        |_err| {},
        None,
    )
}

fn open_output(pcm: &Pcm, done: mpsc::Sender<()>) -> Result<cpal::Stream, (i32, String)> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or((
        error::NO_DEVICE,
        "no default audio output device".to_string(),
    ))?;
    let supported = device
        .default_output_config()
        .map_err(|e| (error::NO_DEVICE, format!("output device unavailable: {e}")))?;
    let config = supported.config();
    let data = adapt(
        pcm,
        config.sample_rate.0,
        usize::from(config.channels.max(1)),
    );
    let built = match supported.sample_format() {
        cpal::SampleFormat::F32 => build_output::<f32>(&device, &config, data, done),
        cpal::SampleFormat::F64 => build_output::<f64>(&device, &config, data, done),
        cpal::SampleFormat::I8 => build_output::<i8>(&device, &config, data, done),
        cpal::SampleFormat::I16 => build_output::<i16>(&device, &config, data, done),
        cpal::SampleFormat::I32 => build_output::<i32>(&device, &config, data, done),
        cpal::SampleFormat::U8 => build_output::<u8>(&device, &config, data, done),
        cpal::SampleFormat::U16 => build_output::<u16>(&device, &config, data, done),
        cpal::SampleFormat::U32 => build_output::<u32>(&device, &config, data, done),
        other => {
            return Err((
                error::ERROR,
                format!("unsupported output sample format {other:?}"),
            ));
        }
    };
    let stream = built.map_err(|e| (error::ERROR, format!("cannot open audio output: {e}")))?;
    stream
        .play()
        .map_err(|e| (error::ERROR, format!("cannot start audio output: {e}")))?;
    Ok(stream)
}

impl Player {
    /// Stop any clip and start playing `path`. Returns once the device is
    /// open; `Err` carries the return code and message.
    pub fn play(&self, path: &Path) -> Result<(), (i32, String)> {
        let mut current = lock(&self.current);
        if let Some(p) = current.take() {
            // Stop what is playing, without the drain pause.
            p.playing.store(false, Ordering::SeqCst);
        }
        let pcm = dsp::read_wav_pcm(path).map_err(|e| (error::ERROR, e))?;

        let playing = Arc::new(AtomicBool::new(true));
        let flag = playing.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let done_tx = stop_tx.clone();
        let thread = std::thread::Builder::new()
            .name("rla-playback".into())
            .spawn(move || {
                match open_output(&pcm, done_tx) {
                    Ok(stream) => {
                        let _ = ready_tx.send(Ok(()));
                        // Either the callback ran out of samples or stop() was called.
                        let _ = stop_rx.recv();
                        if flag.load(Ordering::SeqCst) {
                            std::thread::sleep(DRAIN);
                        }
                        drop(stream);
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                }
                flag.store(false, Ordering::SeqCst);
            })
            .map_err(|e| (error::ERROR, format!("cannot start playback thread: {e}")))?;
        let handle = StreamThread {
            stop: Some(stop_tx),
            thread: Some(thread),
        };
        match ready_rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok(())) => {
                *current = Some(Playback {
                    playing,
                    _thread: handle,
                });
                Ok(())
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Err((error::ERROR, "timed out opening audio output".to_string())),
        }
    }

    /// Stop playback (no-op when idle).
    pub fn stop(&self) {
        let taken = lock(&self.current).take();
        if let Some(p) = taken {
            // Skip the drain sleep: this is an explicit stop.
            p.playing.store(false, Ordering::SeqCst);
            drop(p);
        }
    }

    /// Whether a clip is playing.
    pub fn playing(&self) -> bool {
        lock(&self.current)
            .as_ref()
            .is_some_and(|p| p.playing.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapt_mono_to_stereo_48k() {
        let pcm = Pcm {
            samples: vec![0.5; 16_000],
            channels: 1,
            rate: 16_000,
        };
        let out = adapt(&pcm, 48_000, 2);
        assert_eq!(out.len(), 96_000);
        assert!(out.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn adapt_stereo_keeps_channels_and_mixes_for_mono() {
        let samples: Vec<f32> = (0..1_000).flat_map(|_| [0.2f32, -0.2]).collect();
        let pcm = Pcm {
            samples,
            channels: 2,
            rate: 8_000,
        };
        let stereo = adapt(&pcm, 8_000, 2);
        assert_eq!(&stereo[..4], &[0.2, -0.2, 0.2, -0.2]);
        let mono = adapt(&pcm, 8_000, 1);
        assert_eq!(mono.len(), 1_000);
        assert!(mono.iter().all(|s| s.abs() < 1e-6));
    }
}
