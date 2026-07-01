use async_trait::async_trait;
use cpal::traits::{DeviceTrait, StreamTrait};
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::audio::capture::AudioCapture;
use crate::audio::device::AudioDevice;
use crate::audio::error::{AudioError, AudioResult};
use crate::audio::types::{AudioBuffer, AudioConfig, SampleFormat};

use super::cpal_common;

/// Audio capture implementation using cpal.
pub struct CpalCapture;

impl CpalCapture {
    /// Create a new cpal-based audio capture instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpalCapture {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AudioCapture for CpalCapture {
    fn list_devices(&self) -> AudioResult<Vec<AudioDevice>> {
        cpal_common::list_input_devices()
    }

    fn default_device(&self) -> AudioResult<Option<AudioDevice>> {
        let devices = cpal_common::list_input_devices()?;
        Ok(devices.into_iter().find(|d| d.is_default))
    }

    fn start_capture(
        &self,
        device: Option<&AudioDevice>,
        config: &AudioConfig,
    ) -> AudioResult<BoxStream<'static, AudioResult<AudioBuffer>>> {
        let cpal_device = cpal_common::find_input_device(device)?;
        let stream_config = cpal_common::build_stream_config(config);
        let sample_format = config.sample_format;
        let audio_config = config.clone();

        // Bounded channel to bridge cpal's callback thread to async.
        // A separate stop channel signals when the consumer drops.
        let (tx, rx) = mpsc::channel::<AudioResult<AudioBuffer>>(64);
        let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);

        // Spawn cpal stream on a dedicated thread since cpal::Stream is !Send.
        std::thread::spawn(move || {
            let stream = match sample_format {
                SampleFormat::I16 => {
                    let cfg = audio_config.clone();
                    let tx = tx.clone();
                    cpal_device.build_input_stream(
                        &stream_config,
                        move |data: &[i16], _: &cpal::InputCallbackInfo| {
                            let bytes: Vec<u8> =
                                data.iter().flat_map(|s| s.to_le_bytes()).collect();
                            let buffer = AudioBuffer::from_pcm(bytes, cfg.clone());
                            let _ = tx.try_send(Ok(buffer));
                        },
                        move |err| {
                            tracing::error!("cpal capture error: {err}");
                        },
                        None,
                    )
                }
                SampleFormat::F32 => {
                    let cfg = audio_config.clone();
                    let tx = tx.clone();
                    cpal_device.build_input_stream(
                        &stream_config,
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            let bytes: Vec<u8> =
                                data.iter().flat_map(|s| s.to_le_bytes()).collect();
                            let buffer = AudioBuffer::from_pcm(bytes, cfg.clone());
                            let _ = tx.try_send(Ok(buffer));
                        },
                        move |err| {
                            tracing::error!("cpal capture error: {err}");
                        },
                        None,
                    )
                }
            };

            match stream {
                Ok(stream) => {
                    if let Err(e) = stream.play() {
                        let _ = tx.try_send(Err(AudioError::Capture(format!(
                            "failed to start capture: {e}"
                        ))));
                        return;
                    }
                    // Block this thread until stop signal received.
                    // The stream stays alive as long as we hold it.
                    let _ = stop_rx.blocking_recv();
                    drop(stream);
                }
                Err(e) => {
                    let _ = tx.try_send(Err(AudioError::Capture(format!(
                        "failed to build input stream: {e}"
                    ))));
                }
            }
        });

        // Wrap receiver as a stream. When this stream is dropped, stop_tx drops too,
        // which signals the capture thread to exit.
        let receiver_stream = ReceiverStream::new(rx);
        let output = receiver_stream.map(move |item| {
            let _keep_alive = &stop_tx;
            item
        });

        Ok(Box::pin(output))
    }

    async fn record(
        &self,
        device: Option<&AudioDevice>,
        config: &AudioConfig,
        duration_secs: f64,
    ) -> AudioResult<AudioBuffer> {
        let mut stream = self.start_capture(device, config)?;
        let mut all_data = Vec::new();
        let target_bytes =
            (config.sample_rate as f64 * duration_secs) as usize * config.bytes_per_frame();

        while let Some(result) = stream.next().await {
            let buffer = result?;
            all_data.extend_from_slice(&buffer.data);
            if all_data.len() >= target_bytes {
                all_data.truncate(target_bytes);
                break;
            }
        }

        Ok(AudioBuffer::from_pcm(all_data, config.clone()))
    }
}
