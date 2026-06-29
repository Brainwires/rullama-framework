use async_trait::async_trait;
use cpal::traits::{DeviceTrait, StreamTrait};
use futures::StreamExt;
use futures::stream::BoxStream;
use std::sync::{Arc, Mutex};

use crate::audio::device::AudioDevice;
use crate::audio::error::AudioResult;
use crate::audio::playback::AudioPlayback;
use crate::audio::types::{AudioBuffer, AudioConfig, SampleFormat};

use super::cpal_common;

/// Audio playback implementation using cpal.
pub struct CpalPlayback;

impl CpalPlayback {
    /// Create a new cpal-based audio playback instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpalPlayback {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AudioPlayback for CpalPlayback {
    fn list_devices(&self) -> AudioResult<Vec<AudioDevice>> {
        cpal_common::list_output_devices()
    }

    fn default_device(&self) -> AudioResult<Option<AudioDevice>> {
        let devices = cpal_common::list_output_devices()?;
        Ok(devices.into_iter().find(|d| d.is_default))
    }

    async fn play(&self, device: Option<&AudioDevice>, buffer: &AudioBuffer) -> AudioResult<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        let cpal_device = cpal_common::find_output_device(device)?;
        let stream_config = cpal_common::build_stream_config(&buffer.config);
        let sample_format = buffer.config.sample_format;

        // Shared cursor into the audio data
        let data = Arc::new(buffer.data.clone());
        let pos = Arc::new(Mutex::new(0usize));
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let done_tx = Arc::new(Mutex::new(Some(done_tx)));

        // Spawn playback on a dedicated thread since cpal::Stream is !Send.
        std::thread::spawn(move || {
            let stream = match sample_format {
                SampleFormat::I16 => {
                    let data = data.clone();
                    let pos = pos.clone();
                    let done_tx = done_tx.clone();
                    cpal_device.build_output_stream(
                        &stream_config,
                        move |output: &mut [i16], _: &cpal::OutputCallbackInfo| {
                            let mut p = pos.lock().expect("audio playback position lock poisoned");
                            for sample in output.iter_mut() {
                                if *p + 2 <= data.len() {
                                    *sample = i16::from_le_bytes([data[*p], data[*p + 1]]);
                                    *p += 2;
                                } else {
                                    *sample = 0;
                                    if let Some(tx) = done_tx
                                        .lock()
                                        .expect("audio playback done signal lock poisoned")
                                        .take()
                                    {
                                        let _ = tx.send(());
                                    }
                                }
                            }
                        },
                        move |err| {
                            tracing::error!("cpal playback error: {err}");
                        },
                        None,
                    )
                }
                SampleFormat::F32 => {
                    let data = data.clone();
                    let pos = pos.clone();
                    let done_tx = done_tx.clone();
                    cpal_device.build_output_stream(
                        &stream_config,
                        move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                            let mut p = pos.lock().expect("audio playback position lock poisoned");
                            for sample in output.iter_mut() {
                                if *p + 4 <= data.len() {
                                    *sample = f32::from_le_bytes([
                                        data[*p],
                                        data[*p + 1],
                                        data[*p + 2],
                                        data[*p + 3],
                                    ]);
                                    *p += 4;
                                } else {
                                    *sample = 0.0;
                                    if let Some(tx) = done_tx
                                        .lock()
                                        .expect("audio playback done signal lock poisoned")
                                        .take()
                                    {
                                        let _ = tx.send(());
                                    }
                                }
                            }
                        },
                        move |err| {
                            tracing::error!("cpal playback error: {err}");
                        },
                        None,
                    )
                }
            };

            match stream {
                Ok(stream) => {
                    if let Err(e) = stream.play() {
                        tracing::error!("failed to start playback: {e}");
                        return;
                    }
                    // Block until done_tx fires (via the output callback)
                    // or until all data has been consumed
                    std::thread::park_timeout(std::time::Duration::from_secs(300));
                    drop(stream);
                }
                Err(e) => {
                    tracing::error!("failed to build output stream: {e}");
                }
            }
        });

        // Wait for playback to finish
        let _ = done_rx.await;

        // Small delay to let the audio device flush
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        Ok(())
    }

    async fn play_stream(
        &self,
        device: Option<&AudioDevice>,
        _config: &AudioConfig,
        mut stream: BoxStream<'static, AudioResult<AudioBuffer>>,
    ) -> AudioResult<()> {
        while let Some(result) = stream.next().await {
            let buffer = result?;
            self.play(device, &buffer).await?;
        }
        Ok(())
    }
}
