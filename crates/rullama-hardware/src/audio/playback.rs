use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::audio::device::AudioDevice;
use crate::audio::error::AudioResult;
use crate::audio::types::{AudioBuffer, AudioConfig};

/// Plays audio through a hardware output device (speakers).
#[async_trait]
pub trait AudioPlayback: Send + Sync {
    /// List available output devices.
    fn list_devices(&self) -> AudioResult<Vec<AudioDevice>>;

    /// Get the default output device, if one exists.
    fn default_device(&self) -> AudioResult<Option<AudioDevice>>;

    /// Play a complete audio buffer through the output device.
    ///
    /// Blocks (asynchronously) until playback completes.
    async fn play(&self, device: Option<&AudioDevice>, buffer: &AudioBuffer) -> AudioResult<()>;

    /// Play audio from a stream of buffers (real-time streaming playback).
    ///
    /// Each buffer is queued and played in order. Returns when the stream ends.
    async fn play_stream(
        &self,
        device: Option<&AudioDevice>,
        config: &AudioConfig,
        stream: BoxStream<'static, AudioResult<AudioBuffer>>,
    ) -> AudioResult<()>;
}
