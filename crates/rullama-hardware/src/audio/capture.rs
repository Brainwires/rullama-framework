use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::audio::device::AudioDevice;
use crate::audio::error::AudioResult;
use crate::audio::types::{AudioBuffer, AudioConfig};

/// Captures audio from a hardware input device (microphone).
#[async_trait]
pub trait AudioCapture: Send + Sync {
    /// List available input devices.
    fn list_devices(&self) -> AudioResult<Vec<AudioDevice>>;

    /// Get the default input device, if one exists.
    fn default_device(&self) -> AudioResult<Option<AudioDevice>>;

    /// Start capturing audio, returning a stream of audio buffers.
    ///
    /// Each yielded [`AudioBuffer`] contains a chunk of PCM audio data.
    /// The stream continues until the capture is stopped or an error occurs.
    fn start_capture(
        &self,
        device: Option<&AudioDevice>,
        config: &AudioConfig,
    ) -> AudioResult<BoxStream<'static, AudioResult<AudioBuffer>>>;

    /// Record a fixed duration of audio and return as a single buffer.
    async fn record(
        &self,
        device: Option<&AudioDevice>,
        config: &AudioConfig,
        duration_secs: f64,
    ) -> AudioResult<AudioBuffer>;
}
