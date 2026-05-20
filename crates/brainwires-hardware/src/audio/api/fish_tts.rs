use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::fish::{FishClient, FishTtsRequest};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::tts::TextToSpeech;
use crate::audio::types::{AudioBuffer, OutputFormat, TtsOptions, Voice};
use crate::audio::wav::decode_wav;

/// Fish Audio text-to-speech implementation.
///
/// Wraps a [`FishClient`] from `brainwires-provider`.
pub struct FishTts {
    client: Arc<FishClient>,
}

impl FishTts {
    /// Create a new Fish TTS client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(FishClient::new(api_key));
        Self { client }
    }

    /// Create from an existing [`FishClient`].
    pub fn from_client(client: Arc<FishClient>) -> Self {
        Self { client }
    }
}

/// Map our [`OutputFormat`] to a Fish Audio format string.
fn format_to_fish(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Wav => "wav",
        OutputFormat::Mp3 => "mp3",
        OutputFormat::Pcm => "wav", // Fish doesn't support raw PCM; use WAV
        OutputFormat::Opus => "opus",
        OutputFormat::Flac => "flac",
    }
}

#[async_trait]
impl TextToSpeech for FishTts {
    fn name(&self) -> &str {
        "fish-tts"
    }

    async fn list_voices(&self) -> AudioResult<Vec<Voice>> {
        // Fish Audio does not have a voices listing API; return a static default.
        Ok(vec![Voice {
            id: "default".into(),
            name: Some("Default".into()),
            language: None,
        }])
    }

    async fn synthesize(&self, text: &str, options: &TtsOptions) -> AudioResult<AudioBuffer> {
        let req = FishTtsRequest {
            text: text.to_string(),
            reference_id: Some(options.voice.id.clone()),
            format: Some(format_to_fish(options.output_format).to_string()),
            speed: options.speed,
        };

        let bytes = self
            .client
            .tts(&req)
            .await
            .map_err(|e| AudioError::Api(format!("Fish TTS: {e}")))?;

        match options.output_format {
            OutputFormat::Wav | OutputFormat::Pcm => decode_wav(&bytes),
            _ => Err(AudioError::Unsupported(format!(
                "direct decoding of {:?} not supported; use Wav format",
                options.output_format
            ))),
        }
    }

    fn synthesize_stream(
        &self,
        text: &str,
        options: &TtsOptions,
    ) -> BoxStream<'static, AudioResult<AudioBuffer>> {
        let client = Arc::clone(&self.client);
        let text = text.to_string();
        let options = options.clone();

        let stream = async_stream::stream! {
            let tts = FishTts { client };
            yield tts.synthesize(&text, &options).await;
        };

        Box::pin(stream)
    }
}
