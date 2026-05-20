use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::cartesia::{
    CartesiaClient, CartesiaOutputFormat, CartesiaTtsRequest, CartesiaVoice,
};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::tts::TextToSpeech;
use crate::audio::types::{
    AudioBuffer, AudioConfig, OutputFormat, SampleFormat, TtsOptions, Voice,
};
use crate::audio::wav::decode_wav;

/// Cartesia text-to-speech implementation.
///
/// Wraps a [`CartesiaClient`] from `brainwires-provider`.
pub struct CartesiaTts {
    client: Arc<CartesiaClient>,
    model: String,
}

impl CartesiaTts {
    /// Create a new Cartesia TTS client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(CartesiaClient::new(api_key));
        Self {
            client,
            model: "sonic-english".to_string(),
        }
    }

    /// Create from an existing [`CartesiaClient`].
    pub fn from_client(client: Arc<CartesiaClient>) -> Self {
        Self {
            client,
            model: "sonic-english".to_string(),
        }
    }

    /// Set the model name (e.g., "sonic-english", "sonic-multilingual").
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

/// Build a Cartesia output format from our [`OutputFormat`].
fn build_output_format(format: OutputFormat) -> CartesiaOutputFormat {
    match format {
        OutputFormat::Wav => CartesiaOutputFormat {
            container: "wav".to_string(),
            encoding: "pcm_s16le".to_string(),
            sample_rate: 24000,
        },
        OutputFormat::Pcm => CartesiaOutputFormat {
            container: "raw".to_string(),
            encoding: "pcm_s16le".to_string(),
            sample_rate: 24000,
        },
        // For unsupported containers, default to raw PCM.
        _ => CartesiaOutputFormat {
            container: "raw".to_string(),
            encoding: "pcm_s16le".to_string(),
            sample_rate: 24000,
        },
    }
}

#[async_trait]
impl TextToSpeech for CartesiaTts {
    fn name(&self) -> &str {
        "cartesia-tts"
    }

    async fn list_voices(&self) -> AudioResult<Vec<Voice>> {
        // Cartesia does not expose a public voices listing API; return well-known defaults.
        Ok(vec![
            Voice {
                id: "a0e99841-438c-4a64-b679-ae501e7d6091".into(),
                name: Some("Barbershop Man".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "156fb8d2-335b-4950-9cb3-a2d33f0b4cf7".into(),
                name: Some("British Lady".into()),
                language: Some("en".into()),
            },
        ])
    }

    async fn synthesize(&self, text: &str, options: &TtsOptions) -> AudioResult<AudioBuffer> {
        let output_format = build_output_format(options.output_format);

        let req = CartesiaTtsRequest {
            model_id: self.model.clone(),
            transcript: text.to_string(),
            voice: CartesiaVoice {
                mode: "id".to_string(),
                id: Some(options.voice.id.clone()),
            },
            output_format,
            language: options.voice.language.clone(),
        };

        let bytes = self
            .client
            .tts_bytes(&req)
            .await
            .map_err(|e| AudioError::Api(format!("Cartesia TTS: {e}")))?;

        match options.output_format {
            OutputFormat::Wav => decode_wav(&bytes),
            OutputFormat::Pcm => {
                let config = AudioConfig {
                    sample_rate: 24000,
                    channels: 1,
                    sample_format: SampleFormat::I16,
                };
                Ok(AudioBuffer::from_pcm(bytes, config))
            }
            _ => Err(AudioError::Unsupported(format!(
                "direct decoding of {:?} not supported; use Wav or Pcm format",
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
        let model = self.model.clone();
        let text = text.to_string();
        let options = options.clone();

        let stream = async_stream::stream! {
            let tts = CartesiaTts { client, model };
            yield tts.synthesize(&text, &options).await;
        };

        Box::pin(stream)
    }
}
