use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider::openai_chat::{CreateSpeechRequest, OpenAiClient};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::tts::TextToSpeech;
use crate::audio::types::{AudioBuffer, OutputFormat, TtsOptions, Voice};
use crate::audio::wav::decode_wav;

/// OpenAI TTS API text-to-speech implementation.
///
/// Wraps an [`OpenAiClient`] from `brainwires-provider` for the actual HTTP
/// transport; this struct adds the `TextToSpeech` trait and audio-domain logic.
pub struct OpenAiTts {
    client: Arc<OpenAiClient>,
    model: String,
}

impl OpenAiTts {
    /// Create a new OpenAI TTS client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(OpenAiClient::new(api_key.into(), "tts-1".to_string()));
        Self {
            client,
            model: "tts-1".to_string(),
        }
    }

    /// Create from an existing [`OpenAiClient`].
    pub fn from_client(client: Arc<OpenAiClient>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// Create with a custom base URL.
    pub fn new_with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let model_str = model.into();
        let client = Arc::new(
            OpenAiClient::new(api_key.into(), model_str.clone()).with_base_url(base_url.into()),
        );
        Self {
            client,
            model: model_str,
        }
    }

    /// Set the model name (e.g., "tts-1", "tts-1-hd").
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

fn format_to_string(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Wav => "wav",
        OutputFormat::Mp3 => "mp3",
        OutputFormat::Pcm => "pcm",
        OutputFormat::Opus => "opus",
        OutputFormat::Flac => "flac",
    }
}

#[async_trait]
impl TextToSpeech for OpenAiTts {
    fn name(&self) -> &str {
        "openai-tts"
    }

    async fn list_voices(&self) -> AudioResult<Vec<Voice>> {
        Ok(vec![
            Voice {
                id: "alloy".into(),
                name: Some("Alloy".into()),
                language: None,
            },
            Voice {
                id: "echo".into(),
                name: Some("Echo".into()),
                language: None,
            },
            Voice {
                id: "fable".into(),
                name: Some("Fable".into()),
                language: None,
            },
            Voice {
                id: "onyx".into(),
                name: Some("Onyx".into()),
                language: None,
            },
            Voice {
                id: "nova".into(),
                name: Some("Nova".into()),
                language: None,
            },
            Voice {
                id: "shimmer".into(),
                name: Some("Shimmer".into()),
                language: None,
            },
        ])
    }

    async fn synthesize(&self, text: &str, options: &TtsOptions) -> AudioResult<AudioBuffer> {
        let req = CreateSpeechRequest {
            model: self.model.clone(),
            input: text.to_string(),
            voice: options.voice.id.clone(),
            response_format: Some(format_to_string(options.output_format).to_string()),
            speed: options.speed.map(|s| s as f64),
        };

        let bytes = self
            .client
            .create_speech(&req)
            .await
            .map_err(|e| AudioError::Api(format!("OpenAI TTS: {e}")))?;

        match options.output_format {
            OutputFormat::Wav => decode_wav(&bytes),
            OutputFormat::Pcm => {
                let config = crate::audio::types::AudioConfig {
                    sample_rate: 24000,
                    channels: 1,
                    sample_format: crate::audio::types::SampleFormat::I16,
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
            let tts = OpenAiTts { client, model };
            yield tts.synthesize(&text, &options).await;
        };

        Box::pin(stream)
    }
}
