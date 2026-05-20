use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::deepgram::{DeepgramClient, DeepgramSpeakRequest};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::tts::TextToSpeech;
use crate::audio::types::{
    AudioBuffer, AudioConfig, OutputFormat, SampleFormat, TtsOptions, Voice,
};
use crate::audio::wav::decode_wav;

/// Deepgram Aura TTS text-to-speech implementation.
///
/// Wraps a [`DeepgramClient`] from `brainwires-provider` for the actual HTTP
/// transport; this struct adds the `TextToSpeech` trait and audio-domain logic.
pub struct DeepgramTts {
    client: Arc<DeepgramClient>,
    model: String,
}

impl DeepgramTts {
    /// Create a new Deepgram TTS client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(DeepgramClient::new(api_key));
        Self {
            client,
            model: "aura-asteria-en".to_string(),
        }
    }

    /// Create from an existing [`DeepgramClient`].
    pub fn from_client(client: Arc<DeepgramClient>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// Set the model/voice name (e.g., "aura-asteria-en").
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

/// Map [`OutputFormat`] to a Deepgram encoding string.
fn format_to_encoding(format: OutputFormat) -> Option<&'static str> {
    match format {
        OutputFormat::Wav => None, // Deepgram returns WAV by default when no encoding is set
        OutputFormat::Mp3 => Some("mp3"),
        OutputFormat::Pcm => Some("linear16"),
        OutputFormat::Opus => Some("opus"),
        OutputFormat::Flac => Some("flac"),
    }
}

#[async_trait]
impl TextToSpeech for DeepgramTts {
    fn name(&self) -> &str {
        "deepgram-tts"
    }

    async fn list_voices(&self) -> AudioResult<Vec<Voice>> {
        Ok(vec![
            Voice {
                id: "aura-asteria-en".into(),
                name: Some("Asteria".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-luna-en".into(),
                name: Some("Luna".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-stella-en".into(),
                name: Some("Stella".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-athena-en".into(),
                name: Some("Athena".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-hera-en".into(),
                name: Some("Hera".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-orion-en".into(),
                name: Some("Orion".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-arcas-en".into(),
                name: Some("Arcas".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-perseus-en".into(),
                name: Some("Perseus".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-angus-en".into(),
                name: Some("Angus".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-orpheus-en".into(),
                name: Some("Orpheus".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-helios-en".into(),
                name: Some("Helios".into()),
                language: Some("en".into()),
            },
            Voice {
                id: "aura-zeus-en".into(),
                name: Some("Zeus".into()),
                language: Some("en".into()),
            },
        ])
    }

    async fn synthesize(&self, text: &str, options: &TtsOptions) -> AudioResult<AudioBuffer> {
        let req = DeepgramSpeakRequest {
            text: text.to_string(),
            model: Some(options.voice.id.clone()),
            encoding: format_to_encoding(options.output_format).map(String::from),
            sample_rate: None,
        };

        let bytes = self
            .client
            .speak(&req)
            .await
            .map_err(|e| AudioError::Api(format!("Deepgram TTS: {e}")))?;

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
            let tts = DeepgramTts { client, model };
            yield tts.synthesize(&text, &options).await;
        };

        Box::pin(stream)
    }
}
