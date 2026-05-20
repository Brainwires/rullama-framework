use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use futures::stream::BoxStream;

use brainwires_provider::openai_responses::{
    AudioOutputConfig, CreateResponseRequest, OutputContentBlock, ResponseInput,
    ResponseOutputItem, ResponsesClient,
};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::tts::TextToSpeech;
use crate::audio::types::{
    AudioBuffer, AudioConfig, OutputFormat, SampleFormat, TtsOptions, Voice,
};
use crate::audio::wav::decode_wav;

/// OpenAI Responses API text-to-speech implementation.
///
/// Uses the Responses API with audio output modality instead of the dedicated
/// `/v1/audio/speech` endpoint. Requires an audio-capable model such as
/// `gpt-4o-audio-preview` or `gpt-4o-mini-audio-preview`.
pub struct OpenAiResponsesTts {
    client: Arc<ResponsesClient>,
    model: String,
}

impl OpenAiResponsesTts {
    /// Create a new Responses API TTS client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(ResponsesClient::new(api_key.into()));
        Self {
            client,
            model: "gpt-4o-audio-preview".to_string(),
        }
    }

    /// Create from an existing [`ResponsesClient`].
    pub fn from_client(client: Arc<ResponsesClient>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// Set the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

fn output_format_to_responses_format(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Wav => "wav",
        OutputFormat::Mp3 => "mp3",
        OutputFormat::Flac => "flac",
        OutputFormat::Opus => "opus",
        OutputFormat::Pcm => "pcm16",
    }
}

/// Extract the first `OutputAudio` block from a response.
fn extract_audio_block(output: &[ResponseOutputItem]) -> Option<(&str, Option<&str>)> {
    for item in output {
        if let ResponseOutputItem::Message { content, .. } = item {
            for block in content {
                if let OutputContentBlock::OutputAudio {
                    data, transcript, ..
                } = block
                {
                    return Some((data.as_str(), transcript.as_deref()));
                }
            }
        }
    }
    None
}

#[async_trait]
impl TextToSpeech for OpenAiResponsesTts {
    fn name(&self) -> &str {
        "openai-responses-tts"
    }

    async fn list_voices(&self) -> AudioResult<Vec<Voice>> {
        Ok(vec![
            Voice {
                id: "alloy".into(),
                name: Some("Alloy".into()),
                language: None,
            },
            Voice {
                id: "ash".into(),
                name: Some("Ash".into()),
                language: None,
            },
            Voice {
                id: "ballad".into(),
                name: Some("Ballad".into()),
                language: None,
            },
            Voice {
                id: "coral".into(),
                name: Some("Coral".into()),
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
                id: "sage".into(),
                name: Some("Sage".into()),
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
        let format_str = output_format_to_responses_format(options.output_format);

        let mut req =
            CreateResponseRequest::new(self.model.clone(), ResponseInput::Text(text.to_string()));
        req.modalities = Some(vec!["audio".to_string()]);
        req.audio = Some(AudioOutputConfig {
            voice: options.voice.id.clone(),
            format: format_str.to_string(),
        });

        let resp = self
            .client
            .create(&req)
            .await
            .map_err(|e| AudioError::Api(format!("OpenAI Responses TTS: {e}")))?;

        let (b64_data, _transcript) = extract_audio_block(&resp.output).ok_or_else(|| {
            AudioError::Api("OpenAI Responses TTS: no audio output in response".to_string())
        })?;

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64_data)
            .map_err(|e| AudioError::Format(format!("base64 decode: {e}")))?;

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
            _ => {
                // For mp3/flac/opus, return raw bytes with a best-guess config.
                // The caller should decode using an appropriate codec.
                let config = AudioConfig {
                    sample_rate: 24000,
                    channels: 1,
                    sample_format: SampleFormat::I16,
                };
                Ok(AudioBuffer {
                    data: bytes,
                    config,
                })
            }
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
            let tts = OpenAiResponsesTts { client, model };
            yield tts.synthesize(&text, &options).await;
        };

        Box::pin(stream)
    }
}
