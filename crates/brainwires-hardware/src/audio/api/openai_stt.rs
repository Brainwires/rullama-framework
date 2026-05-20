use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider::openai_chat::{OpenAiClient, TranscriptionRequest};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::stt::SpeechToText;
use crate::audio::types::{AudioBuffer, SttOptions, Transcript, TranscriptSegment};
use crate::audio::wav::encode_wav;

/// OpenAI Whisper API speech-to-text implementation.
///
/// Wraps an [`OpenAiClient`] from `brainwires-provider`.
pub struct OpenAiStt {
    client: Arc<OpenAiClient>,
    model: String,
}

impl OpenAiStt {
    /// Create a new OpenAI STT client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(OpenAiClient::new(api_key.into(), "whisper-1".to_string()));
        Self {
            client,
            model: "whisper-1".to_string(),
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

    /// Set the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl SpeechToText for OpenAiStt {
    fn name(&self) -> &str {
        "openai-whisper"
    }

    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: &SttOptions,
    ) -> AudioResult<Transcript> {
        let wav_data = encode_wav(audio)?;

        let req = TranscriptionRequest {
            model: self.model.clone(),
            language: options.language.clone(),
            prompt: options.prompt.clone(),
            timestamps: Some(options.timestamps),
        };

        let resp = self
            .client
            .create_transcription(wav_data, &req)
            .await
            .map_err(|e| AudioError::Api(format!("OpenAI STT: {e}")))?;

        let segments = resp
            .segments
            .unwrap_or_default()
            .into_iter()
            .filter_map(|seg| {
                Some(TranscriptSegment {
                    text: seg.text?,
                    start: seg.start?,
                    end: seg.end?,
                })
            })
            .collect();

        Ok(Transcript {
            text: resp.text,
            language: resp.language,
            duration_secs: resp.duration,
            segments,
        })
    }

    fn transcribe_stream(
        &self,
        audio_stream: BoxStream<'static, AudioResult<AudioBuffer>>,
        options: &SttOptions,
    ) -> BoxStream<'static, AudioResult<Transcript>> {
        let client = Arc::clone(&self.client);
        let model = self.model.clone();
        let options = options.clone();

        let stream = async_stream::stream! {
            use futures::StreamExt;

            let mut all_data = Vec::new();
            let mut config = None;
            let mut audio_stream = audio_stream;

            while let Some(result) = audio_stream.next().await {
                match result {
                    Ok(buffer) => {
                        if config.is_none() {
                            config = Some(buffer.config.clone());
                        }
                        all_data.extend_from_slice(&buffer.data);
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }

            if let Some(cfg) = config {
                let full_buffer = AudioBuffer::from_pcm(all_data, cfg);
                let stt = OpenAiStt { client, model };
                yield stt.transcribe(&full_buffer, &options).await;
            }
        };

        Box::pin(stream)
    }
}
