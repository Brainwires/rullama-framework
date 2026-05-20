use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use futures::stream::BoxStream;

use brainwires_provider::openai_responses::{
    CreateResponseRequest, InputContent, InputContentPart, OutputContentBlock, ResponseInput,
    ResponseInputItem, ResponseOutputItem, ResponsesClient,
};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::stt::SpeechToText;
use crate::audio::types::{AudioBuffer, SttOptions, Transcript};
use crate::audio::wav::encode_wav;

/// OpenAI Responses API speech-to-text implementation.
///
/// Uses the Responses API with audio input instead of the dedicated
/// `/v1/audio/transcriptions` endpoint. Sends audio as an `input_audio`
/// content part and asks the model to transcribe it.
pub struct OpenAiResponsesStt {
    client: Arc<ResponsesClient>,
    model: String,
}

impl OpenAiResponsesStt {
    /// Create a new Responses API STT client.
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

/// Extract concatenated text from the response output.
fn extract_text(output: &[ResponseOutputItem]) -> Option<String> {
    for item in output {
        if let ResponseOutputItem::Message { content, .. } = item {
            let mut texts = Vec::new();
            for block in content {
                if let OutputContentBlock::OutputText { text, .. } = block {
                    texts.push(text.as_str());
                }
            }
            if !texts.is_empty() {
                return Some(texts.join(" "));
            }
        }
    }
    None
}

#[async_trait]
impl SpeechToText for OpenAiResponsesStt {
    fn name(&self) -> &str {
        "openai-responses-stt"
    }

    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: &SttOptions,
    ) -> AudioResult<Transcript> {
        let wav_data = encode_wav(audio)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav_data);

        let mut parts = vec![InputContentPart::InputAudio {
            data: b64,
            format: "wav".to_string(),
        }];

        let prompt = if let Some(ref lang) = options.language {
            format!(
                "Transcribe this audio. The language is {lang}. Return only the transcription text, nothing else."
            )
        } else {
            "Transcribe this audio. Return only the transcription text, nothing else.".to_string()
        };

        parts.push(InputContentPart::InputText { text: prompt });

        let input = ResponseInput::Items(vec![ResponseInputItem::Message {
            role: "user".to_string(),
            content: InputContent::Parts(parts),
            status: None,
        }]);

        let req = CreateResponseRequest::new(self.model.clone(), input);

        let resp = self
            .client
            .create(&req)
            .await
            .map_err(|e| AudioError::Api(format!("OpenAI Responses STT: {e}")))?;

        let text = extract_text(&resp.output).unwrap_or_default();

        Ok(Transcript {
            text,
            language: options.language.clone(),
            duration_secs: None,
            segments: vec![],
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
                let stt = OpenAiResponsesStt { client, model };
                yield stt.transcribe(&full_buffer, &options).await;
            }
        };

        Box::pin(stream)
    }
}
