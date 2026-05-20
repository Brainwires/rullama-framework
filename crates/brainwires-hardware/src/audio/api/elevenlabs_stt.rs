use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::elevenlabs::{ElevenLabsClient, ElevenLabsSttRequest};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::stt::SpeechToText;
use crate::audio::types::{AudioBuffer, SttOptions, Transcript};
use crate::audio::wav::encode_wav;

/// ElevenLabs speech-to-text implementation.
///
/// Wraps an [`ElevenLabsClient`] from `brainwires-provider` for the actual HTTP
/// transport; this struct adds the `SpeechToText` trait and audio-domain logic.
pub struct ElevenLabsStt {
    client: Arc<ElevenLabsClient>,
}

impl ElevenLabsStt {
    /// Create a new ElevenLabs STT client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(ElevenLabsClient::new(api_key));
        Self { client }
    }

    /// Create from an existing [`ElevenLabsClient`].
    pub fn from_client(client: Arc<ElevenLabsClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl SpeechToText for ElevenLabsStt {
    fn name(&self) -> &str {
        "elevenlabs-stt"
    }

    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: &SttOptions,
    ) -> AudioResult<Transcript> {
        let wav_data = encode_wav(audio)?;

        let req = ElevenLabsSttRequest {
            model: None,
            language_code: options.language.clone(),
        };

        let resp = self
            .client
            .speech_to_text(wav_data, &req)
            .await
            .map_err(|e| AudioError::Api(format!("ElevenLabs STT: {e}")))?;

        Ok(Transcript {
            text: resp.text,
            language: resp.language_code,
            duration_secs: None,
            segments: Vec::new(),
        })
    }

    fn transcribe_stream(
        &self,
        audio_stream: BoxStream<'static, AudioResult<AudioBuffer>>,
        options: &SttOptions,
    ) -> BoxStream<'static, AudioResult<Transcript>> {
        let client = Arc::clone(&self.client);
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
                let stt = ElevenLabsStt { client };
                yield stt.transcribe(&full_buffer, &options).await;
            }
        };

        Box::pin(stream)
    }
}
