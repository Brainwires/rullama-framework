use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::fish::{FishAsrRequest, FishClient};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::stt::SpeechToText;
use crate::audio::types::{AudioBuffer, SttOptions, Transcript};
use crate::audio::wav::encode_wav;

/// Fish Audio speech-to-text (ASR) implementation.
///
/// Wraps a [`FishClient`] from `brainwires-provider`.
pub struct FishStt {
    client: Arc<FishClient>,
}

impl FishStt {
    /// Create a new Fish STT client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(FishClient::new(api_key));
        Self { client }
    }

    /// Create from an existing [`FishClient`].
    pub fn from_client(client: Arc<FishClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl SpeechToText for FishStt {
    fn name(&self) -> &str {
        "fish-stt"
    }

    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: &SttOptions,
    ) -> AudioResult<Transcript> {
        let wav_data = encode_wav(audio)?;

        let req = FishAsrRequest {
            language: options.language.clone(),
        };

        let resp = self
            .client
            .asr(wav_data, &req)
            .await
            .map_err(|e| AudioError::Api(format!("Fish ASR: {e}")))?;

        Ok(Transcript {
            text: resp.text,
            language: options.language.clone(),
            duration_secs: resp.duration,
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
                let stt = FishStt { client };
                yield stt.transcribe(&full_buffer, &options).await;
            }
        };

        Box::pin(stream)
    }
}
