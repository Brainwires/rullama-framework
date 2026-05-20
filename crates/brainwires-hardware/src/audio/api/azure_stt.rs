use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::azure_speech::{AzureSpeechClient, AzureSttRequest};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::stt::SpeechToText;
use crate::audio::types::{AudioBuffer, SttOptions, Transcript};
use crate::audio::wav::encode_wav;

/// Azure Cognitive Services speech-to-text implementation.
///
/// Wraps an [`AzureSpeechClient`] from `brainwires-provider`.
pub struct AzureStt {
    client: Arc<AzureSpeechClient>,
}

impl AzureStt {
    /// Create a new Azure STT client.
    pub fn new(subscription_key: impl Into<String>, region: impl Into<String>) -> Self {
        let client = Arc::new(AzureSpeechClient::new(subscription_key, region));
        Self { client }
    }

    /// Create from an existing [`AzureSpeechClient`].
    pub fn from_client(client: Arc<AzureSpeechClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl SpeechToText for AzureStt {
    fn name(&self) -> &str {
        "azure-stt"
    }

    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: &SttOptions,
    ) -> AudioResult<Transcript> {
        let wav_data = encode_wav(audio)?;

        let req = AzureSttRequest {
            language: options.language.clone(),
            content_type: Some(format!(
                "audio/wav; codecs=audio/pcm; samplerate={}",
                audio.config.sample_rate
            )),
        };

        let resp = self
            .client
            .recognize(wav_data, &req)
            .await
            .map_err(|e| AudioError::Api(format!("Azure STT recognize: {e}")))?;

        if resp.recognition_status != "Success" {
            return Err(AudioError::Transcription(format!(
                "Azure STT status: {}",
                resp.recognition_status
            )));
        }

        Ok(Transcript {
            text: resp.display_text.unwrap_or_default(),
            language: options.language.clone(),
            duration_secs: resp.duration.map(|d| d as f64 / 10_000_000.0), // Azure durations are in 100ns ticks
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
                let stt = AzureStt { client };
                yield stt.transcribe(&full_buffer, &options).await;
            }
        };

        Box::pin(stream)
    }
}
