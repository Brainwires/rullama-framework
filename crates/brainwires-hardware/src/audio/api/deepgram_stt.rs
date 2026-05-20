use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::deepgram::{DeepgramClient, DeepgramListenRequest};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::stt::SpeechToText;
use crate::audio::types::{AudioBuffer, SttOptions, Transcript, TranscriptSegment};
use crate::audio::wav::encode_wav;

/// Deepgram Nova STT speech-to-text implementation.
///
/// Wraps a [`DeepgramClient`] from `brainwires-provider` for the actual HTTP
/// transport; this struct adds the `SpeechToText` trait and audio-domain logic.
pub struct DeepgramStt {
    client: Arc<DeepgramClient>,
    model: String,
}

impl DeepgramStt {
    /// Create a new Deepgram STT client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(DeepgramClient::new(api_key));
        Self {
            client,
            model: "nova-2".to_string(),
        }
    }

    /// Create from an existing [`DeepgramClient`].
    pub fn from_client(client: Arc<DeepgramClient>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

#[async_trait]
impl SpeechToText for DeepgramStt {
    fn name(&self) -> &str {
        "deepgram-stt"
    }

    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: &SttOptions,
    ) -> AudioResult<Transcript> {
        let wav_data = encode_wav(audio)?;

        let req = DeepgramListenRequest {
            model: Some(self.model.clone()),
            language: options.language.clone(),
            punctuate: true,
            diarize: false,
            content_type: Some("audio/wav".to_string()),
        };

        let resp = self
            .client
            .listen(wav_data, &req)
            .await
            .map_err(|e| AudioError::Api(format!("Deepgram STT: {e}")))?;

        // Extract the best alternative from the first channel.
        let alt = resp
            .results
            .channels
            .into_iter()
            .next()
            .and_then(|ch| ch.alternatives.into_iter().next());

        let (text, words) = match alt {
            Some(a) => (a.transcript, a.words),
            None => (String::new(), Vec::new()),
        };

        let segments = words
            .into_iter()
            .map(|w| TranscriptSegment {
                text: w.word,
                start: w.start,
                end: w.end,
            })
            .collect();

        Ok(Transcript {
            text,
            language: options.language.clone(),
            duration_secs: None,
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
                let stt = DeepgramStt { client, model };
                yield stt.transcribe(&full_buffer, &options).await;
            }
        };

        Box::pin(stream)
    }
}
