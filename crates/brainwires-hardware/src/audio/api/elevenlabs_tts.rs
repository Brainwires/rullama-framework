use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::elevenlabs::{ElevenLabsClient, ElevenLabsTtsRequest};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::tts::TextToSpeech;
use crate::audio::types::{
    AudioBuffer, AudioConfig, OutputFormat, SAMPLE_RATE_CD, SAMPLE_RATE_SPEECH, SampleFormat,
    TtsOptions, Voice,
};

/// ElevenLabs text-to-speech implementation.
///
/// Wraps an [`ElevenLabsClient`] from `brainwires-provider` for the actual HTTP
/// transport; this struct adds the `TextToSpeech` trait and audio-domain logic.
pub struct ElevenLabsTts {
    client: Arc<ElevenLabsClient>,
    model_id: String,
}

impl ElevenLabsTts {
    /// Create a new ElevenLabs TTS client with the default model.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(ElevenLabsClient::new(api_key));
        Self {
            client,
            model_id: "eleven_multilingual_v2".to_string(),
        }
    }

    /// Create from an existing [`ElevenLabsClient`].
    pub fn from_client(client: Arc<ElevenLabsClient>, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
        }
    }

    /// Set the model ID (e.g., "eleven_multilingual_v2", "eleven_monolingual_v1").
    pub fn with_model(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }
}

/// Map [`OutputFormat`] to an ElevenLabs `output_format` query string value.
///
/// ElevenLabs uses format strings like `"mp3_44100_128"`, `"pcm_16000"`, `"pcm_44100"`.
/// We pick reasonable defaults for each variant.
fn output_format_string(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Mp3 => "mp3_44100_128",
        OutputFormat::Pcm => "pcm_16000",
        OutputFormat::Wav => "pcm_16000", // request raw PCM, we wrap into AudioBuffer
        OutputFormat::Opus => "mp3_44100_128", // fallback to mp3
        OutputFormat::Flac => "mp3_44100_128", // fallback to mp3
    }
}

/// Return the sample rate implied by the ElevenLabs output format string.
fn sample_rate_for_format(format: OutputFormat) -> u32 {
    match format {
        OutputFormat::Pcm | OutputFormat::Wav => SAMPLE_RATE_SPEECH,
        _ => SAMPLE_RATE_CD,
    }
}

#[async_trait]
impl TextToSpeech for ElevenLabsTts {
    fn name(&self) -> &str {
        "elevenlabs-tts"
    }

    async fn list_voices(&self) -> AudioResult<Vec<Voice>> {
        let response = self
            .client
            .list_voices()
            .await
            .map_err(|e| AudioError::Api(format!("ElevenLabs list_voices: {e}")))?;

        let voices = response
            .voices
            .into_iter()
            .map(|v| Voice {
                id: v.voice_id,
                name: Some(v.name),
                language: v.labels.get("language").cloned(),
            })
            .collect();

        Ok(voices)
    }

    async fn synthesize(&self, text: &str, options: &TtsOptions) -> AudioResult<AudioBuffer> {
        let req = ElevenLabsTtsRequest {
            text: text.to_string(),
            model_id: Some(self.model_id.clone()),
            voice_settings: None,
            output_format: Some(output_format_string(options.output_format).to_string()),
        };

        let bytes = self
            .client
            .text_to_speech(&options.voice.id, &req)
            .await
            .map_err(|e| AudioError::Api(format!("ElevenLabs TTS: {e}")))?;

        // ElevenLabs returns raw PCM when we request pcm_* formats.
        // For PCM/Wav output formats we requested raw PCM, so wrap directly.
        match options.output_format {
            OutputFormat::Pcm | OutputFormat::Wav => {
                let config = AudioConfig {
                    sample_rate: sample_rate_for_format(options.output_format),
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
        let model_id = self.model_id.clone();
        let text = text.to_string();
        let options = options.clone();

        let stream = async_stream::stream! {
            let tts = ElevenLabsTts { client, model_id };
            yield tts.synthesize(&text, &options).await;
        };

        Box::pin(stream)
    }
}
