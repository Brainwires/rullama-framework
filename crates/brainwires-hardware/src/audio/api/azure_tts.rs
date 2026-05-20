use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::azure_speech::AzureSpeechClient;

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::tts::TextToSpeech;
use crate::audio::types::{AudioBuffer, OutputFormat, TtsOptions, Voice};
use crate::audio::wav::decode_wav;

/// Azure Cognitive Services text-to-speech implementation.
///
/// Wraps an [`AzureSpeechClient`] from `brainwires-provider`.
pub struct AzureTts {
    client: Arc<AzureSpeechClient>,
}

impl AzureTts {
    /// Create a new Azure TTS client.
    pub fn new(subscription_key: impl Into<String>, region: impl Into<String>) -> Self {
        let client = Arc::new(AzureSpeechClient::new(subscription_key, region));
        Self { client }
    }

    /// Create from an existing [`AzureSpeechClient`].
    pub fn from_client(client: Arc<AzureSpeechClient>) -> Self {
        Self { client }
    }
}

/// Map our [`OutputFormat`] to an Azure output format string.
fn format_to_azure(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Wav => "riff-24khz-16bit-mono-pcm",
        OutputFormat::Pcm => "raw-24khz-16bit-mono-pcm",
        OutputFormat::Mp3 => "audio-24khz-160kbitrate-mono-mp3",
        OutputFormat::Opus => "ogg-24khz-16bit-mono-opus",
        OutputFormat::Flac => "riff-24khz-16bit-mono-pcm", // Azure doesn't support FLAC; fall back to WAV
    }
}

#[async_trait]
impl TextToSpeech for AzureTts {
    fn name(&self) -> &str {
        "azure-tts"
    }

    async fn list_voices(&self) -> AudioResult<Vec<Voice>> {
        let voices = self
            .client
            .list_voices()
            .await
            .map_err(|e| AudioError::Api(format!("Azure TTS list_voices: {e}")))?;

        let voices = voices
            .into_iter()
            .map(|v| Voice {
                id: v.short_name.clone(),
                name: Some(v.display_name),
                language: Some(v.locale),
            })
            .collect();

        Ok(voices)
    }

    async fn synthesize(&self, text: &str, options: &TtsOptions) -> AudioResult<AudioBuffer> {
        let output_format = format_to_azure(options.output_format);

        let bytes = self
            .client
            .synthesize_text(text, &options.voice.id, output_format)
            .await
            .map_err(|e| AudioError::Api(format!("Azure TTS synthesize: {e}")))?;

        match options.output_format {
            OutputFormat::Wav | OutputFormat::Flac => decode_wav(&bytes),
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
        let text = text.to_string();
        let options = options.clone();

        let stream = async_stream::stream! {
            let tts = AzureTts { client };
            yield tts.synthesize(&text, &options).await;
        };

        Box::pin(stream)
    }
}
