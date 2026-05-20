use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use futures::stream::BoxStream;

use brainwires_provider_speech::google_tts::{
    GoogleTtsAudioConfig, GoogleTtsClient, GoogleTtsInput, GoogleTtsSynthesizeRequest,
    GoogleTtsVoiceSelection,
};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::tts::TextToSpeech;
use crate::audio::types::{
    AudioBuffer, AudioConfig, OutputFormat, SampleFormat, TtsOptions, Voice,
};

/// Google Cloud TTS text-to-speech implementation.
///
/// Wraps a [`GoogleTtsClient`] from `brainwires-provider`.
pub struct GoogleTts {
    client: Arc<GoogleTtsClient>,
}

impl GoogleTts {
    /// Create a new Google TTS client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(GoogleTtsClient::new(api_key));
        Self { client }
    }

    /// Create from an existing [`GoogleTtsClient`].
    pub fn from_client(client: Arc<GoogleTtsClient>) -> Self {
        Self { client }
    }
}

/// Map our [`OutputFormat`] to a Google Cloud audio encoding string.
fn format_to_encoding(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Wav | OutputFormat::Pcm => "LINEAR16",
        OutputFormat::Mp3 => "MP3",
        OutputFormat::Opus => "OGG_OPUS",
        OutputFormat::Flac => "FLAC",
    }
}

#[async_trait]
impl TextToSpeech for GoogleTts {
    fn name(&self) -> &str {
        "google-tts"
    }

    async fn list_voices(&self) -> AudioResult<Vec<Voice>> {
        let resp = self
            .client
            .list_voices(None)
            .await
            .map_err(|e| AudioError::Api(format!("Google TTS list_voices: {e}")))?;

        let voices = resp
            .voices
            .into_iter()
            .map(|v| Voice {
                id: v.name.clone(),
                name: Some(v.name),
                language: v.language_codes.first().cloned(),
            })
            .collect();

        Ok(voices)
    }

    async fn synthesize(&self, text: &str, options: &TtsOptions) -> AudioResult<AudioBuffer> {
        let language = options
            .voice
            .language
            .clone()
            .unwrap_or_else(|| "en-US".to_string());

        let req = GoogleTtsSynthesizeRequest {
            input: GoogleTtsInput {
                text: Some(text.to_string()),
                ssml: None,
            },
            voice: GoogleTtsVoiceSelection {
                language_code: language,
                name: Some(options.voice.id.clone()),
                ssml_gender: None,
            },
            audio_config: GoogleTtsAudioConfig {
                audio_encoding: format_to_encoding(options.output_format).to_string(),
                speaking_rate: options.speed,
                pitch: None,
                sample_rate_hertz: Some(24000),
            },
        };

        let resp = self
            .client
            .synthesize(&req)
            .await
            .map_err(|e| AudioError::Api(format!("Google TTS synthesize: {e}")))?;

        // Google returns base64-encoded audio content.
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&resp.audio_content)
            .map_err(|e| AudioError::Format(format!("base64 decode error: {e}")))?;

        match options.output_format {
            OutputFormat::Wav => crate::audio::wav::decode_wav(&decoded),
            OutputFormat::Pcm => {
                // LINEAR16 returns raw 16-bit signed little-endian PCM.
                let config = AudioConfig {
                    sample_rate: 24000,
                    channels: 1,
                    sample_format: SampleFormat::I16,
                };
                Ok(AudioBuffer::from_pcm(decoded, config))
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
            let tts = GoogleTts { client };
            yield tts.synthesize(&text, &options).await;
        };

        Box::pin(stream)
    }
}
