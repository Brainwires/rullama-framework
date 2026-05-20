use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use brainwires_provider_speech::murf::{MurfClient, MurfGenerateRequest};

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::tts::TextToSpeech;
use crate::audio::types::{AudioBuffer, OutputFormat, TtsOptions, Voice};
use crate::audio::wav::decode_wav;

/// Murf AI text-to-speech implementation.
///
/// Wraps a [`MurfClient`] from `brainwires-provider`.
pub struct MurfTts {
    client: Arc<MurfClient>,
}

impl MurfTts {
    /// Create a new Murf TTS client.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = Arc::new(MurfClient::new(api_key));
        Self { client }
    }

    /// Create from an existing [`MurfClient`].
    pub fn from_client(client: Arc<MurfClient>) -> Self {
        Self { client }
    }
}

/// Map our [`OutputFormat`] to a Murf format string.
fn format_to_murf(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Wav => "WAV",
        OutputFormat::Mp3 => "MP3",
        OutputFormat::Flac => "FLAC",
        // Murf only supports WAV/MP3/FLAC; default to WAV for others.
        OutputFormat::Pcm | OutputFormat::Opus => "WAV",
    }
}

#[async_trait]
impl TextToSpeech for MurfTts {
    fn name(&self) -> &str {
        "murf-tts"
    }

    async fn list_voices(&self) -> AudioResult<Vec<Voice>> {
        let resp = self
            .client
            .list_voices()
            .await
            .map_err(|e| AudioError::Api(format!("Murf list_voices: {e}")))?;

        let voices = resp
            .voices
            .into_iter()
            .map(|v| Voice {
                id: v.voice_id,
                name: Some(v.name),
                language: v.language_code,
            })
            .collect();

        Ok(voices)
    }

    async fn synthesize(&self, text: &str, options: &TtsOptions) -> AudioResult<AudioBuffer> {
        let req = MurfGenerateRequest {
            voice_id: options.voice.id.clone(),
            text: text.to_string(),
            format: Some(format_to_murf(options.output_format).to_string()),
            rate: options.speed,
            pitch: None,
            sample_rate: Some(24000),
        };

        let resp = self
            .client
            .generate_speech(&req)
            .await
            .map_err(|e| AudioError::Api(format!("Murf generate_speech: {e}")))?;

        let audio_url = resp
            .audio_file
            .ok_or_else(|| AudioError::Api("Murf returned no audio URL".to_string()))?;

        let bytes = self
            .client
            .download_audio(&audio_url)
            .await
            .map_err(|e| AudioError::Api(format!("Murf download_audio: {e}")))?;

        match options.output_format {
            OutputFormat::Wav | OutputFormat::Pcm | OutputFormat::Opus => decode_wav(&bytes),
            _ => Err(AudioError::Unsupported(format!(
                "direct decoding of {:?} not supported; use Wav format",
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
            let tts = MurfTts { client };
            yield tts.synthesize(&text, &options).await;
        };

        Box::pin(stream)
    }
}
