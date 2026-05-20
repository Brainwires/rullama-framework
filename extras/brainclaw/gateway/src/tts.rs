//! TTS response processor — synthesises agent text responses to audio files.
//!
//! When a [`TtsProcessor`] is attached to the gateway handler, every agent
//! response is synthesised to an MP3/WAV audio file written to a configurable
//! temp directory.  A randomly-named file is created for each response.
//!
//! The caller is responsible for serving the file at `audio_base_url/<filename>`
//! so that channel adapters can attach or send the audio URL.

#[cfg(feature = "voice")]
mod inner {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use brainwires_hardware::{TextToSpeech, TtsOptions};
    use uuid::Uuid;

    /// Synthesises text responses to audio files and returns a URL.
    pub struct TtsProcessor {
        tts: Arc<dyn TextToSpeech>,
        options: TtsOptions,
        /// Directory where audio files are written.
        audio_dir: PathBuf,
        /// Base URL used to construct the audio attachment URL.
        /// e.g. `"http://localhost:18789/audio"`
        audio_base_url: String,
    }

    impl TtsProcessor {
        /// Create a new TTS processor.
        ///
        /// - `tts`: the TTS backend to use.
        /// - `options`: voice/format options (voice ID, output format, speed).
        /// - `audio_dir`: directory for generated audio files.
        /// - `audio_base_url`: public URL prefix for generated files.
        pub fn new(
            tts: Arc<dyn TextToSpeech>,
            options: TtsOptions,
            audio_dir: impl Into<PathBuf>,
            audio_base_url: impl Into<String>,
        ) -> Self {
            Self {
                tts,
                options,
                audio_dir: audio_dir.into(),
                audio_base_url: audio_base_url.into(),
            }
        }

        /// Synthesise `text` and write the result to a temporary file.
        ///
        /// Returns the public URL for the generated file, or `None` if the
        /// response text is empty or synthesis fails (warning is logged).
        pub async fn synthesize_to_url(&self, text: &str) -> Option<String> {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return None;
            }

            let buffer = match self.tts.synthesize(trimmed, &self.options).await {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(error = %e, "TTS synthesis failed");
                    return None;
                }
            };

            // Determine file extension from output format
            let ext = match self.options.output_format {
                brainwires_hardware::OutputFormat::Mp3 => "mp3",
                brainwires_hardware::OutputFormat::Opus => "opus",
                brainwires_hardware::OutputFormat::Flac => "flac",
                brainwires_hardware::OutputFormat::Wav => "wav",
                _ => "wav",
            };

            let filename = format!("{}.{}", Uuid::new_v4(), ext);
            let path = self.audio_dir.join(&filename);

            if let Err(e) = std::fs::create_dir_all(&self.audio_dir) {
                tracing::warn!(error = %e, dir = %self.audio_dir.display(), "Failed to create TTS audio dir");
                return None;
            }

            if let Err(e) = std::fs::write(&path, &buffer.data) {
                tracing::warn!(error = %e, path = %path.display(), "Failed to write TTS audio file");
                return None;
            }

            let url = format!("{}/{}", self.audio_base_url.trim_end_matches('/'), filename);
            tracing::debug!(url = %url, bytes = buffer.data.len(), "TTS audio written");
            Some(url)
        }

        /// Return the audio directory (for serving static files).
        pub fn audio_dir(&self) -> &Path {
            &self.audio_dir
        }
    }
}

#[cfg(feature = "voice")]
pub use inner::TtsProcessor;
