//! Media processing pipeline for handling image, audio, and video attachments.
//!
//! Provides [`MediaProcessor`] which can download, validate, and produce text
//! descriptions for attachments received from channel messages.
//!
//! When compiled with the `voice` feature and configured with an STT provider,
//! audio attachments are transcribed to text instead of returning a placeholder.

use std::path::{Path, PathBuf};
#[cfg(feature = "voice")]
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use tokio::io::AsyncWriteExt;

use brainwires_network::channels::message::Attachment;

#[cfg(feature = "voice")]
use brainwires_hardware::{AudioBuffer, AudioConfig, SpeechToText, SttOptions};

/// Processes media attachments from channel messages.
///
/// Downloads files to a temporary directory, validates sizes, and produces
/// text descriptions. When compiled with the `voice` feature and given an
/// STT provider, audio attachments are transcribed in real-time.
pub struct MediaProcessor {
    /// Maximum allowed file size in bytes.
    max_size_bytes: u64,
    /// Directory for temporary downloaded files.
    temp_dir: PathBuf,
    /// HTTP client for downloading attachments.
    http_client: Client,
    /// Optional speech-to-text provider for audio transcription.
    #[cfg(feature = "voice")]
    stt: Option<Arc<dyn SpeechToText>>,
}

impl MediaProcessor {
    /// Create a new `MediaProcessor` with the given maximum attachment size in megabytes.
    pub fn new(max_size_mb: u64) -> Self {
        let temp_dir = std::env::temp_dir().join("brainwires-media");
        // Best-effort creation; download_attachment will also attempt it.
        let _ = std::fs::create_dir_all(&temp_dir);

        Self {
            max_size_bytes: max_size_mb * 1024 * 1024,
            temp_dir,
            http_client: Client::new(),
            #[cfg(feature = "voice")]
            stt: None,
        }
    }

    /// Attach a speech-to-text provider for real audio transcription.
    ///
    /// Without this, audio attachments return a `[Audio: ...]` placeholder.
    #[cfg(feature = "voice")]
    pub fn with_stt(mut self, provider: Arc<dyn SpeechToText>) -> Self {
        self.stt = Some(provider);
        self
    }

    /// Download an attachment from a URL to a temporary file.
    ///
    /// Returns the path to the downloaded file.
    pub async fn download_attachment(&self, url: &str) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.temp_dir)
            .context("Failed to create temp directory for media")?;

        let response = self
            .http_client
            .get(url)
            .send()
            .await
            .context("Failed to download attachment")?;

        if !response.status().is_success() {
            bail!(
                "Attachment download failed with status {}",
                response.status()
            );
        }

        // Derive a filename from the URL or fall back to a UUID.
        let filename = url
            .rsplit('/')
            .next()
            .filter(|s| !s.is_empty())
            .unwrap_or("attachment");
        let dest = self
            .temp_dir
            .join(format!("{}_{}", uuid::Uuid::new_v4(), filename));

        let bytes = response
            .bytes()
            .await
            .context("Failed to read attachment bytes")?;

        let mut file = tokio::fs::File::create(&dest)
            .await
            .context("Failed to create temp file")?;
        file.write_all(&bytes)
            .await
            .context("Failed to write attachment to disk")?;
        file.flush().await?;

        Ok(dest)
    }

    /// Produce a text description for an image file.
    ///
    /// Currently returns a placeholder string. A future implementation will
    /// send the image to a vision-capable provider for a real description.
    pub async fn describe_image(&self, path: &Path) -> Result<String> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let metadata = std::fs::metadata(path).context("Failed to read image metadata")?;
        Ok(format!("[Image: {}, {} bytes]", filename, metadata.len()))
    }

    /// Produce a transcription for an audio file.
    ///
    /// When compiled with the `voice` feature and an STT provider is configured,
    /// reads the audio file and calls the provider for a real transcript.
    /// Falls back to a `[Audio: ...]` placeholder when STT is unavailable.
    pub async fn transcribe_audio(&self, path: &Path) -> Result<String> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        #[cfg(feature = "voice")]
        if let Some(ref stt) = self.stt {
            let data = tokio::fs::read(path)
                .await
                .context("Failed to read audio file for transcription")?;

            let buffer = AudioBuffer {
                data,
                config: AudioConfig::speech(),
            };

            let options = SttOptions::default();

            match stt.transcribe(&buffer, &options).await {
                Ok(transcript) => {
                    tracing::debug!(
                        filename = %filename,
                        text_len = transcript.text.len(),
                        "Audio transcribed successfully"
                    );
                    return Ok(transcript.text);
                }
                Err(e) => {
                    tracing::warn!(
                        filename = %filename,
                        error = %e,
                        "STT transcription failed; falling back to placeholder"
                    );
                }
            }
        }

        let metadata = std::fs::metadata(path).context("Failed to read audio metadata")?;
        Ok(format!("[Audio: {}, {} bytes]", filename, metadata.len()))
    }

    /// Validate that a file does not exceed the configured size limit.
    pub fn validate_size(&self, path: &Path) -> Result<()> {
        let metadata = std::fs::metadata(path).context("Failed to read file metadata")?;
        if metadata.len() > self.max_size_bytes {
            bail!(
                "File size {} bytes exceeds limit of {} bytes",
                metadata.len(),
                self.max_size_bytes
            );
        }
        Ok(())
    }

    /// Remove all files in the temporary media directory.
    pub fn cleanup(&self) {
        if self.temp_dir.exists()
            && let Ok(entries) = std::fs::read_dir(&self.temp_dir)
        {
            for entry in entries.flatten() {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    /// Process all attachments in a message, returning text descriptions.
    ///
    /// Each attachment is downloaded, validated, and described based on its
    /// content type. Errors for individual attachments are logged but do not
    /// prevent processing of remaining attachments.
    pub async fn process_attachments(&self, attachments: &[Attachment]) -> Vec<String> {
        let mut descriptions = Vec::new();

        for attachment in attachments {
            match self.process_single_attachment(attachment).await {
                Ok(desc) => descriptions.push(desc),
                Err(e) => {
                    tracing::warn!(
                        filename = %attachment.filename,
                        error = %e,
                        "Failed to process attachment"
                    );
                    descriptions.push(format!(
                        "[Attachment: {} — processing failed]",
                        attachment.filename
                    ));
                }
            }
        }

        descriptions
    }

    /// Process a single attachment: download, validate, and describe.
    async fn process_single_attachment(&self, attachment: &Attachment) -> Result<String> {
        let path = self.download_attachment(&attachment.url).await?;

        // Validate size.
        if let Err(e) = self.validate_size(&path) {
            let _ = std::fs::remove_file(&path);
            return Err(e);
        }

        let result = if attachment.content_type.starts_with("image/") {
            self.describe_image(&path).await
        } else if attachment.content_type.starts_with("audio/") {
            self.transcribe_audio(&path).await
        } else {
            let filename = &attachment.filename;
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            Ok(format!("[File: {}, {} bytes]", filename, size))
        };

        // Clean up the temp file.
        let _ = std::fs::remove_file(&path);

        result
    }
}

impl Drop for MediaProcessor {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn validate_size_passes_for_small_file() {
        let processor = MediaProcessor::new(1); // 1 MB limit
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.txt");
        std::fs::write(&path, "hello").unwrap();
        assert!(processor.validate_size(&path).is_ok());
    }

    #[test]
    fn validate_size_fails_for_large_file() {
        let processor = MediaProcessor::new(1); // 1 MB limit = 1_048_576 bytes
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.bin");

        // Write a file larger than 1 MB.
        let mut f = std::fs::File::create(&path).unwrap();
        let data = vec![0u8; 1_048_577];
        f.write_all(&data).unwrap();

        let result = processor.validate_size(&path);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("exceeds limit"));
    }

    #[tokio::test]
    async fn process_attachments_handles_empty_list() {
        let processor = MediaProcessor::new(10);
        let descriptions = processor.process_attachments(&[]).await;
        assert!(descriptions.is_empty());
    }

    #[tokio::test]
    async fn describe_image_returns_placeholder() {
        let processor = MediaProcessor::new(10);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.png");
        std::fs::write(&path, "fake image data").unwrap();

        let desc = processor.describe_image(&path).await.unwrap();
        assert!(desc.contains("photo.png"));
        assert!(desc.contains("bytes"));
        assert!(desc.starts_with("[Image:"));
    }

    #[tokio::test]
    async fn transcribe_audio_returns_placeholder() {
        let processor = MediaProcessor::new(10);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("recording.mp3");
        std::fs::write(&path, "fake audio data").unwrap();

        let desc = processor.transcribe_audio(&path).await.unwrap();
        assert!(desc.contains("recording.mp3"));
        assert!(desc.contains("bytes"));
        assert!(desc.starts_with("[Audio:"));
    }

    #[test]
    fn new_creates_with_correct_limit() {
        let processor = MediaProcessor::new(5);
        assert_eq!(processor.max_size_bytes, 5 * 1024 * 1024);
    }

    #[test]
    fn cleanup_does_not_panic_on_missing_dir() {
        let processor = MediaProcessor {
            max_size_bytes: 1024,
            temp_dir: PathBuf::from("/tmp/brainwires-media-nonexistent-test-dir"),
            http_client: Client::new(),
            #[cfg(feature = "voice")]
            stt: None,
        };
        processor.cleanup(); // Should not panic.
    }
}
