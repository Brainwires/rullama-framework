//! Attachment handling for remote file uploads
//!
//! Manages chunked file uploads from the web UI, including:
//! - Reassembly of chunks
//! - Decompression (zstd/gzip)
//! - Checksum verification
//! - Temporary file storage

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use super::protocol::CompressionAlgorithm;

/// Default chunk size for attachments (64KB)
pub const ATTACHMENT_CHUNK_SIZE: usize = 64 * 1024;

/// Compression threshold - files larger than this are compressed (10KB)
pub const COMPRESSION_THRESHOLD: usize = 10 * 1024;

/// Maximum attachment size (100MB)
pub const MAX_ATTACHMENT_SIZE: u64 = 100 * 1024 * 1024;

/// State of an in-progress attachment upload
#[derive(Debug)]
#[allow(dead_code)]
struct PendingAttachment {
    /// Unique attachment ID
    id: String,
    /// Original filename
    filename: String,
    /// MIME type
    mime_type: String,
    /// Expected total size (uncompressed)
    expected_size: u64,
    /// Whether data is compressed
    compressed: bool,
    /// Compression algorithm (if compressed)
    compression_algorithm: Option<CompressionAlgorithm>,
    /// Expected number of chunks
    chunks_total: u32,
    /// Received chunks (index -> data)
    chunks: HashMap<u32, Vec<u8>>,
    /// Total bytes received so far
    bytes_received: usize,
    /// Associated agent ID
    agent_id: String,
    /// Command ID for response
    command_id: String,
}

/// Manages attachment uploads
#[derive(Clone)]
pub struct AttachmentReceiver {
    /// Pending attachments by ID
    pending: Arc<RwLock<HashMap<String, PendingAttachment>>>,
    /// Directory to store received attachments
    output_dir: PathBuf,
}

impl AttachmentReceiver {
    /// Create a new attachment receiver
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            pending: Arc::new(RwLock::new(HashMap::new())),
            output_dir,
        }
    }

    /// Start receiving a new attachment
    #[allow(clippy::too_many_arguments)]
    pub async fn start_upload(
        &self,
        command_id: String,
        agent_id: String,
        attachment_id: String,
        filename: String,
        mime_type: String,
        size: u64,
        compressed: bool,
        compression_algorithm: Option<CompressionAlgorithm>,
        chunks_total: u32,
    ) -> Result<()> {
        // Validate size
        if size > MAX_ATTACHMENT_SIZE {
            bail!(
                "Attachment too large: {} bytes (max: {} bytes)",
                size,
                MAX_ATTACHMENT_SIZE
            );
        }

        let pending = PendingAttachment {
            id: attachment_id.clone(),
            filename,
            mime_type,
            expected_size: size,
            compressed,
            compression_algorithm,
            chunks_total,
            chunks: HashMap::new(),
            bytes_received: 0,
            agent_id,
            command_id,
        };

        let mut pending_map = self.pending.write().await;
        pending_map.insert(attachment_id, pending);

        Ok(())
    }

    /// Receive a chunk of attachment data
    pub async fn receive_chunk(
        &self,
        attachment_id: &str,
        chunk_index: u32,
        data: &str,
        is_final: bool,
    ) -> Result<bool> {
        // Decode base64 data
        let decoded = BASE64
            .decode(data)
            .context("Failed to decode base64 chunk data")?;

        let mut pending_map = self.pending.write().await;
        let pending = pending_map
            .get_mut(attachment_id)
            .context("Unknown attachment ID")?;

        // Validate chunk index
        if chunk_index >= pending.chunks_total {
            bail!(
                "Invalid chunk index: {} (expected 0-{})",
                chunk_index,
                pending.chunks_total - 1
            );
        }

        // Store chunk
        pending.bytes_received += decoded.len();
        pending.chunks.insert(chunk_index, decoded);

        // Check if we have all chunks
        let all_received = pending.chunks.len() == pending.chunks_total as usize;

        if is_final && !all_received {
            tracing::warn!(
                "Final chunk received but only have {}/{} chunks",
                pending.chunks.len(),
                pending.chunks_total
            );
        }

        Ok(all_received)
    }

    /// Complete the attachment upload, verify checksum, and save to disk
    pub async fn complete_upload(
        &self,
        attachment_id: &str,
        expected_checksum: &str,
    ) -> Result<PathBuf> {
        let pending = {
            let mut pending_map = self.pending.write().await;
            pending_map
                .remove(attachment_id)
                .context("Unknown attachment ID")?
        };

        // Reassemble chunks in order
        let mut assembled = Vec::with_capacity(pending.bytes_received);
        for i in 0..pending.chunks_total {
            let chunk = pending
                .chunks
                .get(&i)
                .context(format!("Missing chunk {}", i))?;
            assembled.extend_from_slice(chunk);
        }

        // Decompress if needed
        let data = if pending.compressed {
            decompress(&assembled, pending.compression_algorithm)?
        } else {
            assembled
        };

        // Verify checksum
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let actual_checksum = format!("{:x}", hasher.finalize());

        if actual_checksum != expected_checksum {
            bail!(
                "Checksum mismatch: expected {}, got {}",
                expected_checksum,
                actual_checksum
            );
        }

        // Ensure output directory exists
        std::fs::create_dir_all(&self.output_dir)
            .context("Failed to create attachment output directory")?;

        // Generate unique filename
        let safe_filename = sanitize_filename(&pending.filename);
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let output_path = self
            .output_dir
            .join(format!("{}_{}", timestamp, safe_filename));

        // Write to file
        let mut file =
            std::fs::File::create(&output_path).context("Failed to create attachment file")?;
        file.write_all(&data)
            .context("Failed to write attachment data")?;

        tracing::info!(
            "Attachment saved: {} ({} bytes, {})",
            output_path.display(),
            data.len(),
            pending.mime_type
        );

        Ok(output_path)
    }

    /// Cancel a pending upload
    pub async fn cancel_upload(&self, attachment_id: &str) {
        let mut pending_map = self.pending.write().await;
        if pending_map.remove(attachment_id).is_some() {
            tracing::info!("Cancelled attachment upload: {}", attachment_id);
        }
    }

    /// Get status of a pending upload
    pub async fn get_status(&self, attachment_id: &str) -> Option<(u32, u32, usize)> {
        let pending_map = self.pending.read().await;
        pending_map
            .get(attachment_id)
            .map(|p| (p.chunks.len() as u32, p.chunks_total, p.bytes_received))
    }
}

/// Decompress data using the specified algorithm
fn decompress(data: &[u8], algorithm: Option<CompressionAlgorithm>) -> Result<Vec<u8>> {
    match algorithm {
        Some(CompressionAlgorithm::Zstd) => {
            zstd::decode_all(data).context("Failed to decompress zstd data")
        }
        Some(CompressionAlgorithm::Gzip) => {
            let mut decoder = flate2::read::GzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .context("Failed to decompress gzip data")?;
            Ok(decompressed)
        }
        None => Ok(data.to_vec()),
    }
}

/// Sanitize a filename to prevent path traversal attacks
fn sanitize_filename(filename: &str) -> String {
    // Take only the file name (not path)
    let name = std::path::Path::new(filename)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("attachment");

    // Replace problematic characters
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_ascii_control() => '_',
            c => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("file.txt"), "file.txt");
        assert_eq!(sanitize_filename("path/to/file.txt"), "file.txt");
        assert_eq!(sanitize_filename("file:name.txt"), "file_name.txt");
        assert_eq!(sanitize_filename("file<>|name.txt"), "file___name.txt");
    }

    #[tokio::test]
    async fn test_attachment_receiver() {
        let temp_dir = tempfile::tempdir().unwrap();
        let receiver = AttachmentReceiver::new(temp_dir.path().to_path_buf());

        // Start upload
        receiver
            .start_upload(
                "cmd-1".to_string(),
                "agent-1".to_string(),
                "attach-1".to_string(),
                "test.txt".to_string(),
                "text/plain".to_string(),
                13,
                false,
                None,
                1,
            )
            .await
            .unwrap();

        // Send chunk (base64 of "Hello, World!")
        let data = BASE64.encode(b"Hello, World!");
        let all_received = receiver
            .receive_chunk("attach-1", 0, &data, true)
            .await
            .unwrap();
        assert!(all_received);

        // Calculate expected checksum
        let mut hasher = Sha256::new();
        hasher.update(b"Hello, World!");
        let checksum = format!("{:x}", hasher.finalize());

        // Complete upload
        let path = receiver
            .complete_upload("attach-1", &checksum)
            .await
            .unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "Hello, World!");
    }
}
