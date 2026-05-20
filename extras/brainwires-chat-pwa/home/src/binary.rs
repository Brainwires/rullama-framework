//! Binary-blob chunking over the data channel (Phase 2 M11).
//!
//! `bin/begin` → `bin/chunk` × N → `bin/end` lets the PWA push payloads
//! larger than the SCTP frame budget without blocking the main JSON-RPC
//! request stream. Each chunk is a single JSON-RPC frame whose `data`
//! field carries 256 KB raw → ~341 KB base64.
//!
//! Per-session lifecycle:
//!   - `bin/begin` allocates an in-memory buffer keyed by `bin_id` (a
//!     UUID minted by the PWA).
//!   - `bin/chunk` appends in-order; out-of-order or unknown-id calls
//!     surface custom JSON-RPC error codes.
//!   - `bin/end` verifies the optional `sha256` then moves the buffer to
//!     a per-session "finalized blobs" map. The blob is consumed by a
//!     subsequent `message/send` whose A2A `Part` references the
//!     `bin_id` (see [`crate::webrtc`] for the resolve-and-rewrite step).
//!
//! Buffers GC after 30 s; finalized blobs after 5 min. The sweep runs on
//! the same 60 s tick the M2 session-GC uses (kept lazy: caller invokes
//! [`BinaryStore::gc_expired`] from the existing tick loop).
//!
//! Errors return JSON-RPC error codes per the M11 plan:
//!   - [`ERR_UNKNOWN_BIN_ID`]   `-32001`
//!   - [`ERR_SEQ_OUT_OF_ORDER`] `-32002`
//!   - [`ERR_SHA256_MISMATCH`]  `-32003`
//!
//! These collide numerically with the spec-level `TASK_NOT_FOUND` /
//! `TASK_NOT_CANCELABLE` / `PUSH_NOT_SUPPORTED` codes from
//! `brainwires_a2a::error`, but the bin/* methods are transport-level
//! (never reach the A2A bridge), so the wire is unambiguous: the PWA
//! looks at the response to *its* `bin/*` request, never confuses it
//! with a task error.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

/// Custom JSON-RPC error code for `bin/chunk` / `bin/end` against an
/// unknown `bin_id`.
pub const ERR_UNKNOWN_BIN_ID: i32 = -32001;

/// Custom JSON-RPC error code for `bin/chunk` with a `seq` that doesn't
/// match the next expected sequence number.
pub const ERR_SEQ_OUT_OF_ORDER: i32 = -32002;

/// Custom JSON-RPC error code for `bin/end` whose `sha256` doesn't match
/// the assembled bytes.
pub const ERR_SHA256_MISMATCH: i32 = -32003;

/// TTL for an in-progress `bin/begin` buffer that hasn't been finalized.
pub const PENDING_TTL: Duration = Duration::from_secs(30);

/// TTL for a finalized blob waiting to be consumed by a `message/send`.
pub const FINALIZED_TTL: Duration = Duration::from_secs(5 * 60);

/// Fail any single chunk larger than this many decoded bytes — defense
/// against a malicious or buggy PWA shipping a single 100 MB frame and
/// blowing the daemon's heap.
pub const MAX_CHUNK_BYTES: usize = 1 << 20; // 1 MB

/// Hard cap on a single uploaded blob's total size. 64 MB is large enough
/// for any realistic image / audio / short video the PWA might ship and
/// far below the daemon's working-set budget.
pub const MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;

// ── wire types ────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct BinBeginParams {
    pub bin_id: String,
    #[serde(default)]
    pub content_type: Option<String>,
    pub total_size: u64,
    pub total_chunks: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BinChunkParams {
    pub bin_id: String,
    pub seq: u32,
    /// Base64-encoded chunk payload.
    pub data: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BinEndParams {
    pub bin_id: String,
    /// Optional lowercase hex-encoded SHA-256 of the assembled bytes.
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BinOk {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

// ── store ─────────────────────────────────────────────────────

/// In-progress buffer for one `bin_id`.
struct BinaryBuffer {
    content_type: Option<String>,
    total_size: usize,
    total_chunks: u32,
    next_seq: u32,
    buf: BytesMut,
    started_at: Instant,
}

/// A finalized blob ready to be consumed by `message/send`.
#[derive(Debug)]
pub struct FinalizedBlob {
    pub bin_id: String,
    pub content_type: Option<String>,
    pub bytes: Bytes,
    pub finalized_at: Instant,
}

/// Per-session map of pending + finalized binary buffers.
///
/// Cheap to clone (the inner state lives behind `Arc`). One store lives
/// alongside the per-session `outbox` on `SessionState`.
#[derive(Default)]
pub struct BinaryStore {
    pending: Mutex<HashMap<String, BinaryBuffer>>,
    finalized: Mutex<HashMap<String, Arc<FinalizedBlob>>>,
}

/// Error variants surfaced by the store. The webrtc dispatch layer maps
/// these to JSON-RPC error responses with the codes above.
#[derive(Debug, thiserror::Error)]
pub enum BinaryError {
    #[error("unknown bin_id: {0}")]
    UnknownBinId(String),
    #[error("sequence out of order: expected {expected}, got {got}")]
    SeqOutOfOrder { expected: u32, got: u32 },
    #[error("sha256 mismatch")]
    Sha256Mismatch,
    #[error("invalid base64: {0}")]
    InvalidBase64(String),
    #[error("chunk too large: {0} bytes")]
    ChunkTooLarge(usize),
    #[error("total size exceeds the daemon's per-blob cap")]
    TotalTooLarge,
    #[error("buffer would overflow declared total_size {declared}, got {observed}")]
    SizeOverflow { declared: usize, observed: usize },
    #[error("invalid params: {0}")]
    InvalidParams(String),
}

impl BinaryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a new buffer. Replaces any same-id pending buffer (the PWA
    /// may have aborted a partial upload and restarted with the same id).
    pub async fn handle_begin(&self, p: BinBeginParams) -> Result<BinOk, BinaryError> {
        if p.bin_id.is_empty() {
            return Err(BinaryError::InvalidParams("bin_id is empty".to_string()));
        }
        if p.total_size as usize > MAX_TOTAL_BYTES {
            return Err(BinaryError::TotalTooLarge);
        }
        let buf = BinaryBuffer {
            content_type: p.content_type,
            total_size: p.total_size as usize,
            total_chunks: p.total_chunks,
            next_seq: 0,
            // Pre-size the buffer; saves reallocation as chunks land.
            buf: BytesMut::with_capacity(p.total_size as usize),
            started_at: Instant::now(),
        };
        self.pending.lock().await.insert(p.bin_id, buf);
        Ok(BinOk {
            ok: true,
            size: None,
        })
    }

    pub async fn handle_chunk(&self, p: BinChunkParams) -> Result<BinOk, BinaryError> {
        let mut pending = self.pending.lock().await;
        let buffer = pending
            .get_mut(&p.bin_id)
            .ok_or_else(|| BinaryError::UnknownBinId(p.bin_id.clone()))?;
        if p.seq != buffer.next_seq {
            return Err(BinaryError::SeqOutOfOrder {
                expected: buffer.next_seq,
                got: p.seq,
            });
        }
        let decoded = BASE64
            .decode(p.data.as_bytes())
            .map_err(|e| BinaryError::InvalidBase64(e.to_string()))?;
        if decoded.len() > MAX_CHUNK_BYTES {
            return Err(BinaryError::ChunkTooLarge(decoded.len()));
        }
        let new_total = buffer.buf.len().saturating_add(decoded.len());
        if new_total > buffer.total_size {
            return Err(BinaryError::SizeOverflow {
                declared: buffer.total_size,
                observed: new_total,
            });
        }
        buffer.buf.extend_from_slice(&decoded);
        buffer.next_seq = buffer.next_seq.saturating_add(1);
        Ok(BinOk {
            ok: true,
            size: None,
        })
    }

    pub async fn handle_end(&self, p: BinEndParams) -> Result<Arc<FinalizedBlob>, BinaryError> {
        // Remove from pending atomically. Even on sha256 mismatch we drop
        // the buffer (the PWA must restart with a fresh bin_id).
        let buffer = {
            let mut pending = self.pending.lock().await;
            pending
                .remove(&p.bin_id)
                .ok_or_else(|| BinaryError::UnknownBinId(p.bin_id.clone()))?
        };
        let bytes: Bytes = buffer.buf.freeze();
        if let Some(expected_hex) = p.sha256.as_deref() {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let observed = hex::encode(hasher.finalize());
            if !observed.eq_ignore_ascii_case(expected_hex) {
                return Err(BinaryError::Sha256Mismatch);
            }
        }
        // We tolerate `buffer.total_chunks` not matching the observed
        // chunk count exactly; the byte count (bounded by `total_size`)
        // and the optional sha256 are the load-bearing checks.
        let _ = buffer.total_chunks;
        let blob = Arc::new(FinalizedBlob {
            bin_id: p.bin_id.clone(),
            content_type: buffer.content_type,
            bytes,
            finalized_at: Instant::now(),
        });
        self.finalized.lock().await.insert(p.bin_id, blob.clone());
        Ok(blob)
    }

    /// One-shot consume a finalized blob. Returns `None` if the id isn't
    /// finalized (or has already been consumed).
    pub async fn take(&self, bin_id: &str) -> Option<Arc<FinalizedBlob>> {
        self.finalized.lock().await.remove(bin_id)
    }

    /// Drop pending buffers older than [`PENDING_TTL`] and finalized
    /// blobs older than [`FINALIZED_TTL`]. Caller invokes from the same
    /// 60 s GC tick the session map uses.
    pub async fn gc_expired(&self) {
        let now = Instant::now();
        {
            let mut pending = self.pending.lock().await;
            pending.retain(|_, b| now.saturating_duration_since(b.started_at) < PENDING_TTL);
        }
        {
            let mut finalized = self.finalized.lock().await;
            finalized.retain(|_, b| now.saturating_duration_since(b.finalized_at) < FINALIZED_TTL);
        }
    }

    /// Test helper — count of pending buffers.
    #[cfg(test)]
    pub async fn pending_len(&self) -> usize {
        self.pending.lock().await.len()
    }

    /// Test helper — count of finalized blobs.
    #[cfg(test)]
    pub async fn finalized_len(&self) -> usize {
        self.finalized.lock().await.len()
    }

    /// Test helper — replace started_at on all pending buffers (used by
    /// the GC test to simulate the wall clock advancing).
    #[cfg(test)]
    pub async fn _force_pending_age(&self, age: Duration) {
        let now = Instant::now();
        let mut pending = self.pending.lock().await;
        for b in pending.values_mut() {
            b.started_at = now.checked_sub(age).unwrap_or(now);
        }
    }
}

// ── tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn b64(bytes: &[u8]) -> String {
        BASE64.encode(bytes)
    }

    #[tokio::test]
    async fn bin_begin_chunk_end_happy_path() {
        let store = BinaryStore::new();
        let original: Vec<u8> = (0u8..=255u8).cycle().take(700).collect();
        let chunks = original.chunks(256).collect::<Vec<_>>();
        store
            .handle_begin(BinBeginParams {
                bin_id: "id-1".to_string(),
                content_type: Some("application/octet-stream".to_string()),
                total_size: original.len() as u64,
                total_chunks: chunks.len() as u32,
            })
            .await
            .expect("begin");
        for (i, c) in chunks.iter().enumerate() {
            store
                .handle_chunk(BinChunkParams {
                    bin_id: "id-1".to_string(),
                    seq: i as u32,
                    data: b64(c),
                })
                .await
                .expect("chunk");
        }
        let mut hasher = Sha256::new();
        hasher.update(&original);
        let sha = hex::encode(hasher.finalize());
        let blob = store
            .handle_end(BinEndParams {
                bin_id: "id-1".to_string(),
                sha256: Some(sha),
            })
            .await
            .expect("end");
        assert_eq!(blob.bytes.as_ref(), original.as_slice());
        assert_eq!(
            blob.content_type.as_deref(),
            Some("application/octet-stream")
        );
    }

    #[tokio::test]
    async fn bin_chunk_unknown_id_returns_error() {
        let store = BinaryStore::new();
        let err = store
            .handle_chunk(BinChunkParams {
                bin_id: "ghost".to_string(),
                seq: 0,
                data: b64(b"hello"),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, BinaryError::UnknownBinId(_)));
    }

    #[tokio::test]
    async fn bin_chunk_out_of_order_returns_error() {
        let store = BinaryStore::new();
        store
            .handle_begin(BinBeginParams {
                bin_id: "ooo".to_string(),
                content_type: None,
                total_size: 32,
                total_chunks: 2,
            })
            .await
            .unwrap();
        let err = store
            .handle_chunk(BinChunkParams {
                bin_id: "ooo".to_string(),
                seq: 1,
                data: b64(&[0u8; 16]),
            })
            .await
            .unwrap_err();
        match err {
            BinaryError::SeqOutOfOrder { expected, got } => {
                assert_eq!(expected, 0);
                assert_eq!(got, 1);
            }
            other => panic!("expected SeqOutOfOrder, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn bin_end_sha256_mismatch_returns_error_and_drops_buffer() {
        let store = BinaryStore::new();
        store
            .handle_begin(BinBeginParams {
                bin_id: "x".to_string(),
                content_type: None,
                total_size: 4,
                total_chunks: 1,
            })
            .await
            .unwrap();
        store
            .handle_chunk(BinChunkParams {
                bin_id: "x".to_string(),
                seq: 0,
                data: b64(b"abcd"),
            })
            .await
            .unwrap();
        let err = store
            .handle_end(BinEndParams {
                bin_id: "x".to_string(),
                sha256: Some("00".repeat(32)),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, BinaryError::Sha256Mismatch));
        // Buffer should be gone — neither pending nor finalized.
        assert_eq!(store.pending_len().await, 0);
        assert_eq!(store.finalized_len().await, 0);
    }

    #[tokio::test]
    async fn bin_pending_gc_drops_after_30s() {
        let store = BinaryStore::new();
        store
            .handle_begin(BinBeginParams {
                bin_id: "y".to_string(),
                content_type: None,
                total_size: 4,
                total_chunks: 1,
            })
            .await
            .unwrap();
        // Force its started_at into the past beyond PENDING_TTL.
        store
            ._force_pending_age(PENDING_TTL + Duration::from_secs(1))
            .await;
        store.gc_expired().await;
        assert_eq!(store.pending_len().await, 0);
    }

    #[tokio::test]
    async fn bin_finalized_take_is_one_shot() {
        let store = BinaryStore::new();
        store
            .handle_begin(BinBeginParams {
                bin_id: "z".to_string(),
                content_type: None,
                total_size: 3,
                total_chunks: 1,
            })
            .await
            .unwrap();
        store
            .handle_chunk(BinChunkParams {
                bin_id: "z".to_string(),
                seq: 0,
                data: b64(b"abc"),
            })
            .await
            .unwrap();
        store
            .handle_end(BinEndParams {
                bin_id: "z".to_string(),
                sha256: None,
            })
            .await
            .unwrap();
        let first = store.take("z").await;
        let second = store.take("z").await;
        assert!(first.is_some());
        assert!(second.is_none(), "take must be one-shot");
    }

    #[tokio::test]
    async fn bin_chunk_size_overflow_rejected() {
        let store = BinaryStore::new();
        store
            .handle_begin(BinBeginParams {
                bin_id: "ov".to_string(),
                content_type: None,
                total_size: 4,
                total_chunks: 1,
            })
            .await
            .unwrap();
        let err = store
            .handle_chunk(BinChunkParams {
                bin_id: "ov".to_string(),
                seq: 0,
                data: b64(&[0u8; 16]),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, BinaryError::SizeOverflow { .. }));
    }

    #[tokio::test]
    async fn bin_total_size_cap_enforced() {
        let store = BinaryStore::new();
        let err = store
            .handle_begin(BinBeginParams {
                bin_id: "huge".to_string(),
                content_type: None,
                total_size: (MAX_TOTAL_BYTES as u64) + 1,
                total_chunks: 1,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, BinaryError::TotalTooLarge));
    }
}
