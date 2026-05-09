//! `TensorFetcher` adapter that pulls bytes from OPFS via a JS sync-access
//! read callback.
//!
//! Why this lives here and not in `brainwires-llama`:
//! `brainwires-llama` ships two stock fetchers — `InMemoryFetcher`
//! (full file in a `Vec<u8>`, fine on native, blows the 4 GB wasm32 cap)
//! and `HttpRangeFetcher` (HTTP `Range:` server). chat-pwa stores the GGUF
//! in OPFS and already has a JS-side `read_fn(offset, length) -> Uint8Array`
//! callback wired up for its old chunked-init path (see
//! `crate::call_read_fn` in `lib.rs`). This adapter wraps that callback as
//! the third concrete `TensorFetcher` impl so `Model::load_streaming` works
//! against OPFS without dragging the file through wasm linear memory.
//!
//! The callback is invoked synchronously inside `fetch()`. wasm32 is
//! single-threaded so there is no contention concern; the `?Send` bound
//! on `TensorFetcher` reflects exactly this.

use async_trait::async_trait;
use brainwires_llama::error::{Result as LlamaResult, RullamaError};
use brainwires_llama::gguf::TensorFetcher;
use js_sys::Function;

/// OPFS-backed fetcher. Owns the JS `read_fn(offset, length) -> Uint8Array`
/// closure plus the cached file size; `fetch()` is a thin wrapper around
/// `crate::call_read_fn` (which already handles chunked reads above 64 MiB).
pub struct OpfsFetcher {
    read_fn: Function,
    total: u64,
}

impl OpfsFetcher {
    pub fn new(read_fn: Function, total: u64) -> Self {
        Self { read_fn, total }
    }
}

#[async_trait(?Send)]
impl TensorFetcher for OpfsFetcher {
    fn total_len(&self) -> u64 {
        self.total
    }

    async fn fetch(&self, offset: u64, len: u64) -> LlamaResult<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let end = offset.checked_add(len).ok_or_else(|| {
            RullamaError::Gguf(format!("OpfsFetcher: range overflow {offset}+{len}"))
        })?;
        if end > self.total {
            return Err(RullamaError::Gguf(format!(
                "OpfsFetcher: range {offset}..{end} extends past file end ({})",
                self.total
            )));
        }

        let bytes = crate::call_read_fn(&self.read_fn, offset, len).map_err(|e| {
            RullamaError::Gguf(format!(
                "OPFS read at offset {offset} (len {len}) failed: {e:?}"
            ))
        })?;
        if bytes.len() as u64 != len {
            return Err(RullamaError::Gguf(format!(
                "OpfsFetcher: short read — got {} bytes, expected {len}",
                bytes.len()
            )));
        }
        Ok(bytes)
    }
}
