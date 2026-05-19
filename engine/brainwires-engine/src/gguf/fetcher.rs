//! Tensor-byte source abstraction.
//!
//! The GGUF parser used to own the raw `Vec<u8>` for the entire model file. That kept
//! ~7 GB in CPU memory for the lifetime of the `Model` — fine on native, fatal on
//! wasm32 (4 GB linear-memory cap). M6 splits storage from access: the parser keeps
//! only the small (5–10 MB) header, and individual tensor reads go through this trait.
//!
//! Two implementations:
//!   * [`InMemoryFetcher`] — wraps a `Vec<u8>` (or `Arc<[u8]>`). Used by native callers
//!     and by tests. `fetch` is a memcpy of the requested slice.
//!   * `HttpRangeFetcher` (M6.4) — `fetch()` issues an HTTP `Range: bytes=N-M` request.
//!     Only useful in browsers; not built on native.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{Result, RullamaError};

/// Read tensor bytes on demand by absolute offset.
///
/// The trait is `?Send` because `wasm32-unknown-unknown` is single-threaded and
/// browser-side futures are not `Send`. Native impls happen to be `Send + Sync` but
/// we don't require it on the trait.
#[async_trait(?Send)]
pub trait TensorFetcher {
    /// Total length of the underlying source in bytes (used by the parser to bounds-check).
    fn total_len(&self) -> u64;

    /// Read `len` bytes starting at `offset`. Caller owns the returned `Vec<u8>` and
    /// is expected to drop it as soon as the data has been copied to its final home
    /// (e.g. a wgpu buffer).
    async fn fetch(&self, offset: u64, len: u64) -> Result<Vec<u8>>;
}

// ---------- InMemoryFetcher ----------

/// Wraps a `Vec<u8>` (or any `Arc<[u8]>`) so existing in-memory callers fit the trait
/// without refactoring. The fetch is a `Vec::from(&slice)` — synchronous in body,
/// async in signature so it shares the trait surface with [`HttpRangeFetcher`].
pub struct InMemoryFetcher {
    bytes: Arc<[u8]>,
}

impl InMemoryFetcher {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }

    pub fn from_arc(bytes: Arc<[u8]>) -> Self {
        Self { bytes }
    }

    /// Borrow the full source bytes (zero-copy). Only useful when the caller can keep
    /// the fetcher alive longer than the borrow.
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

#[async_trait(?Send)]
impl TensorFetcher for InMemoryFetcher {
    fn total_len(&self) -> u64 {
        self.bytes.len() as u64
    }

    async fn fetch(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let start = offset as usize;
        let end = start.checked_add(len as usize).ok_or_else(|| {
            RullamaError::Gguf(format!("InMemoryFetcher: range overflow {offset}+{len}"))
        })?;
        if end > self.bytes.len() {
            return Err(RullamaError::Gguf(format!(
                "InMemoryFetcher: range {start}..{end} extends past buffer end ({})",
                self.bytes.len()
            )));
        }
        Ok(self.bytes[start..end].to_vec())
    }
}

// ---------- HttpRangeFetcher (wasm32-only) ----------

/// Browser-side fetcher that pulls byte ranges from an HTTP URL via `fetch()` with a
/// `Range: bytes=N-M` header. Native callers don't need this — they wrap the bytes
/// they already have in [`InMemoryFetcher`] — so the impl lives behind `cfg(wasm32)`.
///
/// Construction is async because we need the total file length (read from the
/// Content-Length of an initial HEAD-equivalent Range request) before any tensor
/// reads can be bounds-checked.
#[cfg(target_arch = "wasm32")]
pub struct HttpRangeFetcher {
    url: String,
    total: u64,
}

#[cfg(target_arch = "wasm32")]
impl HttpRangeFetcher {
    /// Build a fetcher for `url`. Issues one initial Range request to discover the
    /// total size (read from `Content-Range` or `X-Total-Size`).
    pub async fn new(url: String) -> Result<Self> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        let req_init = web_sys::RequestInit::new();
        req_init.set_method("GET");
        let headers = web_sys::Headers::new()
            .map_err(|e| RullamaError::Gguf(format!("Headers::new: {e:?}")))?;
        headers
            .set("Range", "bytes=0-0")
            .map_err(|e| RullamaError::Gguf(format!("set Range: {e:?}")))?;
        req_init.set_headers(&headers);

        let request = web_sys::Request::new_with_str_and_init(&url, &req_init)
            .map_err(|e| RullamaError::Gguf(format!("Request::new: {e:?}")))?;

        let resp_value = JsFuture::from(global_fetch(&request)?)
            .await
            .map_err(|e| RullamaError::Gguf(format!("fetch failed: {e:?}")))?;
        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|e| RullamaError::Gguf(format!("response cast: {e:?}")))?;
        if !resp.ok() && resp.status() != 206 {
            return Err(RullamaError::Gguf(format!(
                "HTTP {} from {url}",
                resp.status()
            )));
        }

        // Prefer Content-Range "bytes 0-0/<total>"; fall back to X-Total-Size.
        let total = if let Some(cr) = resp.headers().get("Content-Range").ok().flatten() {
            cr.rsplit('/')
                .next()
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(|| RullamaError::Gguf(format!("bad Content-Range: {cr}")))?
        } else if let Some(xs) = resp.headers().get("X-Total-Size").ok().flatten() {
            xs.parse::<u64>()
                .map_err(|e| RullamaError::Gguf(format!("bad X-Total-Size: {e}")))?
        } else {
            return Err(RullamaError::Gguf(
                "server returned no Content-Range or X-Total-Size; cannot determine GGUF length"
                    .into(),
            ));
        };

        Ok(Self { url, total })
    }
}

#[cfg(target_arch = "wasm32")]
fn global_fetch(request: &web_sys::Request) -> Result<js_sys::Promise> {
    use wasm_bindgen::JsCast;
    // Works in both Window and DedicatedWorkerGlobalScope contexts.
    let global = js_sys::global();
    if let Some(window) = global.dyn_ref::<web_sys::Window>() {
        return Ok(window.fetch_with_request(request));
    }
    if let Some(scope) = global.dyn_ref::<web_sys::WorkerGlobalScope>() {
        return Ok(scope.fetch_with_request(request));
    }
    Err(RullamaError::Gguf(
        "no Window or WorkerGlobalScope for fetch()".into(),
    ))
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
impl TensorFetcher for HttpRangeFetcher {
    fn total_len(&self) -> u64 {
        self.total
    }

    async fn fetch(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        if len == 0 {
            return Ok(Vec::new());
        }
        let end = offset.checked_add(len - 1).ok_or_else(|| {
            RullamaError::Gguf(format!("HttpRangeFetcher: range overflow {offset}+{len}"))
        })?;
        if end >= self.total {
            return Err(RullamaError::Gguf(format!(
                "HttpRangeFetcher: range {offset}..={end} extends past file end ({})",
                self.total
            )));
        }

        let req_init = web_sys::RequestInit::new();
        req_init.set_method("GET");
        let headers = web_sys::Headers::new()
            .map_err(|e| RullamaError::Gguf(format!("Headers::new: {e:?}")))?;
        headers
            .set("Range", &format!("bytes={offset}-{end}"))
            .map_err(|e| RullamaError::Gguf(format!("set Range: {e:?}")))?;
        req_init.set_headers(&headers);

        let request = web_sys::Request::new_with_str_and_init(&self.url, &req_init)
            .map_err(|e| RullamaError::Gguf(format!("Request::new: {e:?}")))?;

        let resp_value = JsFuture::from(global_fetch(&request)?)
            .await
            .map_err(|e| RullamaError::Gguf(format!("fetch failed: {e:?}")))?;
        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|e| RullamaError::Gguf(format!("response cast: {e:?}")))?;
        if !resp.ok() && resp.status() != 206 {
            return Err(RullamaError::Gguf(format!(
                "HTTP {} fetching range {offset}-{end}",
                resp.status()
            )));
        }

        let buf_promise = resp
            .array_buffer()
            .map_err(|e| RullamaError::Gguf(format!("array_buffer: {e:?}")))?;
        let array_buffer = JsFuture::from(buf_promise)
            .await
            .map_err(|e| RullamaError::Gguf(format!("await array_buffer: {e:?}")))?;
        let bytes = js_sys::Uint8Array::new(&array_buffer).to_vec();
        if bytes.len() as u64 != len {
            return Err(RullamaError::Gguf(format!(
                "HttpRangeFetcher: server returned {} bytes, expected {len}",
                bytes.len()
            )));
        }
        Ok(bytes)
    }
}

// ---------- OpfsFetcher (wasm32-only) ----------

/// Browser-side fetcher backed by a JS callback that resolves ranges from an
/// **OPFS** (Origin Private File System) file.
///
/// Why this exists: iOS Safari silently caps a combined Blob at ~5.6 GiB and
/// kills the WebContent process around ~2 GiB of *live* JS memory — both apply
/// during the IndexedDB-Blob path used by `HttpRangeFetcher` callers that
/// cache. OPFS sidesteps both: bytes are written through a
/// `FileSystemSyncAccessHandle` in a Worker (no JS-heap residency) and reads
/// go through `file.slice(offset, end).arrayBuffer()` (disk-backed, also no
/// JS-heap residency for the source).
///
/// The wasm side never touches OPFS directly — it calls into the JS
/// `read_fn(offset_f64, len_f64) -> Promise<Uint8Array>` callback. JS owns the
/// `FileSystemFileHandle` lifetime; this struct just holds the callback and
/// the total file size (passed in at construction time so the GGUF parser can
/// bounds-check without a round-trip).
#[cfg(target_arch = "wasm32")]
pub struct OpfsFetcher {
    read_fn: js_sys::Function,
    total: u64,
}

#[cfg(target_arch = "wasm32")]
impl OpfsFetcher {
    pub fn new(read_fn: js_sys::Function, total: u64) -> Self {
        Self { read_fn, total }
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
impl TensorFetcher for OpfsFetcher {
    fn total_len(&self) -> u64 {
        self.total
    }

    async fn fetch(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        use wasm_bindgen::{JsCast, JsValue};
        use wasm_bindgen_futures::JsFuture;

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

        let result = self
            .read_fn
            .call2(
                &JsValue::NULL,
                &JsValue::from_f64(offset as f64),
                &JsValue::from_f64(len as f64),
            )
            .map_err(|e| RullamaError::Gguf(format!("OPFS read_fn call failed: {e:?}")))?;

        // The JS side may return a Uint8Array directly (sync) or a Promise.
        // Probe for thenable and await it if present.
        let value = if let Ok(promise) = result.clone().dyn_into::<js_sys::Promise>() {
            JsFuture::from(promise)
                .await
                .map_err(|e| RullamaError::Gguf(format!("OPFS read_fn promise rejected: {e:?}")))?
        } else {
            result
        };

        let array = js_sys::Uint8Array::new(&value);
        let bytes = array.to_vec();
        if bytes.len() as u64 != len {
            return Err(RullamaError::Gguf(format!(
                "OpfsFetcher: read_fn returned {} bytes, expected {len}",
                bytes.len()
            )));
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block<F: core::future::Future>(f: F) -> F::Output {
        // Native test path; cfg-guarded out of wasm32 by the surrounding `#[cfg(test)]`
        // attribute on the parent module, which `cargo test` only runs on the host.
        pollster::block_on(f)
    }

    #[test]
    fn in_memory_fetcher_returns_correct_slice() {
        let bytes: Vec<u8> = (0..=255u8).collect();
        let f = InMemoryFetcher::new(bytes);
        assert_eq!(f.total_len(), 256);
        let chunk = block(f.fetch(10, 8)).unwrap();
        assert_eq!(chunk, vec![10, 11, 12, 13, 14, 15, 16, 17]);
    }

    #[test]
    fn in_memory_fetcher_rejects_out_of_range() {
        let f = InMemoryFetcher::new(vec![0u8; 16]);
        assert!(block(f.fetch(0, 17)).is_err());
        assert!(block(f.fetch(20, 1)).is_err());
    }

    #[test]
    fn in_memory_fetcher_zero_length() {
        let f = InMemoryFetcher::new(vec![1, 2, 3, 4]);
        let chunk = block(f.fetch(2, 0)).unwrap();
        assert_eq!(chunk, Vec::<u8>::new());
    }
}
