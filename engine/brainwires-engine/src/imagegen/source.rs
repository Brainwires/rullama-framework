//! Blob source: fetch a content-addressed image-model blob by its on-disk
//! filename (`sha256-<hex>`, from [`crate::imagegen::manifest::BlobRef::blob_filename`]).
//!
//! Image models are many small blobs rather than one big GGUF, so the source
//! is "read a whole blob by digest" rather than the GGUF [`TensorFetcher`]'s
//! "read a byte range of one file". A native [`FileBlobSource`] reads from an
//! Ollama blobs directory; OPFS / HTTP-`Range` sources for wasm are a follow-up
//! (they slot in behind the same trait).
//!
//! [`TensorFetcher`]: crate::gguf::TensorFetcher

use async_trait::async_trait;

use crate::error::Result;

/// Read a complete content-addressed blob by its `sha256-<hex>` filename.
///
/// `?Send` to match the wasm single-threaded story (mirrors `TensorFetcher`).
#[async_trait(?Send)]
pub trait BlobSource {
    /// Read the entire blob named `blob_filename` (e.g. `sha256-ab12…`).
    async fn read_blob(&self, blob_filename: &str) -> Result<Vec<u8>>;

    /// Read at most `max` leading bytes of a blob — enough to parse the
    /// safetensors header without materializing a large weight tensor.
    /// Default impl reads the whole blob and truncates; native sources
    /// override with a positioned read.
    async fn read_prefix(&self, blob_filename: &str, max: usize) -> Result<Vec<u8>> {
        let mut b = self.read_blob(blob_filename).await?;
        b.truncate(max);
        Ok(b)
    }

    /// Read `len` bytes starting at `offset` — the per-tensor streaming read
    /// (a tensor's byte span within a multi-GB shard). Default reads the whole
    /// blob and slices; ranged sources (file `read_at`, HTTP `Range`) override.
    async fn read_range(&self, blob_filename: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        let b = self.read_blob(blob_filename).await?;
        let (s, e) = (offset as usize, (offset + len) as usize);
        b.get(s..e).map(|sl| sl.to_vec()).ok_or_else(|| {
            crate::error::RullamaError::Image(format!("range {s}..{e} past blob {blob_filename}"))
        })
    }
}

// ---------- FileBlobSource (native-only) ----------

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use crate::error::RullamaError;
    use std::path::{Path, PathBuf};

    /// Reads blobs straight from an Ollama blobs directory
    /// (`~/.ollama/models/blobs/`), one file per digest.
    pub struct FileBlobSource {
        blobs_dir: PathBuf,
    }

    impl FileBlobSource {
        pub fn new(blobs_dir: impl Into<PathBuf>) -> Self {
            Self {
                blobs_dir: blobs_dir.into(),
            }
        }

        /// Default Ollama location: `$OLLAMA_MODELS/blobs` or
        /// `~/.ollama/models/blobs`.
        pub fn ollama_default() -> Result<Self> {
            Ok(Self::new(ollama_models_root()?.join("blobs")))
        }

        fn path(&self, name: &str) -> PathBuf {
            self.blobs_dir.join(name)
        }
    }

    #[async_trait(?Send)]
    impl BlobSource for FileBlobSource {
        async fn read_blob(&self, blob_filename: &str) -> Result<Vec<u8>> {
            let p = self.path(blob_filename);
            std::fs::read(&p)
                .map_err(|e| RullamaError::Image(format!("read blob {}: {e}", p.display())))
        }

        async fn read_prefix(&self, blob_filename: &str, max: usize) -> Result<Vec<u8>> {
            use std::io::Read;
            let p = self.path(blob_filename);
            let f = std::fs::File::open(&p)
                .map_err(|e| RullamaError::Image(format!("open blob {}: {e}", p.display())))?;
            let mut buf = Vec::with_capacity(max.min(1 << 20));
            f.take(max as u64)
                .read_to_end(&mut buf)
                .map_err(|e| RullamaError::Image(format!("read blob {}: {e}", p.display())))?;
            Ok(buf)
        }

        async fn read_range(&self, blob_filename: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
            use std::os::unix::fs::FileExt;
            let p = self.path(blob_filename);
            let f = std::fs::File::open(&p)
                .map_err(|e| RullamaError::Image(format!("open blob {}: {e}", p.display())))?;
            let mut buf = vec![0u8; len as usize];
            f.read_exact_at(&mut buf, offset)
                .map_err(|e| RullamaError::Image(format!("read range {}: {e}", p.display())))?;
            Ok(buf)
        }
    }

    /// Ollama models root: `$OLLAMA_MODELS`, else `$HOME/.ollama/models`.
    pub fn ollama_models_root() -> Result<PathBuf> {
        if let Ok(p) = std::env::var("OLLAMA_MODELS") {
            return Ok(PathBuf::from(p));
        }
        let home = std::env::var("HOME")
            .map_err(|_| RullamaError::Image("HOME not set; pass OLLAMA_MODELS".into()))?;
        Ok(PathBuf::from(home).join(".ollama/models"))
    }

    /// Locate a model's manifest file under `models/manifests/**`, matching a
    /// `name[:tag]` (tag defaults to `latest`). Searches the whole manifests
    /// tree so it works regardless of registry namespace
    /// (`registry.ollama.ai/library/<name>/<tag>`, a bare local create, …).
    pub fn find_manifest(name_and_tag: &str) -> Result<PathBuf> {
        let (name, tag) = match name_and_tag.split_once(':') {
            Some((n, t)) => (n, t),
            None => (name_and_tag, "latest"),
        };
        let root = ollama_models_root()?.join("manifests");
        let mut found: Option<PathBuf> = None;
        walk(&root, &mut |p| {
            // match a path ending in .../<name>/<tag>
            let is_tag = p.file_name().map(|f| f == tag).unwrap_or(false);
            let is_name = p
                .parent()
                .and_then(|par| par.file_name())
                .map(|f| f == name)
                .unwrap_or(false);
            if is_tag && is_name && found.is_none() {
                found = Some(p.to_path_buf());
            }
        });
        found.ok_or_else(|| {
            RullamaError::Image(format!(
                "no manifest for {name:?}:{tag:?} under {}",
                root.display()
            ))
        })
    }

    fn walk(dir: &Path, f: &mut impl FnMut(&Path)) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, f);
            } else {
                f(&p);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{FileBlobSource, find_manifest, ollama_models_root};

// ---------- HttpRangeBlobSource (wasm32-only) ----------

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use crate::error::RullamaError;

    /// Browser blob source that fetches component files (`transformer/…safetensors`)
    /// under a base URL, using HTTP `Range` for per-tensor reads — the image
    /// analogue of `gguf::HttpRangeFetcher`. The R2/CDN origin serves `Range`
    /// (206 + Content-Range), so a multi-GB shard is never fetched whole.
    pub struct HttpRangeBlobSource {
        base_url: String,
    }

    impl HttpRangeBlobSource {
        /// `base_url` is the model root, e.g.
        /// `https://models.brainwires.dev/z-image-turbo` (no trailing slash).
        pub fn new(base_url: impl Into<String>) -> Self {
            Self {
                base_url: base_url.into(),
            }
        }

        fn url(&self, name: &str) -> String {
            format!("{}/{}", self.base_url.trim_end_matches('/'), name)
        }

        async fn fetch(&self, name: &str, range: Option<(u64, u64)>) -> Result<Vec<u8>> {
            use wasm_bindgen::JsCast;
            use wasm_bindgen_futures::JsFuture;
            let init = web_sys::RequestInit::new();
            init.set_method("GET");
            if let Some((off, len)) = range {
                if len == 0 {
                    return Ok(Vec::new());
                }
                let headers = web_sys::Headers::new()
                    .map_err(|e| RullamaError::Image(format!("Headers: {e:?}")))?;
                headers
                    .set("Range", &format!("bytes={off}-{}", off + len - 1))
                    .map_err(|e| RullamaError::Image(format!("set Range: {e:?}")))?;
                init.set_headers(&headers);
            }
            let url = self.url(name);
            let request = web_sys::Request::new_with_str_and_init(&url, &init)
                .map_err(|e| RullamaError::Image(format!("Request: {e:?}")))?;
            let resp_val = JsFuture::from(super::wasm_global_fetch(&request)?)
                .await
                .map_err(|e| RullamaError::Image(format!("fetch {url}: {e:?}")))?;
            let resp: web_sys::Response = resp_val
                .dyn_into()
                .map_err(|e| RullamaError::Image(format!("response cast: {e:?}")))?;
            if !resp.ok() && resp.status() != 206 {
                return Err(RullamaError::Image(format!(
                    "HTTP {} for {url}",
                    resp.status()
                )));
            }
            let ab = JsFuture::from(
                resp.array_buffer()
                    .map_err(|e| RullamaError::Image(format!("array_buffer: {e:?}")))?,
            )
            .await
            .map_err(|e| RullamaError::Image(format!("await body: {e:?}")))?;
            Ok(js_sys::Uint8Array::new(&ab).to_vec())
        }
    }

    #[async_trait(?Send)]
    impl BlobSource for HttpRangeBlobSource {
        async fn read_blob(&self, name: &str) -> Result<Vec<u8>> {
            self.fetch(name, None).await
        }
        async fn read_prefix(&self, name: &str, max: usize) -> Result<Vec<u8>> {
            self.fetch(name, Some((0, max as u64))).await
        }
        async fn read_range(&self, name: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
            self.fetch(name, Some((offset, len))).await
        }
    }

    /// Browser blob source backed by **OPFS-cached** component files — the image
    /// analogue of `gguf::OpfsFetcher`. The model is downloaded once to OPFS;
    /// this reads per-tensor ranges from the local files (no network during
    /// generation), keeping wasm memory bounded by the existing per-call paging.
    ///
    /// JS owns the `FileSystemSyncAccessHandle`s (one per file in the component
    /// dir) and exposes a single callback `read_fn(name, offset, len) ->
    /// Uint8Array | Promise<Uint8Array>` that dispatches by file name:
    /// - `len < 0` ⇒ read from `offset` to EOF (whole file / remainder),
    /// - `len == 0` ⇒ empty,
    /// - `len > 0` ⇒ exactly `len` bytes at `offset` (clamped to EOF).
    pub struct OpfsBlobSource {
        read_fn: js_sys::Function,
    }

    impl OpfsBlobSource {
        pub fn new(read_fn: js_sys::Function) -> Self {
            Self { read_fn }
        }

        async fn call(&self, name: &str, offset: f64, len: f64) -> Result<Vec<u8>> {
            use wasm_bindgen::{JsCast, JsValue};
            use wasm_bindgen_futures::JsFuture;
            let result = self
                .read_fn
                .call3(
                    &JsValue::NULL,
                    &JsValue::from_str(name),
                    &JsValue::from_f64(offset),
                    &JsValue::from_f64(len),
                )
                .map_err(|e| RullamaError::Image(format!("OPFS read_fn {name}: {e:?}")))?;
            // sync Uint8Array or a Promise — probe + await if thenable.
            let value = if let Ok(promise) = result.clone().dyn_into::<js_sys::Promise>() {
                JsFuture::from(promise).await.map_err(|e| {
                    RullamaError::Image(format!("OPFS read_fn {name} rejected: {e:?}"))
                })?
            } else {
                result
            };
            Ok(js_sys::Uint8Array::new(&value).to_vec())
        }
    }

    #[async_trait(?Send)]
    impl BlobSource for OpfsBlobSource {
        async fn read_blob(&self, name: &str) -> Result<Vec<u8>> {
            self.call(name, 0.0, -1.0).await
        }
        async fn read_prefix(&self, name: &str, max: usize) -> Result<Vec<u8>> {
            self.call(name, 0.0, max as f64).await
        }
        async fn read_range(&self, name: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
            if len == 0 {
                return Ok(Vec::new());
            }
            self.call(name, offset as f64, len as f64).await
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::{HttpRangeBlobSource, OpfsBlobSource};

/// `fetch()` via Window or WorkerGlobalScope (mirrors gguf::fetcher::global_fetch).
#[cfg(target_arch = "wasm32")]
fn wasm_global_fetch(request: &web_sys::Request) -> Result<js_sys::Promise> {
    use wasm_bindgen::JsCast;
    let global = js_sys::global();
    if let Some(window) = global.dyn_ref::<web_sys::Window>() {
        return Ok(window.fetch_with_request(request));
    }
    if let Some(scope) = global.dyn_ref::<web_sys::WorkerGlobalScope>() {
        return Ok(scope.fetch_with_request(request));
    }
    Err(crate::error::RullamaError::Image(
        "no Window or WorkerGlobalScope for fetch()".into(),
    ))
}
