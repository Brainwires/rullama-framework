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
pub use native::{find_manifest, ollama_models_root, FileBlobSource};
