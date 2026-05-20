//! Read GGUF model weights directly from a local Ollama install
//! (`~/.ollama/models/`) without re-downloading from the registry.
//!
//! Ollama stores models as content-addressed OCI artifacts:
//!
//! ```text
//! ~/.ollama/models/
//!   manifests/
//!     registry.ollama.ai/
//!       library/
//!         <model>/
//!           <tag>            ← JSON manifest pointing at blob digests
//!   blobs/
//!     sha256-<hex>           ← actual GGUF / template / params bytes
//! ```
//!
//! Native targets only — `wasm32` browsers can't access the local FS, so
//! the chat-pwa always goes through the registry download path. The CLI /
//! agent paths skip the round-trip entirely when an Ollama install is
//! present.

#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// `application/vnd.ollama.image.model` — the GGUF blob layer.
pub const MEDIA_TYPE_GGUF: &str = "application/vnd.ollama.image.model";
/// `application/vnd.ollama.image.template` — chat-template Jinja string.
pub const MEDIA_TYPE_TEMPLATE: &str = "application/vnd.ollama.image.template";
/// `application/vnd.ollama.image.tokenizer` — tokenizer.json layer.
pub const MEDIA_TYPE_TOKENIZER: &str = "application/vnd.ollama.image.tokenizer";
/// `application/vnd.ollama.image.params` — sampling defaults JSON.
pub const MEDIA_TYPE_PARAMS: &str = "application/vnd.ollama.image.params";

#[derive(Debug, Deserialize)]
struct OciManifest {
    #[serde(default)]
    layers: Vec<OciLayer>,
}

#[derive(Debug, Deserialize)]
struct OciLayer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
    #[serde(default)]
    size: u64,
}

/// One file (blob layer) discovered in the local Ollama cache.
#[derive(Debug, Clone)]
pub struct OllamaCachedFile {
    /// OCI mediaType (e.g. `application/vnd.ollama.image.model`).
    pub media_type: String,
    /// Content digest, including `sha256:` prefix.
    pub digest: String,
    /// Layer size in bytes (from the manifest).
    pub size: u64,
    /// Absolute path to the blob on disk.
    pub blob_path: PathBuf,
}

impl OllamaCachedFile {
    /// True for the GGUF model layer.
    pub fn is_weights(&self) -> bool {
        self.media_type == MEDIA_TYPE_GGUF
    }

    /// Read the blob bytes into memory. Use sparingly — model layers can
    /// be multiple GB. Prefer mmap or streaming for large weights.
    pub fn read_bytes(&self) -> std::io::Result<Vec<u8>> {
        fs::read(&self.blob_path)
    }
}

/// Resolve `~/.ollama/models` (or `$OLLAMA_MODELS` if set).
pub fn ollama_models_dir() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("OLLAMA_MODELS") {
        let p = PathBuf::from(custom);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = dirs::home_dir()?;
    let p = home.join(".ollama").join("models");
    if p.is_dir() { Some(p) } else { None }
}

/// Compute the manifest path for a given `name:tag`. Defaults to the
/// `library/` namespace (matches `ollama pull` un-prefixed names).
pub fn manifest_path(models_dir: &Path, name: &str, tag: &str) -> PathBuf {
    let canonical = if name.contains('/') {
        name.to_string()
    } else {
        format!("library/{name}")
    };
    models_dir
        .join("manifests")
        .join("registry.ollama.ai")
        .join(canonical)
        .join(tag)
}

/// Resolve a digest like `sha256:abc...` to its on-disk blob path.
pub fn blob_path(models_dir: &Path, digest: &str) -> PathBuf {
    // Ollama uses `sha256-<hex>` (dash, not colon) for filenames.
    let normalized = digest.replacen("sha256:", "sha256-", 1);
    models_dir.join("blobs").join(normalized)
}

/// Look up a model in the local cache. Returns `Ok(Some(layers))` if every
/// blob the manifest points at exists on disk, `Ok(None)` if no manifest
/// is present (caller falls back to registry download), or `Err(_)` on a
/// real I/O / parse error.
pub fn lookup(name: &str, tag: &str) -> std::io::Result<Option<Vec<OllamaCachedFile>>> {
    let models_dir = match ollama_models_dir() {
        Some(d) => d,
        None => return Ok(None),
    };
    let manifest_p = manifest_path(&models_dir, name, tag);
    if !manifest_p.is_file() {
        return Ok(None);
    }
    let manifest_bytes = fs::read(&manifest_p)?;
    let manifest: OciManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut out = Vec::with_capacity(manifest.layers.len());
    for layer in manifest.layers {
        let blob_p = blob_path(&models_dir, &layer.digest);
        if !blob_p.is_file() {
            // Manifest references a layer we don't have on disk — treat as
            // a cache miss so the caller falls through to the registry
            // download. This handles partial / interrupted `ollama pull`
            // cases.
            return Ok(None);
        }
        out.push(OllamaCachedFile {
            media_type: layer.media_type,
            digest: layer.digest,
            size: layer.size,
            blob_path: blob_p,
        });
    }
    Ok(Some(out))
}

/// Convenience: read the GGUF weights blob for a model. Returns `None` if
/// the model isn't in the local cache (caller should fall back to the
/// registry).
pub fn read_weights(name: &str, tag: &str) -> std::io::Result<Option<Vec<u8>>> {
    let layers = match lookup(name, tag)? {
        Some(l) => l,
        None => return Ok(None),
    };
    for layer in layers {
        if layer.is_weights() {
            return Ok(Some(layer.read_bytes()?));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_path_library_default() {
        let dir = PathBuf::from("/tmp/ollama");
        let p = manifest_path(&dir, "gemma4", "e2b");
        assert_eq!(
            p,
            PathBuf::from("/tmp/ollama/manifests/registry.ollama.ai/library/gemma4/e2b"),
        );
    }

    #[test]
    fn manifest_path_user_namespace() {
        let dir = PathBuf::from("/tmp/ollama");
        let p = manifest_path(&dir, "myorg/mymodel", "v1");
        assert_eq!(
            p,
            PathBuf::from("/tmp/ollama/manifests/registry.ollama.ai/myorg/mymodel/v1"),
        );
    }

    #[test]
    fn blob_path_translates_colon_to_dash() {
        let dir = PathBuf::from("/tmp/ollama");
        let p = blob_path(&dir, "sha256:abc123");
        assert_eq!(p, PathBuf::from("/tmp/ollama/blobs/sha256-abc123"));
    }
}
