//! Ollama image-model OCI manifest parser.
//!
//! Unlike the single-blob LLM packaging (`application/vnd.ollama.image.model`),
//! Ollama's experimental image-generation models store **one safetensors blob
//! per logical tensor**, addressed by content digest, plus JSON config blobs.
//! See Ollama's `x/imagegen/manifest/manifest.go` and `docs/blob-format.md`.
//!
//! Manifest shape (OCI v2):
//! ```json
//! {
//!   "schemaVersion": 2,
//!   "mediaType": "application/vnd.ollama.image",
//!   "config": { "mediaType": "application/vnd.ollama.image.json",
//!               "digest": "sha256:…", "size": 473 },
//!   "layers": [
//!     { "mediaType": "application/vnd.ollama.image.tensor",
//!       "digest": "sha256:…", "size": 13107200,
//!       "name": "transformer/layers.0.attn.to_q.weight" },
//!     { "mediaType": "application/vnd.ollama.image.json",
//!       "digest": "sha256:…", "size": 1234,
//!       "name": "model_index.json" },
//!     …
//!   ]
//! }
//! ```
//!
//! Tensor names carry a component prefix (`text_encoder/`, `transformer/`,
//! `vae/`, …). [`ImageManifest::component`] returns a view with the prefix
//! stripped, mirroring Ollama's `LoadWeightsFromManifest(<component>)`.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::{Result, RullamaError};

/// Media type for a per-tensor safetensors blob.
pub const MEDIA_TENSOR: &str = "application/vnd.ollama.image.tensor";
/// Media type for a JSON config blob (`model_index.json`, `config.json`, …).
pub const MEDIA_JSON: &str = "application/vnd.ollama.image.json";

/// One manifest layer: a content-addressed blob plus its logical name.
#[derive(Debug, Clone)]
pub struct BlobRef {
    /// Component-prefixed logical name, e.g. `transformer/layers.0.attn.to_q.weight`
    /// or `model_index.json`. Empty for the (nameless) config descriptor.
    pub name: String,
    /// OCI digest, `sha256:<hex>` as written in the manifest.
    pub digest: String,
    /// Blob size in bytes.
    pub size: u64,
    /// Layer media type (one of [`MEDIA_TENSOR`] / [`MEDIA_JSON`]).
    pub media_type: String,
}

impl BlobRef {
    /// On-disk blob filename under `~/.ollama/models/blobs/`, i.e. the digest
    /// with its single `:` replaced by `-` (`sha256:ab…` → `sha256-ab…`).
    /// Matches Ollama's `manifest.go` filesystem mapping.
    pub fn blob_filename(&self) -> String {
        self.digest.replacen(':', "-", 1)
    }

    pub fn is_tensor(&self) -> bool {
        self.media_type == MEDIA_TENSOR
    }
    pub fn is_json(&self) -> bool {
        self.media_type == MEDIA_JSON
    }
}

/// Parsed image-model manifest: every layer, plus a name→layer index.
#[derive(Debug, Clone)]
pub struct ImageManifest {
    /// The config descriptor (manifest `config` field), if present.
    pub config: Option<BlobRef>,
    /// All layers in manifest order.
    pub layers: Vec<BlobRef>,
    /// `name` → index into `layers`. Built once for O(log n) lookup.
    by_name: BTreeMap<String, usize>,
}

// ---- raw serde shapes (OCI manifest) ----

#[derive(Deserialize)]
struct RawDescriptor {
    #[serde(rename = "mediaType")]
    media_type: Option<String>,
    digest: Option<String>,
    size: Option<u64>,
    name: Option<String>,
}

#[derive(Deserialize)]
struct RawManifest {
    config: Option<RawDescriptor>,
    #[serde(default)]
    layers: Vec<RawDescriptor>,
}

impl ImageManifest {
    /// Parse a manifest JSON document.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let raw: RawManifest = serde_json::from_slice(bytes)
            .map_err(|e| RullamaError::Image(format!("manifest JSON: {e}")))?;

        let to_blob = |d: RawDescriptor| -> Result<BlobRef> {
            let digest = d
                .digest
                .ok_or_else(|| RullamaError::Image("manifest layer missing digest".into()))?;
            Ok(BlobRef {
                name: d.name.unwrap_or_default(),
                digest,
                size: d.size.unwrap_or(0),
                media_type: d.media_type.unwrap_or_default(),
            })
        };

        let config = raw.config.map(to_blob).transpose()?;

        let mut layers = Vec::with_capacity(raw.layers.len());
        let mut by_name = BTreeMap::new();
        for d in raw.layers {
            let blob = to_blob(d)?;
            if !blob.name.is_empty() {
                by_name.insert(blob.name.clone(), layers.len());
            }
            layers.push(blob);
        }

        if layers.is_empty() {
            return Err(RullamaError::Image("manifest has no layers".into()));
        }

        Ok(Self {
            config,
            layers,
            by_name,
        })
    }

    /// Look up a layer by its full (component-prefixed) name.
    pub fn get(&self, name: &str) -> Option<&BlobRef> {
        self.by_name.get(name).map(|&i| &self.layers[i])
    }

    /// Iterate tensor blobs (skips JSON config layers) whose name starts with
    /// `<component>/`, yielding `(stripped_name, &BlobRef)` — the prefix is
    /// removed so callers see the component-local tensor names that the model
    /// code expects (mirrors Ollama's per-component load).
    pub fn component<'a>(
        &'a self,
        component: &'a str,
    ) -> impl Iterator<Item = (&'a str, &'a BlobRef)> + 'a {
        let pfx = format!("{component}/");
        self.layers.iter().filter_map(move |b| {
            if !b.is_tensor() {
                return None;
            }
            b.name.strip_prefix(&pfx).map(|stripped| (stripped, b))
        })
    }

    /// Find the JSON config blob with the given name (e.g. `model_index.json`).
    pub fn json_blob(&self, name: &str) -> Option<&BlobRef> {
        self.get(name).filter(|b| b.is_json())
    }

    /// Count of tensor (non-JSON) layers.
    pub fn tensor_count(&self) -> usize {
        self.layers.iter().filter(|b| b.is_tensor()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "schemaVersion": 2,
        "mediaType": "application/vnd.ollama.image",
        "config": {
            "mediaType": "application/vnd.ollama.image.json",
            "digest": "sha256:c0ffee",
            "size": 473
        },
        "layers": [
            { "mediaType": "application/vnd.ollama.image.tensor",
              "digest": "sha256:aaa", "size": 100,
              "name": "transformer/layers.0.attn.to_q.weight" },
            { "mediaType": "application/vnd.ollama.image.tensor",
              "digest": "sha256:bbb", "size": 200,
              "name": "vae/decoder.conv_in.weight" },
            { "mediaType": "application/vnd.ollama.image.json",
              "digest": "sha256:ccc", "size": 50,
              "name": "model_index.json" }
        ]
    }"#;

    #[test]
    fn parses_layers_and_config() {
        let m = ImageManifest::parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(m.layers.len(), 3);
        assert_eq!(m.tensor_count(), 2);
        assert_eq!(m.config.as_ref().unwrap().digest, "sha256:c0ffee");
    }

    #[test]
    fn digest_to_blob_filename() {
        let m = ImageManifest::parse(SAMPLE.as_bytes()).unwrap();
        let q = m.get("transformer/layers.0.attn.to_q.weight").unwrap();
        assert_eq!(q.blob_filename(), "sha256-aaa");
        assert_eq!(q.size, 100);
    }

    #[test]
    fn component_strips_prefix() {
        let m = ImageManifest::parse(SAMPLE.as_bytes()).unwrap();
        let xf: Vec<_> = m.component("transformer").collect();
        assert_eq!(xf.len(), 1);
        assert_eq!(xf[0].0, "layers.0.attn.to_q.weight");
        // VAE component sees only its own tensor.
        let vae: Vec<_> = m.component("vae").collect();
        assert_eq!(vae.len(), 1);
        assert_eq!(vae[0].0, "decoder.conv_in.weight");
    }

    #[test]
    fn json_blob_lookup() {
        let m = ImageManifest::parse(SAMPLE.as_bytes()).unwrap();
        assert!(m.json_blob("model_index.json").is_some());
        // A tensor name is not a JSON blob.
        assert!(m.json_blob("vae/decoder.conv_in.weight").is_none());
    }
}
