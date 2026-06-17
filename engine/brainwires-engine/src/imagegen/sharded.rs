//! Sharded-safetensors loader for HuggingFace diffusers weights.
//!
//! Components like the Qwen3 text encoder ship as several `.safetensors` shards
//! plus a `model.safetensors.index.json` whose `weight_map` routes each tensor
//! name to its shard. This loader parses the index, reads each shard's header
//! once, and range-reads individual tensors on demand (via positioned reads) so
//! a 4 GB shard never lands in memory whole — the same per-tensor streaming
//! discipline as the GGUF path.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::{Result, RullamaError};

/// Parsed `model.safetensors.index.json`.
#[derive(Debug, Clone)]
pub struct ShardIndex {
    /// tensor name → shard filename.
    pub weight_map: BTreeMap<String, String>,
    /// Declared total tensor bytes (`metadata.total_size`), 0 if absent.
    pub total_size: u64,
}

#[derive(Deserialize)]
struct RawIndex {
    #[serde(default)]
    metadata: BTreeMap<String, serde_json::Value>,
    weight_map: BTreeMap<String, String>,
}

impl ShardIndex {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let raw: RawIndex = serde_json::from_slice(bytes)
            .map_err(|e| RullamaError::Image(format!("shard index JSON: {e}")))?;
        if raw.weight_map.is_empty() {
            return Err(RullamaError::Image("shard index has empty weight_map".into()));
        }
        let total_size = raw
            .metadata
            .get("total_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        Ok(Self {
            weight_map: raw.weight_map,
            total_size,
        })
    }

    /// Distinct shard filenames, sorted.
    pub fn shards(&self) -> Vec<String> {
        let mut s: Vec<String> = self
            .weight_map
            .values()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        s.sort();
        s
    }

    pub fn shard_of(&self, tensor: &str) -> Option<&str> {
        self.weight_map.get(tensor).map(String::as_str)
    }
}

// ---------- native loader ----------

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use crate::imagegen::dtype::StDtype;
    use crate::imagegen::safetensors::{read_header, SafetensorsHeader};
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use std::path::Path;

    struct ShardFile {
        file: File,
        header: SafetensorsHeader,
    }

    /// A directory of safetensors shards + their index, with per-tensor
    /// positioned reads.
    pub struct ShardedSafetensors {
        index: ShardIndex,
        shards: BTreeMap<String, ShardFile>,
    }

    impl ShardedSafetensors {
        /// Open every shard referenced by the index under `dir` and parse its
        /// header. For a single-file component (no index), pass a synthetic
        /// index via [`ShardedSafetensors::open_single`].
        pub fn open(dir: impl AsRef<Path>, index: ShardIndex) -> Result<Self> {
            let dir = dir.as_ref();
            let mut shards = BTreeMap::new();
            for shard in index.shards() {
                let path = dir.join(&shard);
                shards.insert(shard.clone(), open_shard(&path)?);
            }
            Ok(Self { index, shards })
        }

        /// Convenience: load `<dir>/<index_name>` then open all shards.
        pub fn open_dir(dir: impl AsRef<Path>, index_name: &str) -> Result<Self> {
            let dir = dir.as_ref();
            let idx_bytes = std::fs::read(dir.join(index_name))
                .map_err(|e| RullamaError::Image(format!("read {index_name}: {e}")))?;
            Self::open(dir, ShardIndex::parse(&idx_bytes)?)
        }

        /// Open a single, un-sharded `.safetensors` file as a one-shard set.
        pub fn open_single(path: impl AsRef<Path>) -> Result<Self> {
            let path = path.as_ref();
            let fname = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| RullamaError::Image("bad safetensors path".into()))?
                .to_string();
            let sf = open_shard(path)?;
            let weight_map: BTreeMap<String, String> = sf
                .header
                .tensors
                .keys()
                .map(|k| (k.clone(), fname.clone()))
                .collect();
            let index = ShardIndex {
                weight_map,
                total_size: 0,
            };
            let mut shards = BTreeMap::new();
            shards.insert(fname, sf);
            Ok(Self { index, shards })
        }

        pub fn index(&self) -> &ShardIndex {
            &self.index
        }

        pub fn names(&self) -> impl Iterator<Item = &str> {
            self.index.weight_map.keys().map(String::as_str)
        }

        pub fn has(&self, name: &str) -> bool {
            self.index.weight_map.contains_key(name)
        }

        fn locate(&self, name: &str) -> Result<(&ShardFile, (usize, usize), StDtype, &[usize])> {
            let shard = self
                .index
                .shard_of(name)
                .ok_or_else(|| RullamaError::Image(format!("tensor {name:?} not in index")))?;
            let sf = self
                .shards
                .get(shard)
                .ok_or_else(|| RullamaError::Image(format!("shard {shard:?} not opened")))?;
            let entry = sf
                .header
                .tensors
                .get(name)
                .ok_or_else(|| RullamaError::Image(format!("tensor {name:?} not in shard header")))?;
            let range = sf
                .header
                .tensor_range(name)
                .expect("entry present ⇒ range present");
            Ok((sf, range, entry.dtype, &entry.shape))
        }

        /// Shape of a tensor.
        pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
            let (_, _, _, shape) = self.locate(name)?;
            Ok(shape.to_vec())
        }

        /// Storage dtype of a tensor.
        pub fn dtype(&self, name: &str) -> Result<StDtype> {
            let (_, _, dt, _) = self.locate(name)?;
            Ok(dt)
        }

        /// Raw little-endian bytes of a tensor (positioned read from its shard).
        pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
            let (sf, (start, end), _, _) = self.locate(name)?;
            let len = end - start;
            let mut buf = vec![0u8; len];
            sf.file
                .read_exact_at(&mut buf, start as u64)
                .map_err(|e| RullamaError::Image(format!("read tensor {name:?}: {e}")))?;
            Ok(buf)
        }

        /// Dequantize a float tensor to f32.
        pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
            let (_, _, dt, _) = self.locate(name)?;
            dt.dequant_to_f32(&self.tensor_bytes(name)?)
        }
    }

    fn open_shard(path: &Path) -> Result<ShardFile> {
        let file = File::open(path)
            .map_err(|e| RullamaError::Image(format!("open {}: {e}", path.display())))?;
        // Read 8-byte header length, then the header bytes (two positioned reads).
        let mut len8 = [0u8; 8];
        file.read_exact_at(&mut len8, 0)
            .map_err(|e| RullamaError::Image(format!("read header len {}: {e}", path.display())))?;
        let header_size = u64::from_le_bytes(len8) as usize;
        let mut prefix = vec![0u8; 8 + header_size];
        file.read_exact_at(&mut prefix, 0).map_err(|e| {
            RullamaError::Image(format!("read header {} bytes {}: {e}", header_size, path.display()))
        })?;
        let header = read_header(&prefix)?;
        Ok(ShardFile { file, header })
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::ShardedSafetensors;

#[cfg(test)]
mod tests {
    use super::*;

    const INDEX: &str = r#"{
        "metadata": { "total_size": 8044936192 },
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.30.mlp.down_proj.weight": "model-00003-of-00003.safetensors",
            "model.norm.weight": "model-00003-of-00003.safetensors"
        }
    }"#;

    #[test]
    fn parse_index_and_route() {
        let idx = ShardIndex::parse(INDEX.as_bytes()).unwrap();
        assert_eq!(idx.total_size, 8044936192);
        assert_eq!(idx.weight_map.len(), 4);
        assert_eq!(idx.shards().len(), 2);
        assert_eq!(
            idx.shard_of("model.norm.weight"),
            Some("model-00003-of-00003.safetensors")
        );
        assert_eq!(idx.shard_of("nope"), None);
    }

    #[test]
    fn rejects_empty_weight_map() {
        assert!(ShardIndex::parse(br#"{"weight_map":{}}"#).is_err());
    }
}
