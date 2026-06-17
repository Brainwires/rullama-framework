//! Async streaming weight reader over a [`BlobSource`] — the loader an
//! in-browser (or native-streaming) image forward uses. Generic over the
//! source, so it runs against HTTP-`Range` (wasm) or a file (native), and is
//! testable natively via [`FileBlobSource`].
//!
//! Builds a tensor index by reading each shard's safetensors header once
//! (`read_prefix`), then serves individual tensors with per-tensor
//! `read_range` calls — never materializing a whole multi-GB shard, matching
//! the GGUF `TensorFetcher` discipline.
//!
//! [`FileBlobSource`]: crate::imagegen::source::FileBlobSource

use std::collections::BTreeMap;

use crate::error::{Result, RullamaError};
use crate::imagegen::dtype::StDtype;
use crate::imagegen::safetensors::read_header;
use crate::imagegen::sharded::ShardIndex;
use crate::imagegen::source::BlobSource;

/// Where a tensor lives: which shard + its absolute byte range + dtype/shape.
#[derive(Debug, Clone)]
struct Loc {
    shard: String,
    start: u64,
    end: u64,
    dtype: StDtype,
    shape: Vec<usize>,
}

/// Async, streaming view over a set of safetensors shards behind a `BlobSource`.
pub struct StreamingShards<S: BlobSource> {
    src: S,
    index: BTreeMap<String, Loc>,
}

impl<S: BlobSource> StreamingShards<S> {
    /// Build the tensor index by reading each shard's header. `shards` is the
    /// list of component shard filenames (from a `ShardIndex`, or a single
    /// file). `header_probe` bounds the header read (headers are KB–MB).
    pub async fn open(src: S, shards: &[String]) -> Result<Self> {
        let mut index = BTreeMap::new();
        for shard in shards {
            // read the 8-byte length, then exactly the header bytes.
            let len8 = src.read_range(shard, 0, 8).await?;
            let header_size = u64::from_le_bytes(
                len8.as_slice().try_into().map_err(|_| RullamaError::Image("short header len".into()))?,
            );
            let prefix = src.read_prefix(shard, 8 + header_size as usize).await?;
            let hdr = read_header(&prefix)?;
            let data_start = hdr.data_start as u64;
            for (name, e) in hdr.tensors {
                index.insert(
                    name,
                    Loc {
                        shard: shard.clone(),
                        start: data_start + e.data_offsets.0 as u64,
                        end: data_start + e.data_offsets.1 as u64,
                        dtype: e.dtype,
                        shape: e.shape,
                    },
                );
            }
        }
        if index.is_empty() {
            return Err(RullamaError::Image("no tensors across shards".into()));
        }
        Ok(Self { src, index })
    }

    /// Open from an in-memory `ShardIndex` (its distinct shards).
    pub async fn open_index(src: S, index: &ShardIndex) -> Result<Self> {
        Self::open(src, &index.shards()).await
    }

    /// Open a single un-sharded safetensors file.
    pub async fn open_single(src: S, filename: &str) -> Result<Self> {
        Self::open(src, std::slice::from_ref(&filename.to_string())).await
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|l| l.shape.as_slice())
    }

    pub fn dtype(&self, name: &str) -> Option<StDtype> {
        self.index.get(name).map(|l| l.dtype)
    }

    /// Stream a tensor's raw bytes (one `read_range`).
    pub async fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let l = self
            .index
            .get(name)
            .ok_or_else(|| RullamaError::Image(format!("tensor {name:?} not in index")))?;
        self.src.read_range(&l.shard, l.start, l.end - l.start).await
    }

    /// Stream + dequantize a float tensor to f32.
    pub async fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let dt = self
            .index
            .get(name)
            .ok_or_else(|| RullamaError::Image(format!("tensor {name:?} not in index")))?
            .dtype;
        dt.dequant_to_f32(&self.tensor_bytes(name).await?)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::imagegen::source::FileBlobSource;

    // Streams the real VAE (single file) via FileBlobSource and checks a couple
    // of tensor shapes + a dequant — validates the async streaming loader logic
    // against ground truth (skipped if the weights aren't present).
    #[test]
    fn streams_real_vae_if_present() {
        let path = "weights/Z-Image-Turbo/vae/diffusion_pytorch_model.safetensors";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: {path} not present");
            return;
        }
        let src = FileBlobSource::new("weights/Z-Image-Turbo/vae");
        let ss = pollster::block_on(StreamingShards::open_single(
            src,
            "diffusion_pytorch_model.safetensors",
        ))
        .unwrap();
        assert!(ss.has("decoder.conv_in.weight"));
        // conv_in: [512, 16, 3, 3]
        assert_eq!(ss.shape("decoder.conv_in.weight").unwrap(), &[512, 16, 3, 3]);
        let w = pollster::block_on(ss.tensor_f32("decoder.conv_out.weight")).unwrap();
        assert_eq!(w.len(), 3 * 128 * 3 * 3);
        assert!(w.iter().all(|v| v.is_finite()));
    }
}
