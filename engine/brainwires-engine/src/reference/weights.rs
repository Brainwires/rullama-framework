//! Lazy weight access for the CPU reference forward pass.
//!
//! Each `load(name)` call reads + dequantizes the named tensor into a fresh `Vec<f32>`.
//! No caching: callers drop weights when they're done with a layer to keep peak memory
//! bounded. This is fine for parity testing where wall-clock doesn't matter.

use std::sync::Arc;

use crate::error::Result;
use crate::gguf::{
    GgufReader, dequant_expert_slice_to_f32, dequant_expert_slice_to_f32_async,
    dequant_row_to_f32, dequant_row_to_f32_async, dequant_tensor_to_f32,
    dequant_tensor_to_f32_async,
};

/// Wrapper that owns/shares an `Arc<GgufReader>` and serves f32 dequant on demand.
#[derive(Clone)]
pub struct Weights {
    reader: Arc<GgufReader>,
}

impl Weights {
    pub fn new(reader: Arc<GgufReader>) -> Self {
        Self { reader }
    }

    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }
    pub fn reader_arc(&self) -> Arc<GgufReader> {
        self.reader.clone()
    }

    /// Load and dequantize a whole tensor into f32. Allocates.
    pub fn load(&self, name: &str) -> Result<Vec<f32>> {
        dequant_tensor_to_f32(&self.reader, name)
    }

    /// Load and dequantize a single row of a 2-D tensor into f32.
    pub fn load_row(&self, name: &str, row_idx: usize) -> Result<Vec<f32>> {
        dequant_row_to_f32(&self.reader, name, row_idx)
    }

    /// Best-effort load: returns Ok(None) if the tensor isn't present.
    pub fn load_opt(&self, name: &str) -> Result<Option<Vec<f32>>> {
        match self.reader.tensor(name) {
            Ok(_) => self.load(name).map(Some),
            Err(_) => Ok(None),
        }
    }

    /// Whether the named tensor exists (cheap descriptor lookup, no data read).
    pub fn has(&self, name: &str) -> bool {
        self.reader.tensor(name).is_ok()
    }

    /// Load one expert's 2-D `[in, out]` slice of a 3-D stacked MoE tensor
    /// (`blk.N.ffn_*_exps.weight`). Keeps CPU-oracle peak memory at one
    /// expert, not all of them.
    pub fn load_expert(&self, name: &str, expert_idx: usize) -> Result<Vec<f32>> {
        dequant_expert_slice_to_f32(&self.reader, name, expert_idx)
    }

    // ---------- async (streaming-safe) variants ----------

    /// Async load: works for both in-memory and streaming readers. Used by the GPU
    /// forward path so it can run on browser-streamed GGUFs.
    pub async fn load_async(&self, name: &str) -> Result<Vec<f32>> {
        dequant_tensor_to_f32_async(&self.reader, name).await
    }

    pub async fn load_row_async(&self, name: &str, row_idx: usize) -> Result<Vec<f32>> {
        dequant_row_to_f32_async(&self.reader, name, row_idx).await
    }

    pub async fn load_opt_async(&self, name: &str) -> Result<Option<Vec<f32>>> {
        match self.reader.tensor(name) {
            Ok(_) => self.load_async(name).await.map(Some),
            Err(_) => Ok(None),
        }
    }

    pub async fn load_expert_async(&self, name: &str, expert_idx: usize) -> Result<Vec<f32>> {
        dequant_expert_slice_to_f32_async(&self.reader, name, expert_idx).await
    }
}
