//! Lazy GPU weight buffer cache.
//!
//! Each tensor in the GGUF gets uploaded to a `wgpu::Buffer` on first access; future
//! calls return clones of the same buffer (wgpu Buffers are Arc-internally, so
//! cloning is `Arc::clone` and free). Eliminates the per-call weight upload that
//! dominates `forward_token_gpu` cost.
//!
//! Two access modes:
//!
//! * Sync (`buffer`, `buffer_tiles`) — borrows tensor bytes from the in-memory reader
//!   and uploads. Only valid for `GgufReader::is_in_memory()` readers; errors otherwise.
//!   Used by all native / test callers.
//! * Async (`buffer_async`, `buffer_tiles_async`) — fetches the bytes through the
//!   reader's `TensorFetcher`, uploads, and drops the temporary `Vec<u8>` immediately.
//!   The streaming wasm32 path uses this so peak CPU memory stays bounded.
//!
//! Both paths populate the same buffer cache: once a tensor is on the GPU it doesn't
//! matter how it got there.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, RullamaError};
use crate::gguf::{GgmlDtype, GgufReader};

/// One tile of a row-tiled tensor.
pub struct TiledTensor {
    pub buffer: wgpu::Buffer,
    /// Index of the first row (along the slow / second axis) covered by this buffer.
    pub row_start: usize,
    /// Number of rows covered.
    pub n_rows: usize,
}

pub struct WeightCache {
    reader: Arc<GgufReader>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: RefCell<HashMap<String, wgpu::Buffer>>,
    tiles: RefCell<HashMap<(String, usize), Vec<wgpu::Buffer>>>,
    tile_meta: RefCell<HashMap<(String, usize), Vec<(usize, usize)>>>,
}

impl WeightCache {
    pub fn new(reader: Arc<GgufReader>, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            reader,
            device,
            queue,
            buffers: RefCell::new(HashMap::new()),
            tiles: RefCell::new(HashMap::new()),
            tile_meta: RefCell::new(HashMap::new()),
        }
    }

    /// Borrow of the underlying GGUF reader (for callers that occasionally need an
    /// f32 dequant outside the GPU buffer path — e.g. the small RoPE freq-factors tensor).
    pub fn reader(&self) -> &GgufReader { &self.reader }

    /// Internal: create+upload a single GPU buffer from a slice.
    fn upload(&self, name: &str, bytes: &[u8]) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytes);
        buf
    }

    /// Get the GPU buffer for the named tensor, uploading on first access. Sync path:
    /// borrows the bytes directly from an in-memory reader. Errors on a streaming reader.
    pub fn buffer(&self, name: &str) -> Result<wgpu::Buffer> {
        if let Some(b) = self.buffers.borrow().get(name) {
            return Ok(b.clone());
        }
        let bytes = self.reader.tensor_bytes(name)?;
        let buf = self.upload(name, bytes);
        let cloned = buf.clone();
        self.buffers.borrow_mut().insert(name.to_string(), buf);
        Ok(cloned)
    }

    /// Get the GPU buffer for the named tensor, uploading on first access. Async path:
    /// works for both in-memory and streaming readers. The fetched `Vec<u8>` is dropped
    /// the moment the upload finishes — important for wasm32 (4 GB linear-memory cap).
    pub async fn buffer_async(&self, name: &str) -> Result<wgpu::Buffer> {
        if let Some(b) = self.buffers.borrow().get(name) {
            return Ok(b.clone());
        }
        let bytes = self.reader.fetch_tensor_bytes(name).await?;
        let buf = self.upload(name, &bytes);
        drop(bytes);
        let cloned = buf.clone();
        self.buffers.borrow_mut().insert(name.to_string(), buf);
        Ok(cloned)
    }

    /// Best-effort buffer fetch: Ok(None) if the tensor is absent.
    pub fn buffer_opt(&self, name: &str) -> Result<Option<wgpu::Buffer>> {
        if self.reader.tensor(name).is_err() {
            return Ok(None);
        }
        self.buffer(name).map(Some)
    }

    /// Async variant of [`buffer_opt`].
    pub async fn buffer_opt_async(&self, name: &str) -> Result<Option<wgpu::Buffer>> {
        if self.reader.tensor(name).is_err() {
            return Ok(None);
        }
        self.buffer_async(name).await.map(Some)
    }

    /// Look up a tensor's GGML dtype (without uploading).
    pub fn dtype(&self, name: &str) -> Result<GgmlDtype> {
        Ok(self.reader.tensor(name)?.dtype)
    }

    pub fn cached_count(&self) -> usize {
        self.buffers.borrow().len()
    }

    pub fn cached_bytes(&self) -> u64 {
        let single: u64 = self.buffers.borrow().values().map(|b| b.size()).sum();
        let tiled: u64 = self.tiles.borrow().values()
            .flat_map(|v| v.iter().map(|b| b.size()))
            .sum();
        single + tiled
    }

    /// Internal: compute the row tiling layout for a 2-D quantized tensor.
    fn tile_layout(&self, name: &str, max_bytes_per_tile: usize) -> Result<TileLayout> {
        let desc = self.reader.tensor(name)?;
        if desc.dims.len() != 2 {
            return Err(RullamaError::Inference(format!(
                "buffer_tiles: tensor {name} has {} dims, expected 2", desc.dims.len()
            )));
        }
        let row_len = desc.dims[0] as usize;
        let n_rows = desc.dims[1] as usize;
        let block_elems = desc.dtype.block_elems();
        if row_len % block_elems != 0 {
            return Err(RullamaError::Inference(format!(
                "buffer_tiles: row_len {row_len} not multiple of block_elems {block_elems}"
            )));
        }
        let blocks_per_row = row_len / block_elems;
        let row_bytes = blocks_per_row * desc.dtype.block_bytes();
        if row_bytes == 0 {
            return Err(RullamaError::Inference(format!(
                "buffer_tiles: row_bytes is 0 for {name}"
            )));
        }
        let rows_per_tile = (max_bytes_per_tile / row_bytes).max(1);
        Ok(TileLayout { n_rows, row_bytes, rows_per_tile })
    }

    /// Split a 2-D quantized tensor along its slow (second) axis into multiple GPU
    /// buffers, each ≤ `max_bytes_per_tile` bytes. Sync; in-memory reader only.
    pub fn buffer_tiles(&self, name: &str, max_bytes_per_tile: usize) -> Result<Vec<TiledTensor>> {
        let key = (name.to_string(), max_bytes_per_tile);
        if let Some(out) = self.tiles_cached(&key) {
            return Ok(out);
        }
        let layout = self.tile_layout(name, max_bytes_per_tile)?;
        let all_bytes = self.reader.tensor_bytes(name)?;

        let mut bufs = Vec::new();
        let mut metas = Vec::new();
        let mut row_start = 0usize;
        while row_start < layout.n_rows {
            let row_end = (row_start + layout.rows_per_tile).min(layout.n_rows);
            let byte_start = row_start * layout.row_bytes;
            let byte_end   = row_end   * layout.row_bytes;
            let chunk = &all_bytes[byte_start..byte_end];
            let buf = self.upload(&format!("{name}#tile{row_start}"), chunk);
            metas.push((row_start, row_end - row_start));
            bufs.push(buf);
            row_start = row_end;
        }

        Ok(self.commit_tiles(key, bufs, metas))
    }

    /// Async variant of [`buffer_tiles`]. Fetches each tile's bytes through the
    /// fetcher (one Range request per tile when streaming), uploads, drops the
    /// temporary buffer. Works for any reader.
    pub async fn buffer_tiles_async(&self, name: &str, max_bytes_per_tile: usize)
        -> Result<Vec<TiledTensor>>
    {
        let key = (name.to_string(), max_bytes_per_tile);
        if let Some(out) = self.tiles_cached(&key) {
            return Ok(out);
        }
        let layout = self.tile_layout(name, max_bytes_per_tile)?;

        // Fetch the whole tensor once (the fetcher decides whether that's a memcpy or
        // a single Range request), then split + upload + drop.
        let all_bytes = self.reader.fetch_tensor_bytes(name).await?;

        let mut bufs = Vec::new();
        let mut metas = Vec::new();
        let mut row_start = 0usize;
        while row_start < layout.n_rows {
            let row_end = (row_start + layout.rows_per_tile).min(layout.n_rows);
            let byte_start = row_start * layout.row_bytes;
            let byte_end   = row_end   * layout.row_bytes;
            let chunk = &all_bytes[byte_start..byte_end];
            let buf = self.upload(&format!("{name}#tile{row_start}"), chunk);
            metas.push((row_start, row_end - row_start));
            bufs.push(buf);
            row_start = row_end;
        }
        drop(all_bytes);

        Ok(self.commit_tiles(key, bufs, metas))
    }

    fn tiles_cached(&self, key: &(String, usize)) -> Option<Vec<TiledTensor>> {
        let tiles = self.tiles.borrow();
        let meta = self.tile_meta.borrow();
        match (tiles.get(key), meta.get(key)) {
            (Some(bufs), Some(metas)) => Some(
                bufs.iter().zip(metas.iter())
                    .map(|(buf, &(row_start, n_rows))| TiledTensor {
                        buffer: buf.clone(), row_start, n_rows,
                    })
                    .collect()
            ),
            _ => None,
        }
    }

    fn commit_tiles(
        &self,
        key: (String, usize),
        bufs: Vec<wgpu::Buffer>,
        metas: Vec<(usize, usize)>,
    ) -> Vec<TiledTensor> {
        let result: Vec<TiledTensor> = bufs.iter().zip(metas.iter())
            .map(|(buf, &(rs, nr))| TiledTensor { buffer: buf.clone(), row_start: rs, n_rows: nr })
            .collect();
        self.tiles.borrow_mut().insert(key.clone(), bufs);
        self.tile_meta.borrow_mut().insert(key, metas);
        result
    }
}

struct TileLayout {
    n_rows: usize,
    row_bytes: usize,
    rows_per_tile: usize,
}
