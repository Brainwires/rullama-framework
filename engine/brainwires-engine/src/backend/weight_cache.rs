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

use crate::backend::{BindGroupCache, buf_id};
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

/// Key for tile / tile-metadata maps: tensor name + tile size in elements.
type TileKey = (String, usize);
/// Tile metadata: `(byte_offset, row_count)` per tile slice.
type TileMeta = Vec<(usize, usize)>;

pub struct WeightCache {
    reader: Arc<GgufReader>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Shared bind-group cache (same handle as `WgpuCtx::bind_cache`).
    /// Each `drop_*_destroy` invalidates any cached bind groups that
    /// reference the buffers about to be destroyed, BEFORE calling
    /// `Buffer::destroy()` — guards against the use-after-destroy
    /// observed on iOS Safari WebGPU.
    bind_cache: Arc<BindGroupCache>,
    buffers: RefCell<HashMap<String, wgpu::Buffer>>,
    tiles: RefCell<HashMap<TileKey, Vec<wgpu::Buffer>>>,
    tile_meta: RefCell<HashMap<TileKey, TileMeta>>,
}

impl WeightCache {
    pub fn new(
        reader: Arc<GgufReader>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        bind_cache: Arc<BindGroupCache>,
    ) -> Self {
        Self {
            reader,
            device,
            queue,
            bind_cache,
            buffers: RefCell::new(HashMap::new()),
            tiles: RefCell::new(HashMap::new()),
            tile_meta: RefCell::new(HashMap::new()),
        }
    }

    /// Borrow of the underlying GGUF reader (for callers that occasionally need an
    /// f32 dequant outside the GPU buffer path — e.g. the small RoPE freq-factors tensor).
    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    /// Shared `Arc` to the underlying GGUF reader. Used by callers that need to
    /// re-build a sibling like `VisionForward` after the cache + struct was
    /// released to free GPU memory (the rebuild has to re-read `VisionConfig::from_gguf`).
    pub fn reader_arc(&self) -> Arc<GgufReader> {
        self.reader.clone()
    }

    /// Internal: create+upload a single GPU buffer from a slice.
    fn upload(&self, name: &str, bytes: &[u8]) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytes);
        crate::backend::gpu_mem::record_alloc(&format!("weight:{name}"), bytes.len() as u64);
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
        let tiled: u64 = self
            .tiles
            .borrow()
            .values()
            .flat_map(|v| v.iter().map(|b| b.size()))
            .sum();
        single + tiled
    }

    /// Evict all cached buffers whose tensor name starts with `prefix`,
    /// dropping the Rust handles only (no explicit `destroy()`). Returns
    /// the number of entries removed (single + tiled combined).
    ///
    /// **Safe to call mid-step**, even while in-flight GPU commands or
    /// cached bind groups still reference the buffers: dropping the handle
    /// doesn't invalidate the underlying `GPUBuffer`, it just lets wgpu's
    /// allocator reuse that memory for subsequent allocations in the same
    /// device. Use this at the forward→backward boundary (the backward
    /// re-fetches layers it needs) where the forward's commands may still
    /// be in flight.
    ///
    /// Does NOT promptly reclaim GPU memory on the WebGPU backend (that
    /// waits for browser GC of the dropped wrapper) — for cross-step
    /// reclaim use [`drop_prefix_destroy`](Self::drop_prefix_destroy) at a
    /// GPU-idle point instead.
    pub fn drop_prefix(&self, prefix: &str) -> usize {
        let mut removed = 0usize;
        self.buffers.borrow_mut().retain(|k, v| {
            let hit = k.starts_with(prefix);
            if hit {
                crate::backend::gpu_mem::record_free(&format!("weight:{k}"), v.size());
                removed += 1;
            }
            !hit
        });
        self.tiles.borrow_mut().retain(|(k, _), v| {
            let hit = k.starts_with(prefix);
            if hit {
                for b in v.iter() {
                    crate::backend::gpu_mem::record_free(&format!("weight:{k}"), b.size());
                }
                removed += 1;
            }
            !hit
        });
        self.tile_meta
            .borrow_mut()
            .retain(|(k, _), _| !k.starts_with(prefix));
        removed
    }

    /// Like [`drop_prefix`](Self::drop_prefix) but ALSO calls
    /// `wgpu::Buffer::destroy()` on every evicted buffer to force prompt
    /// GPU-memory reclaim.
    ///
    /// **Only call at a GPU-idle point** — after a fence / map / readback
    /// that guarantees no in-flight command (and no cached bind group
    /// about to be re-used) references these buffers. `destroy()` while a
    /// buffer is still referenced by pending work or a live bind group is
    /// a use-after-destroy: on iOS Safari WebGPU it crashes the tab (we
    /// observed training die at the head→backward transition when destroy
    /// fired at the backward *start*, before the forward's commands had
    /// drained).
    ///
    /// On the WebGPU backend dropping the handle alone leaves the
    /// `GPUBuffer` resident until GC; `destroy()` reclaims it immediately
    /// so the next training step's forward re-cache starts from genuinely
    /// freed VRAM instead of stacking on the previous step's leftovers and
    /// crossing the iOS WebContent ceiling → jetsam. On native it frees
    /// immediately either way. Used by the training step at end-of-step
    /// (post loss-readback, GPU drained) and by
    /// `Model::release_vision_weights` between inference turns.
    pub fn drop_prefix_destroy(&self, prefix: &str) -> usize {
        // **Use-after-destroy guard.** Collect ids of every buffer
        // about to be destroyed and invalidate any cached bind groups
        // referencing them BEFORE we call `Buffer::destroy()`. On iOS
        // Safari WebGPU a bind group referencing a destroyed buffer
        // becomes a device-lost trigger on next use; per WebGPU spec
        // destroy is supposed to be safe but WebKit's implementation
        // is observably non-compliant (bug 302711 family).
        let mut victims: Vec<u64> = Vec::new();
        for (k, v) in self.buffers.borrow().iter() {
            if k.starts_with(prefix) {
                victims.push(buf_id(v));
            }
        }
        for ((k, _), tiles) in self.tiles.borrow().iter() {
            if k.starts_with(prefix) {
                for b in tiles {
                    victims.push(buf_id(b));
                }
            }
        }
        self.bind_cache.invalidate_buffers(&victims);

        let mut removed = 0usize;
        self.buffers.borrow_mut().retain(|k, v| {
            if k.starts_with(prefix) {
                crate::backend::gpu_mem::record_free(&format!("weight:{k}"), v.size());
                v.destroy();
                removed += 1;
                false
            } else {
                true
            }
        });
        self.tiles.borrow_mut().retain(|(k, _), v| {
            if k.starts_with(prefix) {
                for b in v.iter() {
                    crate::backend::gpu_mem::record_free(&format!("weight:{k}"), b.size());
                    b.destroy();
                }
                removed += 1;
                false
            } else {
                true
            }
        });
        self.tile_meta
            .borrow_mut()
            .retain(|(k, _), _| !k.starts_with(prefix));
        removed
    }

    /// Single-pass targeted destroy for the fwd→bwd boundary on iOS.
    /// Destroys every cached `blk.{i}.*` weight where `i` is in
    /// `[start_layer, end_layer)`, in ONE iteration through the cache
    /// instead of the N separate `drop_prefix_destroy` calls the caller
    /// would otherwise make. On iOS Safari WebGPU each
    /// `GPUBuffer.destroy()` is an IPC round-trip to the GPU process;
    /// firing 25 × ~7 = 175 of them in a tight loop with separate
    /// HashMap traversals was empirically tripping jetsam right at the
    /// forward→head transition (real-device trail: `step 2 forward 35/35
    /// gpuMiB=1417` → 💥). One pass through, one retain closure, fewer
    /// IPC dispatches.
    ///
    /// Returns the number of cache entries removed.
    pub fn drop_blk_layer_range_destroy(&self, start_layer: u32, end_layer: u32) -> usize {
        if end_layer <= start_layer {
            return 0;
        }
        // Parse the "blk.{N}." prefix out of a key without allocating;
        // returns the layer number or None if the key doesn't match the
        // "blk.<digits>.<rest>" shape.
        fn parse_blk_layer(key: &str) -> Option<u32> {
            let rest = key.strip_prefix("blk.")?;
            let dot = rest.find('.')?;
            rest[..dot].parse().ok()
        }
        let in_range = |key: &str| -> bool {
            match parse_blk_layer(key) {
                Some(n) => n >= start_layer && n < end_layer,
                None => false,
            }
        };

        // **Use-after-destroy guard** — see drop_prefix_destroy.
        let mut victims: Vec<u64> = Vec::new();
        for (k, v) in self.buffers.borrow().iter() {
            if in_range(k) {
                victims.push(buf_id(v));
            }
        }
        for ((k, _), tiles) in self.tiles.borrow().iter() {
            if in_range(k) {
                for b in tiles {
                    victims.push(buf_id(b));
                }
            }
        }
        self.bind_cache.invalidate_buffers(&victims);

        let mut removed = 0usize;
        self.buffers.borrow_mut().retain(|k, v| {
            if in_range(k) {
                crate::backend::gpu_mem::record_free(&format!("weight:{k}"), v.size());
                v.destroy();
                removed += 1;
                false
            } else {
                true
            }
        });
        self.tiles.borrow_mut().retain(|(k, _), v| {
            if in_range(k) {
                for b in v.iter() {
                    crate::backend::gpu_mem::record_free(&format!("weight:{k}"), b.size());
                    b.destroy();
                }
                removed += 1;
                false
            } else {
                true
            }
        });
        self.tile_meta.borrow_mut().retain(|(k, _), _| !in_range(k));
        removed
    }

    /// Internal: compute the row tiling layout for a 2-D quantized tensor.
    fn tile_layout(&self, name: &str, max_bytes_per_tile: usize) -> Result<TileLayout> {
        let desc = self.reader.tensor(name)?;
        if desc.dims.len() != 2 {
            return Err(RullamaError::Inference(format!(
                "buffer_tiles: tensor {name} has {} dims, expected 2",
                desc.dims.len()
            )));
        }
        let row_len = desc.dims[0] as usize;
        let n_rows = desc.dims[1] as usize;
        let block_elems = desc.dtype.block_elems();
        if !row_len.is_multiple_of(block_elems) {
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
        Ok(TileLayout {
            n_rows,
            row_bytes,
            rows_per_tile,
        })
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
            let byte_end = row_end * layout.row_bytes;
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
    pub async fn buffer_tiles_async(
        &self,
        name: &str,
        max_bytes_per_tile: usize,
    ) -> Result<Vec<TiledTensor>> {
        let key = (name.to_string(), max_bytes_per_tile);
        if let Some(out) = self.tiles_cached(&key) {
            return Ok(out);
        }
        let layout = self.tile_layout(name, max_bytes_per_tile)?;

        // Per-tile fetch — only one tile's bytes live in wasm linear memory at a
        // time. The old code path pulled the whole tensor (315 MiB for
        // `token_embd.weight` in gemma4:e2b) into one `Vec<u8>` before tiling,
        // which on iPhone 16e (8 GB shared RAM) was the spike that crashed the
        // WebContent process during the first `step()` — even with 1 GB
        // `max_buffer_size`, the wasm-side 315 MB allocation on top of ~2 GB of
        // already-resident layer weights tipped iOS Jetsam over.
        let mut bufs = Vec::new();
        let mut metas = Vec::new();
        let mut row_start = 0usize;
        while row_start < layout.n_rows {
            let row_end = (row_start + layout.rows_per_tile).min(layout.n_rows);
            let byte_start = (row_start * layout.row_bytes) as u64;
            let byte_end = (row_end * layout.row_bytes) as u64;
            let chunk = self
                .reader
                .fetch_tensor_range(name, byte_start, byte_end - byte_start)
                .await?;
            let buf = self.upload(&format!("{name}#tile{row_start}"), &chunk);
            drop(chunk);
            metas.push((row_start, row_end - row_start));
            bufs.push(buf);
            row_start = row_end;
        }

        Ok(self.commit_tiles(key, bufs, metas))
    }

    fn tiles_cached(&self, key: &(String, usize)) -> Option<Vec<TiledTensor>> {
        let tiles = self.tiles.borrow();
        let meta = self.tile_meta.borrow();
        match (tiles.get(key), meta.get(key)) {
            (Some(bufs), Some(metas)) => Some(
                bufs.iter()
                    .zip(metas.iter())
                    .map(|(buf, &(row_start, n_rows))| TiledTensor {
                        buffer: buf.clone(),
                        row_start,
                        n_rows,
                    })
                    .collect(),
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
        let result: Vec<TiledTensor> = bufs
            .iter()
            .zip(metas.iter())
            .map(|(buf, &(rs, nr))| TiledTensor {
                buffer: buf.clone(),
                row_start: rs,
                n_rows: nr,
            })
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
