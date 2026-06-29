//! Bind-group cache for chained dispatchers.
//!
//! Background: on iOS Safari WebGPU every `create_bind_group` is an
//! IPC round-trip from WebContent to GPUProcess + Metal-side
//! descriptor bookkeeping. Each fresh uniform `wgpu::Buffer` is the
//! same. A LoRA-only inference workload fires ~500 dispatches per
//! generated token; without caching that's a steady ~50 ms/tok of
//! pure bind-group/uniform allocation overhead.
//!
//! Training is far worse. A single training step does prefill (20
//! tokens × 35 layers × ~50 dispatches) + final forward + head + per-
//! layer recompute (10 × ~50) + per-layer backward (10 × ~72), all
//! within ~30-60 s. That's ~30,000 fresh `(uniform, bind_group)`
//! allocations and the matching releases per step, hammering the
//! GPUProcess message pipe. WebKit bug 302711 ("Crash on iOS with
//! Time-Varying Mesh Access") is the closest documented analog — same
//! shape, render-pass side. The compute-pass analog manifests as
//! WebContent jetsam at very low tracked memory because the kill
//! happens in the GPUProcess, not WebContent.
//!
//! Design:
//!
//! - Cache key is `(pipeline_id, storage_buffer_ids…)` — the
//!   identity of the GPU resources the bind group actually binds.
//!   Stable for the lifetime of the underlying buffers.
//! - Each cache entry OWNS a dedicated uniform buffer. On cache hit,
//!   the dispatcher calls `queue.write_buffer` to update the uniform
//!   with this call's params, then reuses the cached bind group.
//!   wgpu sequences `write_buffer` before any dispatch in the same
//!   submission, so the dispatch reads the freshly-written params.
//! - Buffer slots: 0..=7 storage buffers (b0..b6 fields plus
//!   optionality). Verified worst case is
//!   `attention_backward_dq_chained` with 6 storage buffers. Pure
//!   LoRA usage continues to call `two`/`three`/`four` constructors
//!   for back-compat.
//! - Mutex (not RefCell) because wgpu handles are `Send + Sync` and
//!   the cache lives inside `WgpuCtx` which is `Clone` and shared
//!   across cloned ctx handles. The lock is held only for the
//!   HashMap lookup/insert.
//! - Cleared by `clear()` on adapter swap (`loadAdapter` /
//!   `clearAdapter`) and on KV-cache shrink. Invalidation on weight
//!   destroy is a separate path: `invalidate_buffers(&[u64])` evicts
//!   only entries whose key references any of the given buffer
//!   identities, called from `WeightCache::drop_*_destroy` BEFORE the
//!   actual `Buffer::destroy()`.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// Stable identity for a wgpu resource. **Must be stable across clones**
/// and **must NOT use the Rust struct pointer** — wgpu's resource types
/// (`Buffer`, `ComputePipeline`) are Arc-internal, so `clone()` makes a
/// new Rust struct sharing the underlying GPU resource. Stack-allocated
/// clones in different function calls can land at the same Rust pointer
/// address even though they wrap different GPU resources, which silently
/// breaks pointer-based identity. wgpu provides Hash/PartialEq impls that
/// proxy to the Arc-internal `inner` field — hash that instead.
pub fn buf_id(b: &wgpu::Buffer) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(b, &mut h);
    std::hash::Hasher::finish(&h)
}
fn pipeline_id(p: &wgpu::ComputePipeline) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(p, &mut h);
    std::hash::Hasher::finish(&h)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub pipeline: u64,
    pub b0: u64,
    pub b1: u64,
    pub b2: u64,
    pub b3: Option<u64>,
    pub b4: Option<u64>,
    pub b5: Option<u64>,
    pub b6: Option<u64>,
}

impl CacheKey {
    pub fn one(p: &wgpu::ComputePipeline, b0: &wgpu::Buffer) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: 0,
            b2: 0,
            b3: None,
            b4: None,
            b5: None,
            b6: None,
        }
    }
    pub fn two(p: &wgpu::ComputePipeline, b0: &wgpu::Buffer, b1: &wgpu::Buffer) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: buf_id(b1),
            b2: 0,
            b3: None,
            b4: None,
            b5: None,
            b6: None,
        }
    }
    pub fn three(
        p: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
    ) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: buf_id(b1),
            b2: buf_id(b2),
            b3: None,
            b4: None,
            b5: None,
            b6: None,
        }
    }
    pub fn four(
        p: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
        b3: &wgpu::Buffer,
    ) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: buf_id(b1),
            b2: buf_id(b2),
            b3: Some(buf_id(b3)),
            b4: None,
            b5: None,
            b6: None,
        }
    }
    pub fn five(
        p: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
        b3: &wgpu::Buffer,
        b4: &wgpu::Buffer,
    ) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: buf_id(b1),
            b2: buf_id(b2),
            b3: Some(buf_id(b3)),
            b4: Some(buf_id(b4)),
            b5: None,
            b6: None,
        }
    }
    pub fn six(
        p: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
        b3: &wgpu::Buffer,
        b4: &wgpu::Buffer,
        b5: &wgpu::Buffer,
    ) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: buf_id(b1),
            b2: buf_id(b2),
            b3: Some(buf_id(b3)),
            b4: Some(buf_id(b4)),
            b5: Some(buf_id(b5)),
            b6: None,
        }
    }
    // Sibling builder for 7-buffer bind groups (the head section's lm_head
    // LoRA path). The 8-arg shape is intentional symmetry with the rest of
    // the `n_buffers` constructor family above; no destructuring win to
    // collapse them.
    #[allow(clippy::too_many_arguments)]
    pub fn seven(
        p: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
        b3: &wgpu::Buffer,
        b4: &wgpu::Buffer,
        b5: &wgpu::Buffer,
        b6: &wgpu::Buffer,
    ) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: buf_id(b1),
            b2: buf_id(b2),
            b3: Some(buf_id(b3)),
            b4: Some(buf_id(b4)),
            b5: Some(buf_id(b5)),
            b6: Some(buf_id(b6)),
        }
    }

    /// True if any of this key's storage-buffer slots references one
    /// of the given ids. Used by `invalidate_buffers`.
    fn touches(&self, ids: &HashSet<u64>) -> bool {
        ids.contains(&self.b0)
            || ids.contains(&self.b1)
            || ids.contains(&self.b2)
            || self.b3.is_some_and(|id| ids.contains(&id))
            || self.b4.is_some_and(|id| ids.contains(&id))
            || self.b5.is_some_and(|id| ids.contains(&id))
            || self.b6.is_some_and(|id| ids.contains(&id))
    }
}

#[derive(Clone)]
pub struct CachedDispatch {
    /// Persistent uniform buffer owned by this cache entry. The
    /// caller writes the per-call params here BEFORE dispatching,
    /// then submits — wgpu sequences the write before the dispatch
    /// on the same queue.
    pub uniform: wgpu::Buffer,
    /// Cached bind group; references `uniform` plus the storage
    /// buffers the key was derived from.
    pub bind_group: wgpu::BindGroup,
}

pub struct BindGroupCache {
    inner: Mutex<HashMap<CacheKey, CachedDispatch>>,
}

impl BindGroupCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Get or insert a cache entry. `build` is called only on miss;
    /// it must construct both the uniform buffer and the bind group
    /// that references it. The returned handles are clones (wgpu
    /// types are Arc-internal — cheap).
    pub fn get_or_create<F>(&self, key: CacheKey, build: F) -> CachedDispatch
    where
        F: FnOnce() -> CachedDispatch,
    {
        let mut guard = self.inner.lock().unwrap();
        guard.entry(key).or_insert_with(build).clone()
    }

    /// Drop every cached entry. Called when the adapter is loaded or
    /// cleared — the buffer ids in the keys would otherwise be stale
    /// (and could collide with freshly-allocated buffers at the same
    /// address).
    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }

    /// Drop every cached entry whose key references any of the given
    /// buffer ids. Called from `WeightCache::drop_*_destroy` BEFORE
    /// the actual `Buffer::destroy()` so no cached bind group survives
    /// pointing at dead memory.
    ///
    /// On iOS Safari WebGPU, calling `Buffer::destroy()` on a buffer
    /// that has a still-live bind group referencing it is observably
    /// a use-after-destroy that surfaces as device-lost (WebKit bug
    /// 302711 family). Eager invalidation here removes that class of
    /// bug at the source.
    pub fn invalidate_buffers(&self, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        let id_set: HashSet<u64> = ids.iter().copied().collect();
        let mut guard = self.inner.lock().unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        let before = guard.len();
        guard.retain(|k, _| !k.touches(&id_set));
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("RULLAMA_TRACE_BINDCACHE").is_ok() {
            eprintln!(
                "[bindcache] invalidate_buffers: {} ids, removed {} entries ({} -> {})",
                ids.len(),
                before - guard.len(),
                before,
                guard.len()
            );
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}
