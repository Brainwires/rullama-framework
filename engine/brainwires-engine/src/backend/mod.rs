//! wgpu backend: device + queue + pipeline cache + buffer allocator.

pub mod bind_cache;
mod context;
pub mod dispatch;
pub mod elementwise;
pub mod gpu_mem;
pub mod matmul;
pub mod pipelines;
mod spike;
pub mod weight_cache;

pub use bind_cache::{BindGroupCache, CacheKey, CachedDispatch, buf_id};
pub use context::WgpuCtx;
pub use pipelines::Pipelines;
pub use spike::compute_spike;
pub use weight_cache::WeightCache;
