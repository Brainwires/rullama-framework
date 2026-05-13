//! wgpu backend: device + queue + pipeline cache + buffer allocator.

mod context;
pub mod dispatch;
pub mod elementwise;
pub mod matmul;
pub mod pipelines;
mod spike;
pub mod weight_cache;

pub use context::WgpuCtx;
pub use pipelines::Pipelines;
pub use spike::compute_spike;
pub use weight_cache::WeightCache;
