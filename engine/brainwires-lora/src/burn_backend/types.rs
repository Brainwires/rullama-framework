use burn_autodiff::Autodiff;
use burn_wgpu::Wgpu;

pub(super) type WgpuBackend = Wgpu;
pub(super) type TrainBackend = Autodiff<WgpuBackend>;

/// Burn framework training backend with WGPU GPU support.
pub struct BurnBackend;

impl BurnBackend {
    /// Create a new Burn training backend instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BurnBackend {
    fn default() -> Self {
        Self::new()
    }
}
