//! Process-wide wgpu context: instance, adapter, device, queue.

use crate::error::{Result, RullamaError};

/// Holds the wgpu device and queue for the lifetime of a [`crate::api::Model`].
pub struct WgpuCtx {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuCtx {
    /// Initialize wgpu against the best available adapter.
    ///
    /// On wasm32 this binds to `navigator.gpu`; on native it picks Metal/Vulkan/DX12 via
    /// wgpu's default backend selection. Test helper, used during M0–M2 bring-up.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| RullamaError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("brainwires-llama device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| RullamaError::DeviceRequest(format!("{e}")))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}
