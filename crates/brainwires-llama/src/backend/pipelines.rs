//! Cached compute pipelines for the full forward pass.
//!
//! Built once per [`Backend`] (i.e., once per model load). Pipeline / shader-module
//! creation is expensive (tens to hundreds of milliseconds in the browser); a 35-layer
//! Gemma 4 forward dispatches dozens of compute calls per layer × hundreds of tokens,
//! so amortizing this cost is the difference between "one-shot demo" and "interactive".

use std::borrow::Cow;

use crate::kernels;

pub struct Pipelines {
    pub f16_matmul:    wgpu::ComputePipeline,
    pub q4_k_matmul:   wgpu::ComputePipeline,
    pub q6_k_matmul:   wgpu::ComputePipeline,
    pub rmsnorm:       wgpu::ComputePipeline,
    pub softcap:       wgpu::ComputePipeline,
    pub geglu:         wgpu::ComputePipeline,
    pub rope_neox:     wgpu::ComputePipeline,
    pub attention:     wgpu::ComputePipeline,
    pub residual_add:      wgpu::ComputePipeline,
    pub scale:             wgpu::ComputePipeline,
    pub rmsnorm_per_row:   wgpu::ComputePipeline,
    pub q4_k_matmul_tiled: wgpu::ComputePipeline,
    pub q6_k_matmul_tiled: wgpu::ComputePipeline,
}

impl Pipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            f16_matmul:        build(device, "f16_matmul",        kernels::F16_MATMUL),
            q4_k_matmul:       build(device, "q4_k_matmul",       kernels::Q4_K_DEQUANT_MATMUL),
            q6_k_matmul:       build(device, "q6_k_matmul",       kernels::Q6_K_DEQUANT_MATMUL),
            rmsnorm:           build(device, "rmsnorm",           kernels::RMSNORM),
            softcap:           build(device, "softcap",           kernels::SOFTCAP),
            geglu:             build(device, "geglu",             kernels::GEGLU),
            rope_neox:         build(device, "rope_neox",         kernels::ROPE_NEOX),
            attention:         build(device, "attention",         kernels::ATTENTION),
            residual_add:      build(device, "residual_add",      kernels::RESIDUAL_ADD),
            scale:             build(device, "scale",             kernels::SCALE),
            rmsnorm_per_row:   build(device, "rmsnorm_per_row",   kernels::RMSNORM_PER_ROW),
            q4_k_matmul_tiled: build(device, "q4_k_matmul_tiled", kernels::Q4_K_DEQUANT_MATMUL_TILED),
            q6_k_matmul_tiled: build(device, "q6_k_matmul_tiled", kernels::Q6_K_DEQUANT_MATMUL_TILED),
        }
    }
}

fn build(device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label}.module")),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}.pipeline")),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
