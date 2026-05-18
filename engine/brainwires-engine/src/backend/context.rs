//! Process-wide wgpu context: instance, adapter, device, queue.

use crate::error::{Result, RullamaError};

/// Holds the wgpu device and queue for the lifetime of a [`crate::api::Model`].
///
/// All inner handles are Arc-internal in wgpu, so `clone()` is cheap and lets us
/// hand the same ctx to both `Forward` (text) and `VisionForward` (multimodal)
/// without juggling Arc/Rc wrappers.
#[derive(Clone)]
pub struct WgpuCtx {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// True iff the device was created with `Features::SUBGROUP`. Kernels that
    /// require `enable subgroups;` only get registered/dispatched when this is set.
    pub has_subgroups: bool,
    /// True iff `Features::SHADER_F16` was granted. Kernels that declare
    /// `enable f16;` only get registered when this is set.
    pub has_f16: bool,
}

impl WgpuCtx {
    /// Initialize wgpu against the best available adapter.
    ///
    /// On wasm32 this binds to `navigator.gpu`; on native it picks Metal/Vulkan/DX12 via
    /// wgpu's default backend selection. Test helper, used during M0–M2 bring-up.
    pub async fn new() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| RullamaError::NoAdapter)?;

        // Opportunistically opt into perf-relevant features. Each test runs:
        //   * SUBGROUP: subgroup intrinsics (subgroupAdd/Max) collapse the
        //     barrier-tree reductions in vision attention; on AMD GCN a
        //     64-thread WG == 1 subgroup so a whole-WG reduction becomes a
        //     single op.
        //   * SUBGROUP_BARRIER: cross-subgroup ordering required by the kernel
        //     when WGs span >1 subgroup.
        //   * SHADER_F16: lets us keep tile data in workgroup memory as f16,
        //     halving LDS bandwidth and (with v3-style register tiles) doubling
        //     the in-flight tile size for the same LDS budget.
        // If an adapter lacks any of these, fall back to the f32-only path; the
        // f32 kernels stay as the correctness oracle either way.
        let adapter_feats = adapter.features();
        let adapter_info = adapter.get_info();
        let mut requested = wgpu::Features::empty();
        // SUBGROUP feature alone isn't enough — our kernels declare
        // `@workgroup_size(64)` and reduce over the whole WG via `subgroupMax`
        // / `subgroupAdd`. Those intrinsics only reduce **within a subgroup**,
        // so we need a runtime guarantee that subgroups can hold all 64 lanes
        // of the WG. `AdapterInfo::subgroup_max_size` is the ceiling the
        // adapter may produce; if it's below 64, the WG gets split across
        // multiple subgroups and the reduction is incorrect.
        //
        // Typical values (wgpu/Metal-reported):
        //   AMD GCN / Vega / Qualcomm Adreno: max ≥ 64 — kernels correct.
        //   AMD RDNA+: max 64 — correct.
        //   Apple Silicon:    max 32 — split, would silently produce wrong output → skip.
        //   NVIDIA:           max 32 — same.
        //   Intel:            max 16-32 — same.
        let subgroup_fits = adapter_info.subgroup_max_size >= 64;
        let has_subgroups = adapter_feats.contains(wgpu::Features::SUBGROUP)
            && adapter_feats.contains(wgpu::Features::SUBGROUP_BARRIER)
            && subgroup_fits;
        if has_subgroups {
            requested |= wgpu::Features::SUBGROUP;
            requested |= wgpu::Features::SUBGROUP_BARRIER;
        }
        let has_f16 = adapter_feats.contains(wgpu::Features::SHADER_F16);
        if has_f16 {
            requested |= wgpu::Features::SHADER_F16;
        }

        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rullama device"),
                required_features: requested,
                required_limits: {
                    // WebGPU spec mandates max_storage_buffers_per_shader_stage >= 8;
                    // downlevel_defaults caps it at 4 (legacy OpenGL ES targets).
                    // The Conformer block-local attention kernel needs 5 storage
                    // buffers (Q, K, V, pos_proj, out) so we bump just that field.
                    let mut l = wgpu::Limits::downlevel_defaults();
                    l.max_storage_buffers_per_shader_stage = 8;
                    // Raise LDS to whatever the adapter actually supports (Pro 555
                    // exposes 32 KB vs the WebGPU minimum 16 KB). Kernels that need
                    // >16 KB are gated; everyone else just gets more headroom.
                    l.max_compute_workgroup_storage_size = adapter_limits
                        .max_compute_workgroup_storage_size
                        .max(l.max_compute_workgroup_storage_size);
                    l.max_compute_invocations_per_workgroup = adapter_limits
                        .max_compute_invocations_per_workgroup
                        .max(l.max_compute_invocations_per_workgroup);
                    l.max_compute_workgroup_size_x = adapter_limits
                        .max_compute_workgroup_size_x
                        .max(l.max_compute_workgroup_size_x);
                    // Take whatever the adapter actually advertises for
                    // max_buffer_size / max_storage_buffer_binding_size. The
                    // downlevel defaults are 256 MiB / 128 MiB; iPad Pro
                    // reportedly advertises ~993 MB and Apple A18 likely
                    // beats the 256 MiB floor too. Raising the request
                    // costs nothing when the adapter caps it; the floor
                    // remains the spec minimum if the adapter says less.
                    l.max_buffer_size = adapter_limits.max_buffer_size.max(l.max_buffer_size);
                    l.max_storage_buffer_binding_size = adapter_limits
                        .max_storage_buffer_binding_size
                        .max(l.max_storage_buffer_binding_size);
                    l
                },
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| RullamaError::DeviceRequest(format!("{e}")))?;

        // Surface what the adapter actually granted, so iPhone runs tell us
        // whether `max_buffer_size` is the 256 MiB downlevel floor or the
        // (hopefully) larger A18-class number. Useful for diagnosing OOMs
        // on tight-RAM phones.
        let granted = device.limits();
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsValue;
            web_sys::console::log_1(&JsValue::from_str(&format!(
                "[rullama wgpu limits] max_buffer_size={} MiB \
                 max_storage_buffer_binding_size={} MiB \
                 max_storage_buffers_per_shader_stage={}",
                granted.max_buffer_size / (1024 * 1024),
                granted.max_storage_buffer_binding_size / (1024 * 1024),
                granted.max_storage_buffers_per_shader_stage,
            )));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            eprintln!(
                "[rullama wgpu limits] max_buffer_size={} MiB \
                 max_storage_buffer_binding_size={} MiB",
                granted.max_buffer_size / (1024 * 1024),
                granted.max_storage_buffer_binding_size / (1024 * 1024),
            );
        }

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            has_subgroups,
            has_f16,
        })
    }
}
