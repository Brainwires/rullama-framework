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
    /// f16-LDS variants of the Q4_K / Q6_K dequant matmul. Inner loop uses
    /// f16 multiplies (accumulator stays f32) so naga emits packed-FP16 MAD
    /// instructions on GCN 1.2+ / Apple Silicon. Built only when SHADER_F16
    /// is available; routed first by `matmul_q[46]_k_chained`.
    pub q4_k_matmul_f16lds: Option<wgpu::ComputePipeline>,
    pub q6_k_matmul_f16lds: Option<wgpu::ComputePipeline>,
    /// WG=256 non-tiled Q4_K — 4 waves per WG for better latency hiding.
    pub q4_k_matmul_wg256: wgpu::ComputePipeline,
    pub conv2d:            wgpu::ComputePipeline,
    pub avg_pool2d:        wgpu::ComputePipeline,
    pub clamp:             wgpu::ComputePipeline,
    pub quick_geglu:       wgpu::ComputePipeline,
    pub rope_2d:           wgpu::ComputePipeline,
    pub f16_matmul_batched: wgpu::ComputePipeline,
    pub f16_matmul_batched_tiled: wgpu::ComputePipeline,
    pub f16_matmul_batched_tiled_v2: wgpu::ComputePipeline,
    pub f16_matmul_batched_tiled_v3: wgpu::ComputePipeline,
    pub f16_matmul_batched_tiled_v4: wgpu::ComputePipeline,
    /// v3-layout matmul with f16 LDS storage + f16 inner-loop arithmetic.
    /// Only built when `Features::SHADER_F16` is available.
    pub f16_matmul_batched_tiled_v3_f16lds: Option<wgpu::ComputePipeline>,
    pub pos_embed_add:     wgpu::ComputePipeline,
    pub vision_attention:  wgpu::ComputePipeline,
    pub vision_attention_flash: wgpu::ComputePipeline,
    pub vision_attention_flash_q4: wgpu::ComputePipeline,
    pub vision_attention_flash_q8: wgpu::ComputePipeline,
    pub vision_attention_flash_q16: wgpu::ComputePipeline,
    /// Subgroup-collapsed flash attention. Only built when the device was created
    /// with `Features::SUBGROUP` (see [`crate::backend::WgpuCtx::has_subgroups`]).
    /// Routed automatically by [`crate::backend::dispatch::vision_attention_chained`].
    pub vision_attention_flash_subgroup: Option<wgpu::ComputePipeline>,
    /// TILE_T=64, Q=12 subgroup variant. Needs LDS ≥ 22 KB; built only when the
    /// device exceeds that ceiling.
    pub vision_attention_flash_sub_t64: Option<wgpu::ComputePipeline>,
    /// Head-major subgroup variant. Reads Q/K/V as [n_heads, n_patches, head_dim]
    /// so per-WG tile loads coalesce. Caller pre-transposes inputs and post-
    /// transposes the output via `transpose_phd_to_hpd` / `transpose_hpd_to_phd`.
    pub vision_attention_flash_sub_hpd: Option<wgpu::ComputePipeline>,
    /// f16-LDS variant of `vision_attention_flash_sub_hpd`. Halves workgroup
    /// memory footprint → ~2× higher per-CU wave occupancy. Requires
    /// `Features::SHADER_F16` and SUBGROUP.
    pub vision_attention_flash_sub_hpd_f16: Option<wgpu::ComputePipeline>,
    /// Q=16 variant of `vision_attention_flash_sub_hpd_f16`. Halves WG count
    /// at the same total work — amortises per-WG cost.
    pub vision_attention_flash_sub_hpd_f16_q16: Option<wgpu::ComputePipeline>,
    /// HPD + f16-LDS attention **without subgroups** (barrier-tree reduction).
    /// Targets devices that have SHADER_F16 but where subgroup_max_size < 64
    /// — i.e. Apple Silicon (32) / NVIDIA (32) / Intel — so our subgroup-
    /// collapsed kernels would produce wrong output. Built when has_f16 is
    /// set, regardless of subgroup support.
    pub vision_attention_flash_hpd_f16: Option<wgpu::ComputePipeline>,
    pub transpose_phd_to_hpd: wgpu::ComputePipeline,
    pub transpose_hpd_to_phd: wgpu::ComputePipeline,
    pub half_residual_add: wgpu::ComputePipeline,
    pub silu:              wgpu::ComputePipeline,
    pub glu_split:         wgpu::ComputePipeline,
    pub depthwise_conv1d:  wgpu::ComputePipeline,
    pub block_local_attention: wgpu::ComputePipeline,
    pub bf16_matmul:       wgpu::ComputePipeline,
    pub bf16_matmul_batched: wgpu::ComputePipeline,
    pub bf16_matmul_batched_tiled: wgpu::ComputePipeline,
    pub bf16_matmul_batched_tiled_v3: wgpu::ComputePipeline,
    /// f16-LDS bf16-matmul. Built only when SHADER_F16 is available.
    pub bf16_matmul_batched_tiled_v3_f16lds: Option<wgpu::ComputePipeline>,
    pub bf16_matmul_batched_tiled_v2: wgpu::ComputePipeline,
    pub scale_per_inner_dim: wgpu::ComputePipeline,
    pub add_bias_batched: wgpu::ComputePipeline,

    // --- Training kernels (M0 backward pass) ---
    /// Cross-entropy forward + backward over a single logit vector. Produces
    /// `d_logits = softmax(logits) - one_hot(target)` plus the scalar loss.
    /// Safe to call on masked positions (`target == u32::MAX`): emits zero
    /// gradient and zero loss.
    pub cross_entropy_backward: wgpu::ComputePipeline,
    /// Backward of Q4_K matmul w.r.t. the input vector.
    /// Computes `dx[i] = Σ_j dy[j] * dequant(W)[j, i]`. The weight matrix
    /// stays in Q4_K (frozen by LoRA convention) — no weight gradient.
    pub matmul_q4_k_backward_input: wgpu::ComputePipeline,
}

impl Pipelines {
    /// Build all pipelines for the given device. `has_subgroups` controls
    /// whether the subgroup-only kernels get compiled — they fail to validate
    /// when `Features::SUBGROUP` is absent. The TILE_T=64 variant is gated
    /// additionally on the device's actual LDS limit being ≥ 22 KB.
    pub fn new_with(device: &wgpu::Device, has_subgroups: bool) -> Self {
        Self::new_with_features(device, has_subgroups, false)
    }

    /// Full constructor: builds the f16-LDS variants when `has_f16` is set
    /// (caller must have requested `Features::SHADER_F16`).
    pub fn new_with_features(device: &wgpu::Device, has_subgroups: bool, has_f16: bool) -> Self {
        let mut me = Self::new(device);
        if has_subgroups {
            me.vision_attention_flash_subgroup = Some(build(
                device,
                "vision_attention_flash_subgroup",
                kernels::VISION_ATTENTION_FLASH_SUBGROUP,
            ));
            if device.limits().max_compute_workgroup_storage_size >= 23_000 {
                me.vision_attention_flash_sub_t64 = Some(build(
                    device,
                    "vision_attention_flash_sub_t64",
                    kernels::VISION_ATTENTION_FLASH_SUB_T64,
                ));
            }
            me.vision_attention_flash_sub_hpd = Some(build(
                device,
                "vision_attention_flash_sub_hpd",
                kernels::VISION_ATTENTION_FLASH_SUB_HPD,
            ));
            if has_f16 {
                me.vision_attention_flash_sub_hpd_f16 = Some(build(
                    device,
                    "vision_attention_flash_sub_hpd_f16",
                    kernels::VISION_ATTENTION_FLASH_SUB_HPD_F16,
                ));
                me.vision_attention_flash_sub_hpd_f16_q16 = Some(build(
                    device,
                    "vision_attention_flash_sub_hpd_f16_q16",
                    kernels::VISION_ATTENTION_FLASH_SUB_HPD_F16_Q16,
                ));
            }
        }
        if has_f16 {
            // Universal subgroup-free fast path — works on any adapter with
            // SHADER_F16, including the ones where our subgroup-collapsed
            // kernels would be incorrect (subgroup_max_size < 64).
            me.vision_attention_flash_hpd_f16 = Some(build(
                device,
                "vision_attention_flash_hpd_f16",
                kernels::VISION_ATTENTION_FLASH_HPD_F16,
            ));
            me.f16_matmul_batched_tiled_v3_f16lds = Some(build(
                device,
                "f16_matmul_batched_tiled_v3_f16lds",
                kernels::F16_MATMUL_BATCHED_TILED_V3_F16LDS,
            ));
            me.bf16_matmul_batched_tiled_v3_f16lds = Some(build(
                device,
                "bf16_matmul_batched_tiled_v3_f16lds",
                kernels::BF16_MATMUL_BATCHED_TILED_V3_F16LDS,
            ));
            me.q4_k_matmul_f16lds = Some(build(
                device,
                "q4_k_matmul_f16lds",
                kernels::Q4_K_DEQUANT_MATMUL_F16LDS,
            ));
            me.q6_k_matmul_f16lds = Some(build(
                device,
                "q6_k_matmul_f16lds",
                kernels::Q6_K_DEQUANT_MATMUL_F16LDS,
            ));
        }
        me
    }

    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            f16_matmul:        build(device, "f16_matmul",        kernels::F16_MATMUL),
            q4_k_matmul:       build(device, "q4_k_matmul",       kernels::Q4_K_DEQUANT_MATMUL),
            q6_k_matmul:       build(device, "q6_k_matmul",       kernels::Q6_K_DEQUANT_MATMUL),
            rmsnorm:           build(device, "rmsnorm",           kernels::RMSNORM),
            softcap:           build(device, "softcap",           kernels::SOFTCAP),
            cross_entropy_backward: build(
                device,
                "cross_entropy_backward",
                kernels::CROSS_ENTROPY_BACKWARD,
            ),
            matmul_q4_k_backward_input: build(
                device,
                "matmul_q4_k_backward_input",
                kernels::MATMUL_Q4_K_BACKWARD_INPUT,
            ),
            geglu:             build(device, "geglu",             kernels::GEGLU),
            rope_neox:         build(device, "rope_neox",         kernels::ROPE_NEOX),
            attention:         build(device, "attention",         kernels::ATTENTION),
            residual_add:      build(device, "residual_add",      kernels::RESIDUAL_ADD),
            scale:             build(device, "scale",             kernels::SCALE),
            rmsnorm_per_row:   build(device, "rmsnorm_per_row",   kernels::RMSNORM_PER_ROW),
            q4_k_matmul_tiled: build(device, "q4_k_matmul_tiled", kernels::Q4_K_DEQUANT_MATMUL_TILED),
            q6_k_matmul_tiled: build(device, "q6_k_matmul_tiled", kernels::Q6_K_DEQUANT_MATMUL_TILED),
            conv2d:            build(device, "conv2d",            kernels::CONV2D),
            avg_pool2d:        build(device, "avg_pool2d",        kernels::AVG_POOL2D),
            clamp:             build(device, "clamp",             kernels::CLAMP),
            quick_geglu:       build(device, "quick_geglu",       kernels::QUICK_GEGLU),
            rope_2d:           build(device, "rope_2d",           kernels::ROPE_2D),
            f16_matmul_batched: build(device, "f16_matmul_batched", kernels::F16_MATMUL_BATCHED),
            f16_matmul_batched_tiled: build(device, "f16_matmul_batched_tiled", kernels::F16_MATMUL_BATCHED_TILED),
            f16_matmul_batched_tiled_v2: build(device, "f16_matmul_batched_tiled_v2", kernels::F16_MATMUL_BATCHED_TILED_V2),
            f16_matmul_batched_tiled_v3: build(device, "f16_matmul_batched_tiled_v3", kernels::F16_MATMUL_BATCHED_TILED_V3),
            f16_matmul_batched_tiled_v4: build(device, "f16_matmul_batched_tiled_v4", kernels::F16_MATMUL_BATCHED_TILED_V4),
            f16_matmul_batched_tiled_v3_f16lds: None,
            bf16_matmul_batched_tiled_v3_f16lds: None,
            q4_k_matmul_f16lds: None,
            q6_k_matmul_f16lds: None,
            q4_k_matmul_wg256: build(device, "q4_k_matmul_wg256", kernels::Q4_K_DEQUANT_MATMUL_WG256),
            pos_embed_add:     build(device, "pos_embed_add",     kernels::POS_EMBED_ADD),
            vision_attention:  build(device, "vision_attention",  kernels::VISION_ATTENTION),
            vision_attention_flash: build(device, "vision_attention_flash", kernels::VISION_ATTENTION_FLASH),
            vision_attention_flash_q4: build(device, "vision_attention_flash_q4", kernels::VISION_ATTENTION_FLASH_Q4),
            vision_attention_flash_q8: build(device, "vision_attention_flash_q8", kernels::VISION_ATTENTION_FLASH_Q8),
            vision_attention_flash_q16: build(device, "vision_attention_flash_q16", kernels::VISION_ATTENTION_FLASH_Q16),
            vision_attention_flash_subgroup: None,
            vision_attention_flash_sub_t64: None,
            vision_attention_flash_sub_hpd: None,
            vision_attention_flash_sub_hpd_f16: None,
            vision_attention_flash_sub_hpd_f16_q16: None,
            vision_attention_flash_hpd_f16: None,
            transpose_phd_to_hpd: build(device, "transpose_phd_to_hpd", kernels::TRANSPOSE_PHD_TO_HPD),
            transpose_hpd_to_phd: build(device, "transpose_hpd_to_phd", kernels::TRANSPOSE_HPD_TO_PHD),
            half_residual_add: build(device, "half_residual_add", kernels::HALF_RESIDUAL_ADD),
            silu:              build(device, "silu",              kernels::SILU),
            glu_split:         build(device, "glu_split",         kernels::GLU_SPLIT),
            depthwise_conv1d:  build(device, "depthwise_conv1d",  kernels::DEPTHWISE_CONV1D),
            block_local_attention: build(device, "block_local_attention", kernels::BLOCK_LOCAL_ATTENTION),
            bf16_matmul:       build(device, "bf16_matmul",       kernels::BF16_MATMUL),
            bf16_matmul_batched: build(device, "bf16_matmul_batched", kernels::BF16_MATMUL_BATCHED),
            bf16_matmul_batched_tiled: build(device, "bf16_matmul_batched_tiled", kernels::BF16_MATMUL_BATCHED_TILED),
            bf16_matmul_batched_tiled_v3: build(device, "bf16_matmul_batched_tiled_v3", kernels::BF16_MATMUL_BATCHED_TILED_V3),
            bf16_matmul_batched_tiled_v2: build(device, "bf16_matmul_batched_tiled_v2", kernels::BF16_MATMUL_BATCHED_TILED_V2),
            scale_per_inner_dim: build(device, "scale_per_inner_dim", kernels::SCALE_PER_INNER_DIM),
            add_bias_batched: build(device, "add_bias_batched", kernels::ADD_BIAS_BATCHED),
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
