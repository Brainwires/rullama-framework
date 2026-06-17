//! Cached compute pipelines for the full forward pass.
//!
//! Built once per [`Backend`] (i.e., once per model load). Pipeline / shader-module
//! creation is expensive (tens to hundreds of milliseconds in the browser); a 35-layer
//! Gemma 4 forward dispatches dozens of compute calls per layer × hundreds of tokens,
//! so amortizing this cost is the difference between "one-shot demo" and "interactive".

use std::borrow::Cow;

use crate::kernels;

pub struct Pipelines {
    pub f16_matmul: wgpu::ComputePipeline,
    pub q4_0_matmul: wgpu::ComputePipeline,
    pub q5_0_matmul: wgpu::ComputePipeline,
    pub q8_0_matmul: wgpu::ComputePipeline,
    pub moe_router: wgpu::ComputePipeline,
    pub moe_geglu_halves: wgpu::ComputePipeline,
    pub moe_combine: wgpu::ComputePipeline,
    pub moe_expert_matmul_q4_k: wgpu::ComputePipeline,
    pub moe_expert_matmul_q5_0: wgpu::ComputePipeline,
    pub moe_expert_matmul_q8_0: wgpu::ComputePipeline,
    pub diffusion_attention: wgpu::ComputePipeline,
    pub moe_router_batched: wgpu::ComputePipeline,
    pub moe_expert_matmul_batched_q4_k: wgpu::ComputePipeline,
    pub moe_expert_matmul_batched_q5_0: wgpu::ComputePipeline,
    pub moe_expert_matmul_batched_q8_0: wgpu::ComputePipeline,
    pub moe_geglu_halves_batched: wgpu::ComputePipeline,
    pub moe_combine_batched: wgpu::ComputePipeline,
    pub q4_k_matmul: wgpu::ComputePipeline,
    pub q6_k_matmul: wgpu::ComputePipeline,
    pub rmsnorm: wgpu::ComputePipeline,
    pub softcap: wgpu::ComputePipeline,
    pub geglu: wgpu::ComputePipeline,
    pub rope_neox: wgpu::ComputePipeline,
    pub attention: wgpu::ComputePipeline,
    pub residual_add: wgpu::ComputePipeline,
    pub scale: wgpu::ComputePipeline,
    pub rmsnorm_per_row: wgpu::ComputePipeline,
    pub layernorm_affine: wgpu::ComputePipeline,
    pub groupnorm: wgpu::ComputePipeline,
    pub conv1d: wgpu::ComputePipeline,
    pub conv1d_f16: wgpu::ComputePipeline,
    pub conv_transpose1d: wgpu::ComputePipeline,
    pub conv_transpose1d_f16: wgpu::ComputePipeline,
    pub leaky_relu: wgpu::ComputePipeline,
    pub gelu_exact: wgpu::ComputePipeline,
    pub snake: wgpu::ComputePipeline,
    pub adain: wgpu::ComputePipeline,
    pub istft: wgpu::ComputePipeline,
    pub transpose2d: wgpu::ComputePipeline,
    pub nearest_upsample2x: wgpu::ComputePipeline,
    pub spec_phase: wgpu::ComputePipeline,
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
    pub conv2d: wgpu::ComputePipeline,
    pub conv2d_chf: wgpu::ComputePipeline,
    pub conv2d_chf_f16: wgpu::ComputePipeline,
    pub avg_pool2d: wgpu::ComputePipeline,
    pub avg_pool2d_half_chf: wgpu::ComputePipeline,
    pub clamp: wgpu::ComputePipeline,
    pub quick_geglu: wgpu::ComputePipeline,
    pub rope_2d: wgpu::ComputePipeline,
    pub f16_matmul_batched: wgpu::ComputePipeline,
    pub f16_matmul_batched_tiled: wgpu::ComputePipeline,
    pub f16_matmul_batched_tiled_v2: wgpu::ComputePipeline,
    pub f16_matmul_batched_tiled_v3: wgpu::ComputePipeline,
    pub f16_matmul_batched_tiled_v4: wgpu::ComputePipeline,
    /// v3-layout matmul with f16 LDS storage + f16 inner-loop arithmetic.
    /// Only built when `Features::SHADER_F16` is available.
    pub f16_matmul_batched_tiled_v3_f16lds: Option<wgpu::ComputePipeline>,
    pub pos_embed_add: wgpu::ComputePipeline,
    pub vision_attention: wgpu::ComputePipeline,
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
    pub silu: wgpu::ComputePipeline,
    pub glu_split: wgpu::ComputePipeline,
    pub depthwise_conv1d: wgpu::ComputePipeline,
    pub block_local_attention: wgpu::ComputePipeline,
    pub bf16_matmul: wgpu::ComputePipeline,
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
    /// Backward of Q4_0 matmul w.r.t. the input vector — fine-tuning on a Q4_0
    /// (QAT) base. Same `dx[i] = Σ_j dy[j] * dequant(W)[j, i]`, frozen weight.
    pub matmul_q4_0_backward_input: wgpu::ComputePipeline,
    /// Backward of Q6_K matmul w.r.t. the input vector. Same convention
    /// as the Q4_K variant — `dx[i] = Σ_j dy[j] * dequant(W)[j, i]`.
    /// Required for the tied embedding (Gemma 4's `token_embd` is Q6_K)
    /// so the output projection backward stays on-GPU.
    pub matmul_q6_k_backward_input: wgpu::ComputePipeline,
    /// RMSNorm backward w.r.t. the input. Weight `w` is frozen — no `dw`.
    pub rmsnorm_backward: wgpu::ComputePipeline,
    /// Per-row RMSNorm backward (one workgroup per row). Mirrors the
    /// per-row forward used for q/k/v head normalisations.
    pub rmsnorm_per_row_backward: wgpu::ComputePipeline,
    /// GeGLU backward — `d_gate` and `d_up` from `dy`, `gate`, `up`.
    pub geglu_backward: wgpu::ComputePipeline,
    /// NeoX RoPE backward — inverse in-place rotation of `dx`.
    pub rope_neox_backward: wgpu::ComputePipeline,
    /// Attention backward — pass 1. Produces `d_scores` (staged) and `d_q`.
    pub attention_backward_dq: wgpu::ComputePipeline,
    /// Attention backward — pass 2. Consumes pass-1 `d_scores`, produces
    /// `d_k_hist` and `d_v_hist`.
    pub attention_backward_dkv: wgpu::ComputePipeline,
    /// Compute attention softmax probabilities (Phase A–D of the forward
    /// attention kernel) without applying them against V. Used by the
    /// training backward pass to reconstruct probs from `q_post_rope`
    /// and the KV cache without modifying the forward kernel.
    pub attention_probs: wgpu::ComputePipeline,
    /// Tiny f32 row-major matmul: `y = scale * W @ x` (or `y += scale * W @ x`).
    /// Building block of the LoRA forward correction.
    pub lora_matmul_row: wgpu::ComputePipeline,
    /// Tiny f32 transposed matmul: `y = scale * Wᵀ @ x` (or `y += …`).
    /// Building block of the LoRA backward path for both `u = Bᵀ·dy` and
    /// `dx += s · Aᵀ·u`.
    pub lora_matmul_col: wgpu::ComputePipeline,
    /// Rank-1 outer-product accumulator: `out[i, j] += scale · a[i] · b[j]`.
    /// Builds both `dA` (`a=u, b=x`) and `dB` (`a=dy, b=z`) in LoRA backward.
    pub lora_outer_add: wgpu::ComputePipeline,
    /// Column extract for the embed_tokens LoRA forward.
    /// `z[r] = A[r, col]` — picks one column out of `A` shape `[rank, vocab]`.
    pub lora_embed_col_read: wgpu::ComputePipeline,
    /// Column scatter-add for the embed_tokens LoRA backward.
    /// `d_A[r, col] += scale · u[r]`.
    pub lora_embed_col_scatter_add: wgpu::ComputePipeline,
    /// Fused per-target LoRA forward correction.
    /// One dispatch does both `z = A·x` and `y += scale · B·z`,
    /// halving the per-target dispatch count from 2 to 1.
    pub lora_matmul_fused: wgpu::ComputePipeline,
    /// `lora_matmul_fused` variant where B is stored as packed f16
    /// (two elements per u32). Halves bandwidth on the vocab×rank
    /// matmul; only routed for the lm_head LoRA target.
    pub lora_matmul_fused_f16b: wgpu::ComputePipeline,
    /// AdamW optimizer step — elementwise update of `(param, m, v)` from
    /// the gradient buffer. Standard β₁/β₂ bias correction, decoupled weight
    /// decay. Drives every LoRA A and B at the end of `TrainingSession::step`.
    pub adam_step: wgpu::ComputePipeline,
    /// Single-workgroup sum-of-squares reduction. Used by
    /// `TrainingSession`'s global gradient clipping (`max_grad_norm`)
    /// to compute per-buffer SoS into a tiny scratch scalar without
    /// reading the whole gradient buffer back to host.
    pub sum_of_squares: wgpu::ComputePipeline,
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
            f16_matmul: build(device, "f16_matmul", kernels::F16_MATMUL),
            q4_0_matmul: build(device, "q4_0_matmul", kernels::Q4_0_DEQUANT_MATMUL),
            q5_0_matmul: build(device, "q5_0_matmul", kernels::Q5_0_DEQUANT_MATMUL),
            q8_0_matmul: build(device, "q8_0_matmul", kernels::Q8_0_DEQUANT_MATMUL),
            moe_router: build(device, "moe_router", kernels::MOE_ROUTER),
            moe_geglu_halves: build(device, "moe_geglu_halves", kernels::MOE_GEGLU_HALVES),
            moe_combine: build(device, "moe_combine", kernels::MOE_COMBINE),
            moe_expert_matmul_q4_k: build(
                device,
                "moe_expert_matmul_q4_k",
                kernels::MOE_EXPERT_MATMUL_Q4_K,
            ),
            moe_expert_matmul_q5_0: build(
                device,
                "moe_expert_matmul_q5_0",
                kernels::MOE_EXPERT_MATMUL_Q5_0,
            ),
            moe_expert_matmul_q8_0: build(
                device,
                "moe_expert_matmul_q8_0",
                kernels::MOE_EXPERT_MATMUL_Q8_0,
            ),
            diffusion_attention: build(device, "diffusion_attention", kernels::DIFFUSION_ATTENTION),
            moe_router_batched: build(device, "moe_router_batched", kernels::MOE_ROUTER_BATCHED),
            moe_expert_matmul_batched_q4_k: build(
                device,
                "moe_expert_matmul_batched_q4_k",
                kernels::MOE_EXPERT_MATMUL_BATCHED_Q4_K,
            ),
            moe_expert_matmul_batched_q5_0: build(
                device,
                "moe_expert_matmul_batched_q5_0",
                kernels::MOE_EXPERT_MATMUL_BATCHED_Q5_0,
            ),
            moe_expert_matmul_batched_q8_0: build(
                device,
                "moe_expert_matmul_batched_q8_0",
                kernels::MOE_EXPERT_MATMUL_BATCHED_Q8_0,
            ),
            moe_geglu_halves_batched: build(
                device,
                "moe_geglu_halves_batched",
                kernels::MOE_GEGLU_HALVES_BATCHED,
            ),
            moe_combine_batched: build(device, "moe_combine_batched", kernels::MOE_COMBINE_BATCHED),
            q4_k_matmul: build(device, "q4_k_matmul", kernels::Q4_K_DEQUANT_MATMUL),
            q6_k_matmul: build(device, "q6_k_matmul", kernels::Q6_K_DEQUANT_MATMUL),
            rmsnorm: build(device, "rmsnorm", kernels::RMSNORM),
            softcap: build(device, "softcap", kernels::SOFTCAP),
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
            matmul_q4_0_backward_input: build(
                device,
                "matmul_q4_0_backward_input",
                kernels::MATMUL_Q4_0_BACKWARD_INPUT,
            ),
            matmul_q6_k_backward_input: build(
                device,
                "matmul_q6_k_backward_input",
                kernels::MATMUL_Q6_K_BACKWARD_INPUT,
            ),
            rmsnorm_backward: build(device, "rmsnorm_backward", kernels::RMSNORM_BACKWARD),
            rmsnorm_per_row_backward: build(
                device,
                "rmsnorm_per_row_backward",
                kernels::RMSNORM_PER_ROW_BACKWARD,
            ),
            geglu_backward: build(device, "geglu_backward", kernels::GEGLU_BACKWARD),
            rope_neox_backward: build(device, "rope_neox_backward", kernels::ROPE_NEOX_BACKWARD),
            attention_backward_dq: build(
                device,
                "attention_backward_dq",
                kernels::ATTENTION_BACKWARD_DQ,
            ),
            attention_backward_dkv: build(
                device,
                "attention_backward_dkv",
                kernels::ATTENTION_BACKWARD_DKV,
            ),
            attention_probs: build(device, "attention_probs", kernels::ATTENTION_PROBS),
            lora_matmul_row: build(device, "lora_matmul_row", kernels::LORA_MATMUL_ROW),
            lora_matmul_col: build(device, "lora_matmul_col", kernels::LORA_MATMUL_COL),
            lora_outer_add: build(device, "lora_outer_add", kernels::LORA_OUTER_ADD),
            lora_embed_col_read: build(device, "lora_embed_col_read", kernels::LORA_EMBED_COL_READ),
            lora_embed_col_scatter_add: build(
                device,
                "lora_embed_col_scatter_add",
                kernels::LORA_EMBED_COL_SCATTER_ADD,
            ),
            lora_matmul_fused: build(device, "lora_matmul_fused", kernels::LORA_MATMUL_FUSED),
            lora_matmul_fused_f16b: build(
                device,
                "lora_matmul_fused_f16b",
                kernels::LORA_MATMUL_FUSED_F16B,
            ),
            adam_step: build(device, "adam_step", kernels::ADAM_STEP),
            sum_of_squares: build(device, "sum_of_squares", kernels::SUM_OF_SQUARES),
            geglu: build(device, "geglu", kernels::GEGLU),
            rope_neox: build(device, "rope_neox", kernels::ROPE_NEOX),
            attention: build(device, "attention", kernels::ATTENTION),
            residual_add: build(device, "residual_add", kernels::RESIDUAL_ADD),
            scale: build(device, "scale", kernels::SCALE),
            rmsnorm_per_row: build(device, "rmsnorm_per_row", kernels::RMSNORM_PER_ROW),
            layernorm_affine: build(device, "layernorm_affine", kernels::LAYERNORM_AFFINE),
            groupnorm: build(device, "groupnorm", kernels::GROUPNORM),
            conv1d: build(device, "conv1d", kernels::CONV1D),
            conv1d_f16: build(device, "conv1d_f16", kernels::CONV1D_F16),
            conv_transpose1d: build(device, "conv_transpose1d", kernels::CONV_TRANSPOSE1D),
            conv_transpose1d_f16: build(
                device,
                "conv_transpose1d_f16",
                kernels::CONV_TRANSPOSE1D_F16,
            ),
            leaky_relu: build(device, "leaky_relu", kernels::LEAKY_RELU),
            gelu_exact: build(device, "gelu_exact", kernels::GELU_EXACT),
            snake: build(device, "snake", kernels::SNAKE),
            adain: build(device, "adain", kernels::ADAIN),
            istft: build(device, "istft", kernels::ISTFT),
            transpose2d: build(device, "transpose2d", kernels::TRANSPOSE2D),
            nearest_upsample2x: build(device, "nearest_upsample2x", kernels::NEAREST_UPSAMPLE2X),
            spec_phase: build(device, "spec_phase", kernels::SPEC_PHASE),
            q4_k_matmul_tiled: build(
                device,
                "q4_k_matmul_tiled",
                kernels::Q4_K_DEQUANT_MATMUL_TILED,
            ),
            q6_k_matmul_tiled: build(
                device,
                "q6_k_matmul_tiled",
                kernels::Q6_K_DEQUANT_MATMUL_TILED,
            ),
            conv2d: build(device, "conv2d", kernels::CONV2D),
            conv2d_chf: build(device, "conv2d_chf", kernels::CONV2D_CHF),
            conv2d_chf_f16: build(device, "conv2d_chf_f16", kernels::CONV2D_CHF_F16),
            avg_pool2d: build(device, "avg_pool2d", kernels::AVG_POOL2D),
            avg_pool2d_half_chf: build(device, "avg_pool2d_half_chf", kernels::AVG_POOL2D_HALF_CHF),
            clamp: build(device, "clamp", kernels::CLAMP),
            quick_geglu: build(device, "quick_geglu", kernels::QUICK_GEGLU),
            rope_2d: build(device, "rope_2d", kernels::ROPE_2D),
            f16_matmul_batched: build(device, "f16_matmul_batched", kernels::F16_MATMUL_BATCHED),
            f16_matmul_batched_tiled: build(
                device,
                "f16_matmul_batched_tiled",
                kernels::F16_MATMUL_BATCHED_TILED,
            ),
            f16_matmul_batched_tiled_v2: build(
                device,
                "f16_matmul_batched_tiled_v2",
                kernels::F16_MATMUL_BATCHED_TILED_V2,
            ),
            f16_matmul_batched_tiled_v3: build(
                device,
                "f16_matmul_batched_tiled_v3",
                kernels::F16_MATMUL_BATCHED_TILED_V3,
            ),
            f16_matmul_batched_tiled_v4: build(
                device,
                "f16_matmul_batched_tiled_v4",
                kernels::F16_MATMUL_BATCHED_TILED_V4,
            ),
            f16_matmul_batched_tiled_v3_f16lds: None,
            bf16_matmul_batched_tiled_v3_f16lds: None,
            q4_k_matmul_f16lds: None,
            q6_k_matmul_f16lds: None,
            q4_k_matmul_wg256: build(
                device,
                "q4_k_matmul_wg256",
                kernels::Q4_K_DEQUANT_MATMUL_WG256,
            ),
            pos_embed_add: build(device, "pos_embed_add", kernels::POS_EMBED_ADD),
            vision_attention: build(device, "vision_attention", kernels::VISION_ATTENTION),
            vision_attention_flash: build(
                device,
                "vision_attention_flash",
                kernels::VISION_ATTENTION_FLASH,
            ),
            vision_attention_flash_q4: build(
                device,
                "vision_attention_flash_q4",
                kernels::VISION_ATTENTION_FLASH_Q4,
            ),
            vision_attention_flash_q8: build(
                device,
                "vision_attention_flash_q8",
                kernels::VISION_ATTENTION_FLASH_Q8,
            ),
            vision_attention_flash_q16: build(
                device,
                "vision_attention_flash_q16",
                kernels::VISION_ATTENTION_FLASH_Q16,
            ),
            vision_attention_flash_subgroup: None,
            vision_attention_flash_sub_t64: None,
            vision_attention_flash_sub_hpd: None,
            vision_attention_flash_sub_hpd_f16: None,
            vision_attention_flash_sub_hpd_f16_q16: None,
            vision_attention_flash_hpd_f16: None,
            transpose_phd_to_hpd: build(
                device,
                "transpose_phd_to_hpd",
                kernels::TRANSPOSE_PHD_TO_HPD,
            ),
            transpose_hpd_to_phd: build(
                device,
                "transpose_hpd_to_phd",
                kernels::TRANSPOSE_HPD_TO_PHD,
            ),
            half_residual_add: build(device, "half_residual_add", kernels::HALF_RESIDUAL_ADD),
            silu: build(device, "silu", kernels::SILU),
            glu_split: build(device, "glu_split", kernels::GLU_SPLIT),
            depthwise_conv1d: build(device, "depthwise_conv1d", kernels::DEPTHWISE_CONV1D),
            block_local_attention: build(
                device,
                "block_local_attention",
                kernels::BLOCK_LOCAL_ATTENTION,
            ),
            bf16_matmul: build(device, "bf16_matmul", kernels::BF16_MATMUL),
            bf16_matmul_batched: build(device, "bf16_matmul_batched", kernels::BF16_MATMUL_BATCHED),
            bf16_matmul_batched_tiled: build(
                device,
                "bf16_matmul_batched_tiled",
                kernels::BF16_MATMUL_BATCHED_TILED,
            ),
            bf16_matmul_batched_tiled_v3: build(
                device,
                "bf16_matmul_batched_tiled_v3",
                kernels::BF16_MATMUL_BATCHED_TILED_V3,
            ),
            bf16_matmul_batched_tiled_v2: build(
                device,
                "bf16_matmul_batched_tiled_v2",
                kernels::BF16_MATMUL_BATCHED_TILED_V2,
            ),
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
