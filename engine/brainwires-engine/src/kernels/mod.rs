//! WGSL kernels — included as static strings via `include_str!`.
//!
//! Each kernel is paired with a typed Rust dispatcher in [`crate::backend::matmul`]
//! that creates the pipeline, uploads inputs, dispatches, and reads back results.

pub const ATTENTION: &str = include_str!("wgsl/attention.wgsl");
pub const AVG_POOL2D: &str = include_str!("wgsl/avg_pool2d.wgsl");
pub const CLAMP: &str = include_str!("wgsl/clamp.wgsl");
pub const ADAIN: &str = include_str!("wgsl/adain.wgsl");
pub const CONV1D: &str = include_str!("wgsl/conv1d.wgsl");
pub const CONV_TRANSPOSE1D: &str = include_str!("wgsl/conv_transpose1d.wgsl");
pub const CONV2D: &str = include_str!("wgsl/conv2d.wgsl");
pub const ISTFT: &str = include_str!("wgsl/istft.wgsl");
pub const LAYERNORM_AFFINE: &str = include_str!("wgsl/layernorm_affine.wgsl");
pub const LEAKY_RELU: &str = include_str!("wgsl/leaky_relu.wgsl");
pub const SNAKE: &str = include_str!("wgsl/snake.wgsl");
pub const F16_MATMUL: &str = include_str!("wgsl/f16_matmul.wgsl");
pub const F16_MATMUL_BATCHED: &str = include_str!("wgsl/f16_matmul_batched.wgsl");
pub const F16_MATMUL_BATCHED_TILED: &str = include_str!("wgsl/f16_matmul_batched_tiled.wgsl");
pub const F16_MATMUL_BATCHED_TILED_V2: &str = include_str!("wgsl/f16_matmul_batched_tiled_v2.wgsl");
pub const F16_MATMUL_BATCHED_TILED_V3: &str = include_str!("wgsl/f16_matmul_batched_tiled_v3.wgsl");
pub const F16_MATMUL_BATCHED_TILED_V4: &str = include_str!("wgsl/f16_matmul_batched_tiled_v4.wgsl");
pub const F16_MATMUL_BATCHED_TILED_V3_F16LDS: &str =
    include_str!("wgsl/f16_matmul_batched_tiled_v3_f16lds.wgsl");
pub const QUICK_GEGLU: &str = include_str!("wgsl/quick_geglu.wgsl");
pub const POS_EMBED_ADD: &str = include_str!("wgsl/pos_embed_add.wgsl");
pub const ROPE_2D: &str = include_str!("wgsl/rope_2d.wgsl");
pub const VISION_ATTENTION: &str = include_str!("wgsl/vision_attention.wgsl");
pub const VISION_ATTENTION_FLASH: &str = include_str!("wgsl/vision_attention_flash.wgsl");
pub const VISION_ATTENTION_FLASH_Q4: &str = include_str!("wgsl/vision_attention_flash_q4.wgsl");
pub const VISION_ATTENTION_FLASH_Q8: &str = include_str!("wgsl/vision_attention_flash_q8.wgsl");
pub const VISION_ATTENTION_FLASH_Q16: &str = include_str!("wgsl/vision_attention_flash_q16.wgsl");
pub const VISION_ATTENTION_FLASH_SUBGROUP: &str =
    include_str!("wgsl/vision_attention_flash_subgroup.wgsl");
pub const VISION_ATTENTION_FLASH_SUB_T64: &str =
    include_str!("wgsl/vision_attention_flash_sub_t64.wgsl");
pub const VISION_ATTENTION_FLASH_SUB_HPD: &str =
    include_str!("wgsl/vision_attention_flash_sub_hpd.wgsl");
pub const VISION_ATTENTION_FLASH_SUB_HPD_F16: &str =
    include_str!("wgsl/vision_attention_flash_sub_hpd_f16.wgsl");
pub const VISION_ATTENTION_FLASH_SUB_HPD_F16_Q16: &str =
    include_str!("wgsl/vision_attention_flash_sub_hpd_f16_q16.wgsl");
pub const VISION_ATTENTION_FLASH_HPD_F16: &str =
    include_str!("wgsl/vision_attention_flash_hpd_f16.wgsl");
pub const TRANSPOSE_PHD_TO_HPD: &str = include_str!("wgsl/transpose_phd_to_hpd.wgsl");
pub const TRANSPOSE_HPD_TO_PHD: &str = include_str!("wgsl/transpose_hpd_to_phd.wgsl");
pub const HALF_RESIDUAL_ADD: &str = include_str!("wgsl/half_residual_add.wgsl");
pub const SILU: &str = include_str!("wgsl/silu.wgsl");
pub const GLU_SPLIT: &str = include_str!("wgsl/glu_split.wgsl");
pub const DEPTHWISE_CONV1D: &str = include_str!("wgsl/depthwise_conv1d.wgsl");
pub const BLOCK_LOCAL_ATTENTION: &str = include_str!("wgsl/block_local_attention.wgsl");
pub const BF16_MATMUL: &str = include_str!("wgsl/bf16_matmul.wgsl");
pub const BF16_MATMUL_BATCHED: &str = include_str!("wgsl/bf16_matmul_batched.wgsl");
pub const BF16_MATMUL_BATCHED_TILED: &str = include_str!("wgsl/bf16_matmul_batched_tiled.wgsl");
pub const BF16_MATMUL_BATCHED_TILED_V3: &str =
    include_str!("wgsl/bf16_matmul_batched_tiled_v3.wgsl");
pub const BF16_MATMUL_BATCHED_TILED_V3_F16LDS: &str =
    include_str!("wgsl/bf16_matmul_batched_tiled_v3_f16lds.wgsl");
pub const BF16_MATMUL_BATCHED_TILED_V2: &str =
    include_str!("wgsl/bf16_matmul_batched_tiled_v2.wgsl");
pub const SCALE_PER_INNER_DIM: &str = include_str!("wgsl/scale_per_inner_dim.wgsl");
pub const ADD_BIAS_BATCHED: &str = include_str!("wgsl/add_bias_batched.wgsl");
pub const GEGLU: &str = include_str!("wgsl/geglu.wgsl");
pub const Q4_K_DEQUANT_MATMUL: &str = include_str!("wgsl/q4_k_dequant_matmul.wgsl");
pub const Q4_K_DEQUANT_MATMUL_TILED: &str = include_str!("wgsl/q4_k_dequant_matmul_tiled.wgsl");
pub const Q4_K_DEQUANT_MATMUL_F16LDS: &str = include_str!("wgsl/q4_k_dequant_matmul_f16lds.wgsl");
pub const Q4_K_DEQUANT_MATMUL_WG256: &str = include_str!("wgsl/q4_k_dequant_matmul_wg256.wgsl");
pub const Q6_K_DEQUANT_MATMUL: &str = include_str!("wgsl/q6_k_dequant_matmul.wgsl");
pub const Q6_K_DEQUANT_MATMUL_TILED: &str = include_str!("wgsl/q6_k_dequant_matmul_tiled.wgsl");
pub const Q6_K_DEQUANT_MATMUL_F16LDS: &str = include_str!("wgsl/q6_k_dequant_matmul_f16lds.wgsl");
pub const RESIDUAL_ADD: &str = include_str!("wgsl/residual_add.wgsl");
pub const RMSNORM: &str = include_str!("wgsl/rmsnorm.wgsl");
pub const RMSNORM_PER_ROW: &str = include_str!("wgsl/rmsnorm_per_row.wgsl");
pub const ROPE_NEOX: &str = include_str!("wgsl/rope_neox.wgsl");
pub const SCALE: &str = include_str!("wgsl/scale.wgsl");
pub const SOFTCAP: &str = include_str!("wgsl/softcap.wgsl");

// --- Training kernels (M0 backward pass) ---
pub const CROSS_ENTROPY_BACKWARD: &str = include_str!("wgsl/cross_entropy_backward.wgsl");
pub const MATMUL_Q4_K_BACKWARD_INPUT: &str = include_str!("wgsl/matmul_q4_k_backward_input.wgsl");
pub const MATMUL_Q6_K_BACKWARD_INPUT: &str = include_str!("wgsl/matmul_q6_k_backward_input.wgsl");
pub const RMSNORM_BACKWARD: &str = include_str!("wgsl/rmsnorm_backward.wgsl");
pub const RMSNORM_PER_ROW_BACKWARD: &str = include_str!("wgsl/rmsnorm_per_row_backward.wgsl");
pub const GEGLU_BACKWARD: &str = include_str!("wgsl/geglu_backward.wgsl");
pub const ROPE_NEOX_BACKWARD: &str = include_str!("wgsl/rope_neox_backward.wgsl");
pub const ATTENTION_BACKWARD_DQ: &str = include_str!("wgsl/attention_backward_dq.wgsl");
pub const ATTENTION_BACKWARD_DKV: &str = include_str!("wgsl/attention_backward_dkv.wgsl");
pub const ATTENTION_PROBS: &str = include_str!("wgsl/attention_probs.wgsl");
pub const LORA_MATMUL_ROW: &str = include_str!("wgsl/lora_matmul_row.wgsl");
pub const LORA_MATMUL_COL: &str = include_str!("wgsl/lora_matmul_col.wgsl");
pub const LORA_OUTER_ADD: &str = include_str!("wgsl/lora_outer_add.wgsl");
pub const LORA_EMBED_COL_READ: &str = include_str!("wgsl/lora_embed_col_read.wgsl");
pub const LORA_EMBED_COL_SCATTER_ADD: &str = include_str!("wgsl/lora_embed_col_scatter_add.wgsl");
pub const LORA_MATMUL_FUSED: &str = include_str!("wgsl/lora_matmul_fused.wgsl");
pub const LORA_MATMUL_FUSED_F16B: &str = include_str!("wgsl/lora_matmul_fused_f16b.wgsl");
pub const ADAM_STEP: &str = include_str!("wgsl/adam_step.wgsl");
pub const SUM_OF_SQUARES: &str = include_str!("wgsl/sum_of_squares.wgsl");
