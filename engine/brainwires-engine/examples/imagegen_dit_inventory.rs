//! Validate the real Z-Image DiT (transformer) structure against the parsed
//! TransformerConfig: the 3-shard load + every global/per-layer tensor shape,
//! the foundation for the DiT forward (IM2). Structure (S3-DiT single-stream):
//!   t_embedder (sinusoidal→MLP) · cap_embedder (Qwen3→dim) · x_embedder (patch)
//!   · noise_refiner×2 · context_refiner×2 · layers×30 (adaLN, QK-normed attn,
//!   SwiGLU) · final_layer (adaLN + unpatch linear)
//!
//! Usage:
//!   cargo run -p brainwires-engine --example imagegen_dit_inventory -- \
//!       weights/Z-Image-Turbo/transformer

use brainwires_engine::imagegen::{ShardedSafetensors, TransformerConfig};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/Z-Image-Turbo/transformer".to_string());

    let cfg =
        TransformerConfig::parse(&std::fs::read(format!("{dir}/config.json")).expect("config"))
            .expect("parse DiT config");
    let dim = cfg.dim as usize;
    let hd = cfg.head_dim() as usize;
    let inter = 10240usize; // SwiGLU width (from weights; not in config.json)
    let patch_in = (cfg.in_channels * cfg.patch_size() * cfg.patch_size()) as usize; // 16*2*2=64
    let temb = 256usize; // timestep-embedding dim
    println!(
        "DiT: dim={dim} heads={} head_dim={hd} layers={} refiners={} cap_feat={} patch_in={patch_in}",
        cfg.n_heads, cfg.n_layers, cfg.n_refiner_layers, cfg.cap_feat_dim
    );

    let st = ShardedSafetensors::open_dir(&dir, "diffusion_pytorch_model.safetensors.index.json")
        .expect("open DiT shards");
    println!(
        "loaded {} tensors across {} shards",
        st.names().count(),
        st.index().shards().len()
    );

    // global embedders / final layer
    check(&st, "t_embedder.mlp.0.weight", &[1024, temb]);
    check(&st, "t_embedder.mlp.2.weight", &[temb, 1024]);
    check(
        &st,
        "cap_embedder.1.weight",
        &[dim, cfg.cap_feat_dim as usize],
    );
    check(&st, "all_x_embedder.2-1.weight", &[dim, patch_in]);
    check(&st, "all_final_layer.2-1.linear.weight", &[patch_in, dim]);
    check(
        &st,
        "all_final_layer.2-1.adaLN_modulation.1.weight",
        &[dim, temb],
    );

    // a transformer block's tensors. `has_adaln` is true for the timestep-
    // modulated blocks (main layers + noise_refiner); the context_refiner
    // refines the text stream and carries no adaLN modulation.
    let check_block = |p: &str, has_adaln: bool| {
        for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
            check(&st, &format!("{p}.attention.{proj}.weight"), &[dim, dim]);
        }
        check(&st, &format!("{p}.attention.norm_q.weight"), &[hd]);
        check(&st, &format!("{p}.attention.norm_k.weight"), &[hd]);
        check(&st, &format!("{p}.feed_forward.w1.weight"), &[inter, dim]);
        check(&st, &format!("{p}.feed_forward.w3.weight"), &[inter, dim]);
        check(&st, &format!("{p}.feed_forward.w2.weight"), &[dim, inter]);
        check(&st, &format!("{p}.attention_norm1.weight"), &[dim]);
        if has_adaln {
            // adaLN modulation: 4·dim params from the 256-d timestep embedding.
            check(
                &st,
                &format!("{p}.adaLN_modulation.0.weight"),
                &[4 * dim, temb],
            );
        } else {
            assert!(
                !st.has(&format!("{p}.adaLN_modulation.0.weight")),
                "{p} unexpectedly has adaLN_modulation"
            );
        }
    };
    for i in 0..cfg.n_layers as usize {
        check_block(&format!("layers.{i}"), true);
    }
    for i in 0..cfg.n_refiner_layers as usize {
        check_block(&format!("noise_refiner.{i}"), true);
        check_block(&format!("context_refiner.{i}"), false);
    }

    // spot-check range-read + dequant
    let w = st
        .tensor_f32("layers.0.attention_norm1.weight")
        .expect("read");
    println!(
        "layers.0.attention_norm1: {} elems dtype {:?} mean {:.4}",
        w.len(),
        st.dtype("layers.0.attention_norm1.weight").unwrap(),
        w.iter().sum::<f32>() / w.len() as f32
    );

    println!(
        "\nOK — DiT structure matches config across {} main + {} refiner blocks.",
        cfg.n_layers,
        2 * cfg.n_refiner_layers
    );
}

fn check(st: &ShardedSafetensors, name: &str, expect: &[usize]) {
    match st.shape(name) {
        Ok(s) if s == expect => {}
        Ok(s) => panic!("{name}: shape {s:?} != expected {expect:?}"),
        Err(e) => panic!("{name}: {e}"),
    }
}
