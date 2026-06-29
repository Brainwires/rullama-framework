//! Validate the real Z-Image Qwen3 text-encoder weights against the parsed
//! config: every per-layer projection's shape must match (hidden, q/kv dims,
//! intermediate), proving the sharded safetensors loader + config agree with
//! ground truth before we wire the encoder forward (IM1).
//!
//! Usage:
//!   cargo run -p brainwires-engine --example imagegen_qwen3_inventory -- \
//!       weights/Z-Image-Turbo/text_encoder
//!
//! (Default path is weights/Z-Image-Turbo/text_encoder.)

use brainwires_engine::imagegen::{Qwen3Config, ShardedSafetensors};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/Z-Image-Turbo/text_encoder".to_string());

    let cfg_path = format!("{dir}/config.json");
    let cfg = Qwen3Config::parse(&std::fs::read(&cfg_path).expect("read config.json"))
        .expect("parse Qwen3 config");
    println!(
        "Qwen3: hidden={} layers={} q_dim={} kv_dim={} inter={} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.q_dim(),
        cfg.kv_dim(),
        cfg.intermediate_size,
        cfg.vocab_size
    );

    let st = ShardedSafetensors::open_dir(&dir, "model.safetensors.index.json")
        .expect("open sharded text_encoder");
    println!(
        "loaded {} tensors across {} shards",
        st.names().count(),
        st.index().shards().len()
    );

    let h = cfg.hidden_size as usize;
    let q = cfg.q_dim() as usize;
    let kv = cfg.kv_dim() as usize;
    let inter = cfg.intermediate_size as usize;
    let hd = cfg.head_dim as usize;

    // embed_tokens: [vocab, hidden]
    check(
        &st,
        "model.embed_tokens.weight",
        &[cfg.vocab_size as usize, h],
    );
    check(&st, "model.norm.weight", &[h]);

    // Every layer's projection/norm shapes.
    let mut checked = 0usize;
    for i in 0..cfg.num_hidden_layers as usize {
        let p = format!("model.layers.{i}");
        check(&st, &format!("{p}.input_layernorm.weight"), &[h]);
        check(&st, &format!("{p}.post_attention_layernorm.weight"), &[h]);
        check(&st, &format!("{p}.self_attn.q_proj.weight"), &[q, h]);
        check(&st, &format!("{p}.self_attn.k_proj.weight"), &[kv, h]);
        check(&st, &format!("{p}.self_attn.v_proj.weight"), &[kv, h]);
        check(&st, &format!("{p}.self_attn.o_proj.weight"), &[h, q]);
        // Qwen3 QK-norm is per-head RMSNorm over head_dim.
        check(&st, &format!("{p}.self_attn.q_norm.weight"), &[hd]);
        check(&st, &format!("{p}.self_attn.k_norm.weight"), &[hd]);
        // SwiGLU MLP.
        check(&st, &format!("{p}.mlp.gate_proj.weight"), &[inter, h]);
        check(&st, &format!("{p}.mlp.up_proj.weight"), &[inter, h]);
        check(&st, &format!("{p}.mlp.down_proj.weight"), &[h, inter]);
        checked += 1;
    }

    // Spot-check we can actually range-read + dequantize a tensor.
    let norm = st.tensor_f32("model.norm.weight").expect("read final norm");
    let mean = norm.iter().sum::<f32>() / norm.len() as f32;
    println!(
        "final norm: {} elems, dtype {:?}, mean {mean:.4}",
        norm.len(),
        st.dtype("model.norm.weight").unwrap()
    );

    println!("\nOK — all shapes match config across {checked} layers.");
}

fn check(st: &ShardedSafetensors, name: &str, expect: &[usize]) {
    match st.shape(name) {
        Ok(shape) if shape == expect => {}
        Ok(shape) => panic!("{name}: shape {shape:?} != expected {expect:?}"),
        Err(e) => panic!("{name}: {e}"),
    }
}
