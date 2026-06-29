// Convert one f32 KV row (n_kv_heads*head_dim contiguous f32 values) into packed
// f16 — two values per u32 via the CORE built-in `pack2x16float` (no `enable
// f16;` / SHADER_F16 dependency) — and write it into the f16 KV cache at a
// per-token word offset. The f32 producers (rope / rmsnorm) stay untouched; this
// is the only place f16 packing happens on the write path, so the f32 KV path is
// byte-for-byte unchanged.
//
// `pack2x16float(vec2(a,b))` puts `a` in the low 16 bits and `b` in the high 16
// bits, matching the Rust helper `pack_f32_to_f16_pairs` (lora.rs) and the
// `unpack2x16float` reads in attention_f16kv.wgsl.

struct Params {
    dst_word_off: u32,  // starting u32 index in dst = kv_lens * n_pairs
    n_pairs:      u32,  // (n_kv_heads * head_dim) / 2
    _pad0:        u32,
    _pad1:        u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       src:    array<f32>;  // one row, n_kv_heads*head_dim elems
@group(0) @binding(2) var<storage, read_write> dst:    array<u32>;  // the f16 KV cache (packed)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.n_pairs) { return; }
    let a = src[2u * p];
    let b = src[2u * p + 1u];
    dst[params.dst_word_off + p] = pack2x16float(vec2<f32>(a, b));
}
