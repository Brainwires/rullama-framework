// bf16 weight matmul: y[j] = Σ_i x[i] * W[j, i]
//
// Mirrors f16_matmul.wgsl, but the packed u32 holds two BF16 values (each is
// just the upper 16 bits of an F32). We unpack as `f32(bits << 16)` —
// reinterpret the bit pattern after left-shifting into the F32 sign/exp/mantissa
// layout. NaN/Inf handling matches IEEE 754 semantics through bitcast.
//
// Used by the audio Conformer tower: every audio block linear (attn_q/k/v/out,
// ffn_up/down, conv_pw1/2) is BF16 in `gemma4:e2b`'s GGUF.

struct Params {
    k: u32,
    n: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;   // bf16 pairs packed per u32
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j: u32 = gid.x;
    if (j >= params.n) { return; }

    let half_k: u32 = params.k / 2u;
    let row_start: u32 = j * half_k;

    var acc: f32 = 0.0;
    for (var p: u32 = 0u; p < half_k; p = p + 1u) {
        let packed: u32 = weight[row_start + p];
        // Low 16 bits = first element, high 16 bits = second element.
        let lo: f32 = bf16_to_f32(packed & 0x0000FFFFu);
        let hi: f32 = bf16_to_f32(packed >> 16u);
        acc = acc + x[p * 2u] * lo + x[p * 2u + 1u] * hi;
    }
    y[j] = acc;
}
