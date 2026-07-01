// Interleaved (GPT-J style) RoPE for the Z-Image DiT, applied with precomputed
// per-token cos/sin (the multi-axis frequency table is built host-side).
//
// x is [seq, heads, head_dim] flat; cos/sin are [seq, half] (half = head_dim/2),
// shared across heads. For each (token t, head h) and pair i in 0..half:
//   x1 = x[.., 2i]   x2 = x[.., 2i+1]
//   out[2i]   = x1*cos[t,i] - x2*sin[t,i]
//   out[2i+1] = x1*sin[t,i] + x2*cos[t,i]
//
// Distinct from rope_neox.wgsl (which rotates the two *halves* of head_dim).
// Reference oracle: reference::imagegen::rope_interleaved.

struct Params {
    seq:   u32,
    heads: u32,
    hd:    u32,   // head_dim (even)
    half:  u32,   // hd / 2
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> x:      array<f32>;
@group(0) @binding(2) var<storage, read>       cos_t:  array<f32>;
@group(0) @binding(3) var<storage, read>       sin_t:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let total = params.seq * params.heads * params.half; // one thread per pair
    if (p >= total) { return; }

    let i = p % params.half;                 // pair index within the head
    let th = (p / params.half) % params.heads;
    let t = p / (params.half * params.heads); // token

    let base = (t * params.heads + th) * params.hd;
    let c = cos_t[t * params.half + i];
    let s = sin_t[t * params.half + i];

    let x1 = x[base + 2u * i];
    let x2 = x[base + 2u * i + 1u];
    x[base + 2u * i]      = x1 * c - x2 * s;
    x[base + 2u * i + 1u] = x1 * s + x2 * c;
}
