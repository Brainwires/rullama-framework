// NeoX-style RoPE applied in-place to x[head_dim, n_heads].
//
// Pair layout: (x[i], x[i + rope_dims/2]) for i in 0..rope_dims/2. Indices
// [rope_dims, head_dim) are untouched (allows partial rotation when rope_dims < head_dim).
//
// `has_factors=1` means `factors` is a real array of length rope_dims/2; angles are
// divided by `factors[i]`. This is how Gemma 4's global layers achieve proportional
// RoPE — `rope_freqs.weight` has 1.0 on dimensions to rotate and a huge sentinel
// (≈1e30) on the rest, effectively zeroing the angle there.

struct Params {
    head_dim:    u32,
    n_heads:     u32,
    rope_dims:   u32,
    pos:         u32,
    base:        f32,
    has_factors: u32,
    _pad0:       u32,
    _pad1:       u32,
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read_write> x:       array<f32>;
@group(0) @binding(2) var<storage, read>       factors: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half: u32 = params.rope_dims / 2u;
    let total = params.n_heads * half;
    let id = gid.x;
    if (id >= total) { return; }

    let h = id / half;
    let i = id % half;

    let head_off = h * params.head_dim;
    let exp_v = -2.0 * f32(i) / f32(params.rope_dims);
    var theta = f32(params.pos) * pow(params.base, exp_v);
    if (params.has_factors != 0u) {
        theta = theta / factors[i];
    }
    let c = cos(theta);
    let s = sin(theta);

    let a = x[head_off + i];
    let b = x[head_off + i + half];
    x[head_off + i]        = a * c - b * s;
    x[head_off + i + half] = a * s + b * c;
}
