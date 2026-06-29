// NeoX RoPE backward — inverse rotation applied in-place to `dx`.
//
// Forward rotation by θ:
//   x[i]        ← x[i]·cos θ - x[i + half]·sin θ
//   x[i + half] ← x[i]·sin θ + x[i + half]·cos θ
//
// Backward by the same θ rotates by -θ (rotations are orthogonal):
//   dx[i]        ← dx[i]·cos θ + dx[i + half]·sin θ
//   dx[i + half] ← -dx[i]·sin θ + dx[i + half]·cos θ
//
// Identical layout / `Params` shape to `rope_neox.wgsl`; the caller
// reuses the same `factors` buffer and `pos` value as the forward.

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
    // Inverse rotation: cos symmetric, sin flips sign.
    x[head_off + i]        =  a * c + b * s;
    x[head_off + i + half] = -a * s + b * c;
}
