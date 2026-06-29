// In-place per-inner-dim scale: x[i] *= s[i % inner_dim].
//
// Used by the audio Conformer attention to apply per-dim Q scaling
// (`q[t, h, d] *= q_scale_base * per_dim_scale[d]`). The scale is
// pre-multiplied with q_scale_base on the host side so this kernel
// only does the broadcast multiply.

struct Params {
    n:         u32,
    inner_dim: u32,
    _p0:       u32,
    _p1:       u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> x:      array<f32>;
@group(0) @binding(2) var<storage, read>       s:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let d = i % params.inner_dim;
    x[i] = x[i] * s[d];
}
