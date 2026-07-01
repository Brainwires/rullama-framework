// GLU split: y[t, d] = x[t, d] * sigmoid(x[t, d_inner + d]).
//
// `x` is [T, 2*D] flat (channel-LAST). The first D channels are the data half,
// the second D are the gate half. Used in Conformer LightConv after conv_pw1.
//
// Output `y` is [T, D] (half the channel count). Element-major. Caller dispatches
// `T * D / 64` workgroups.

struct Params {
    seq:    u32,
    inner:  u32,    // D — half of x's channel count
    _p0:    u32,
    _p1:    u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total: u32 = params.seq * params.inner;
    let idx: u32 = gid.x;
    if (idx >= total) { return; }
    let t: u32 = idx / params.inner;
    let d: u32 = idx - t * params.inner;
    let two_inner: u32 = params.inner * 2u;
    let data: f32 = x[t * two_inner + d];
    let g:    f32 = x[t * two_inner + params.inner + d];
    let sig:  f32 = 1.0 / (1.0 + exp(-g));
    y[idx] = data * sig;
}
