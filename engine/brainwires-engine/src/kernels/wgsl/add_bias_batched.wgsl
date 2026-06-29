// In-place bias add: y[b, j] += bias[j], y is [batch, n] flat.
//
// Used by the audio multimodal projector's FC linear (mm.a.fc has both
// weight and bias). Generalises naturally to any batched linear that
// needs a per-output-dim bias.

struct Params {
    n:     u32,
    batch: u32,
    _p0:   u32,
    _p1:   u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> y:     array<f32>;
@group(0) @binding(2) var<storage, read>       bias:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.batch;
    if (i >= total) { return; }
    let j = i % params.n;
    y[i] = y[i] + bias[j];
}
