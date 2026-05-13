// SiLU activation in-place: x[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i])).
// Used by the Conformer FFW (between up and down) and LightConv post-norm.

struct Params {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> x:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let v = x[i];
    x[i] = v / (1.0 + exp(-v));
}
