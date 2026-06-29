// DiT tanh-gated residual add: x[t,c] += tanh(gate[c]) * branch[t,c].
// x/branch are [seq, dim] flat; gate is length-dim (from the adaLN modulation),
// broadcast across tokens. Matches the Z-Image DiT block's gated residual.
// Reference oracle: reference::imagegen::gated_residual_add.

struct Params {
    dim:   u32,
    total: u32, // seq * dim
    _p0:   u32,
    _p1:   u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> x:      array<f32>;
@group(0) @binding(2) var<storage, read>       gate:   array<f32>;
@group(0) @binding(3) var<storage, read>       branch: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.total) { return; }
    let c = i % params.dim;
    x[i] = x[i] + tanh(gate[c]) * branch[i];
}
