// Half-residual add: x[i] = x[i] + 0.5 * y[i]. In-place into `x`.
// Used by Conformer's two FFW sandwiches (residual + 0.5 * post_ffw_norm).

struct Params {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> x:      array<f32>;
@group(0) @binding(2) var<storage, read>       y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    x[i] = x[i] + 0.5 * y[i];
}
