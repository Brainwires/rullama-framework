// Snake1D activation, channel-major [C, T]: y = x + (1/alpha_c) * sin(alpha_c * x)^2.
// Per-channel learnable alpha[C]. Used by the ISTFTNet generator resblocks.

struct Params { c: u32, t: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       alpha:  array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.c * params.t) { return; }
    let ch = idx / params.t;
    let a = alpha[ch];
    let v = x[idx];
    let s = sin(a * v);
    y[idx] = v + (1.0 / a) * s * s;
}
