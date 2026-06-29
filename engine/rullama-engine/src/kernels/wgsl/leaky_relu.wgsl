// Elementwise LeakyReLU, in-place. y[i] = y[i] < 0 ? y[i]*slope : y[i].
// Kokoro acoustic blocks use slope 0.2; the generator uses 0.1 / 0.01.

struct Params { n: u32, slope: f32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if (i >= params.n) { return; }
    let v = y[i];
    if (v < 0.0) { y[i] = v * params.slope; }
}
