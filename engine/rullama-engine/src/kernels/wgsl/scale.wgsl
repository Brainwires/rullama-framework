// Vector scale: x[i] *= s. In-place into `x`.
//
// `offset` lets the dispatcher chunk a single logical scale across
// multiple `dispatch_workgroups` calls so we don't exceed wgpu's
// per-dimension cap (65_535 groups × 64 threads = 4_194_240 max
// elements per single dispatch). The dispatcher fires one dispatch
// per chunk and bumps `offset` so each thread still computes a unique
// linear index.

struct Params {
    n:      u32,
    s:      f32,
    offset: u32,
    _p1:    u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> x:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * nwg.x * 64u + gid.x + params.offset;
    if (i >= params.n) { return; }
    x[i] = x[i] * params.s;
}
