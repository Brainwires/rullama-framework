// Nearest-neighbour ×2 upsample along time, channel-major [C, T] -> [C, 2T].
// out[c, to] = in[c, to/2]. Used by the AdainResBlk1d shortcut on upsample blocks.

struct Params { c: u32, t: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let tout = params.t * 2u;
    if (idx >= params.c * tout) { return; }
    let ch = idx / tout;
    let to = idx % tout;
    y[idx] = x[ch * params.t + (to / 2u)];
}
