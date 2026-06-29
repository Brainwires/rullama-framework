// Channel-FIRST 2×2 average pool (StyleTTS2 `DownSample('half')`). Input [C, in_H, in_W]
// (in_H even), output [C, in_H/2, ceil(in_W/2)]. For odd width, the missing last column
// repeats the last real column (matches torch's `cat([x, x[..., -1:]])` before avg_pool2d).
// One thread per output element; 2D workgroup grid for large outputs.

struct Params { c: u32, in_h: u32, in_w: u32, out_h: u32, out_w: u32, _p0: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if (idx >= params.c * params.out_h * params.out_w) { return; }

    let hw = params.out_h * params.out_w;
    let c = idx / hw;
    let rem = idx % hw;
    let oy = rem / params.out_w;
    let ox = rem % params.out_w;

    let base = c * params.in_h * params.in_w;
    let y0 = 2u * oy;
    let y1 = y0 + 1u;
    let x0 = 2u * ox;
    var x1 = x0 + 1u;
    if (x1 >= params.in_w) { x1 = params.in_w - 1u; } // repeat last column (odd width)

    let s = x[base + y0 * params.in_w + x0] + x[base + y0 * params.in_w + x1]
          + x[base + y1 * params.in_w + x0] + x[base + y1 * params.in_w + x1];
    y[idx] = s * 0.25;
}
