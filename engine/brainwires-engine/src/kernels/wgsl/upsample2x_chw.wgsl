// 2D nearest-neighbor 2× upsample, channel-first: [C,H,W] → [C,2H,2W].
//   out[c, y, x] = in[c, y/2, x/2]
// (The audio nearest_upsample2x doubles only one axis; the VAE needs both.)
// One thread per output element. Oracle: reference::imagegen::upsample2x_chw.

struct Params {
    c:  u32,
    h:  u32,
    w:  u32,
    _p: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h2 = params.h * 2u;
    let w2 = params.w * 2u;
    let total = params.c * h2 * w2;
    let o = gid.x;
    if (o >= total) { return; }

    let ox = o % w2;
    let t1 = o / w2;
    let oy = t1 % h2;
    let c = t1 / h2;

    let iy = oy / 2u;
    let ix = ox / 2u;
    y[o] = x[(c * params.h + iy) * params.w + ix];
}
