// Channel-FIRST 2D convolution for the StyleTTS2 style encoder. (The vision `conv2d`
// is channel-last, f16, no bias, no groups — doesn't fit the encoder.)
//   input  x:    f32 [in_C, in_H, in_W]
//   weight w:    f32 [out_C, in_C/groups, kH, kW]  (PyTorch layout)
//   bias:        f32 [out_C]  (used iff has_bias != 0)
//   output y:    f32 [out_C, out_H, out_W]
// Supports stride / pad / groups (depthwise = groups == in_C). One thread per output
// element; 2D workgroup grid (idx = gid.y*num_workgroups.x*64 + gid.x) for large outputs.

struct Params {
    in_c: u32, in_h: u32, in_w: u32,
    out_c: u32, out_h: u32, out_w: u32,
    kh: u32, kw: u32, sh: u32, sw: u32, ph: u32, pw: u32,
    groups: u32, has_bias: u32, _p0: u32, _p1: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       w:      array<f32>;
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read>       bias:   array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if (idx >= params.out_c * params.out_h * params.out_w) { return; }

    let hw = params.out_h * params.out_w;
    let oc = idx / hw;
    let rem = idx % hw;
    let oy = rem / params.out_w;
    let ox = rem % params.out_w;

    let icpg = params.in_c / params.groups;
    let ocpg = params.out_c / params.groups;
    let g = oc / ocpg;

    var acc: f32 = 0.0;
    if (params.has_bias != 0u) { acc = bias[oc]; }

    for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
        let ci = g * icpg + icg;
        let wbase = ((oc * icpg + icg) * params.kh) * params.kw;
        let xbase = ci * params.in_h * params.in_w;
        for (var ky: u32 = 0u; ky < params.kh; ky = ky + 1u) {
            let iy = i32(oy * params.sh + ky) - i32(params.ph);
            if (iy < 0 || iy >= i32(params.in_h)) { continue; }
            for (var kx: u32 = 0u; kx < params.kw; kx = kx + 1u) {
                let ix = i32(ox * params.sw + kx) - i32(params.pw);
                if (ix < 0 || ix >= i32(params.in_w)) { continue; }
                acc = acc + x[xbase + u32(iy) * params.in_w + u32(ix)] * w[wbase + ky * params.kw + kx];
            }
        }
    }
    y[idx] = acc;
}
