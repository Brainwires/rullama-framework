// Channel-first 2D convolution with f32 weights + bias, stride 1, zero-pad.
// Matches the image VAE's diffusers conv layout (reference::reference vae oracle)
// exactly, so GPU-vs-CPU parity is tight (no f16 rounding, no layout transpose).
//
//   input:  f32 [in_C, in_H, in_W]   (channel-first, per-channel contiguous)
//   weight: f32 [out_C, in_C, kH, kW]
//   bias:   f32 [out_C]
//   output: f32 [out_C, out_H, out_W]  (channel-first)
//
// out_H/out_W = in + 2*pad - k + 1 (stride 1) — caller passes them. One thread
// per output element; OOB input reads (padding) contribute 0.

struct Params {
    in_C:  u32,
    in_H:  u32,
    in_W:  u32,
    out_C: u32,
    out_H: u32,
    out_W: u32,
    k:     u32,   // square kernel (kH == kW)
    pad:   u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read>       bias:   array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let total: u32 = params.out_C * params.out_H * params.out_W;
    // 2D workgroup grid (wg_grid): reconstruct the linear index so VAE convs at
    // ≥256px (>65535×64 elements) don't overflow the 1-D dispatch cap.
    let o: u32 = gid.y * nwg.x * 64u + gid.x;
    if (o >= total) { return; }

    // output channel-first: o = (co * out_H + oy) * out_W + ox
    let ox: u32 = o % params.out_W;
    let t1: u32 = o / params.out_W;
    let oy: u32 = t1 % params.out_H;
    let co: u32 = t1 / params.out_H;

    var acc: f32 = bias[co];
    let iy0: i32 = i32(oy) - i32(params.pad);
    let ix0: i32 = i32(ox) - i32(params.pad);
    let k = params.k;

    for (var ci: u32 = 0u; ci < params.in_C; ci = ci + 1u) {
        let xb: u32 = ci * params.in_H * params.in_W;
        let wb: u32 = (co * params.in_C + ci) * k * k;
        for (var ky: u32 = 0u; ky < k; ky = ky + 1u) {
            let iy: i32 = iy0 + i32(ky);
            if (iy < 0 || iy >= i32(params.in_H)) { continue; }
            for (var kx: u32 = 0u; kx < k; kx = kx + 1u) {
                let ix: i32 = ix0 + i32(kx);
                if (ix < 0 || ix >= i32(params.in_W)) { continue; }
                acc = acc + x[xb + u32(iy) * params.in_W + u32(ix)]
                          * weight[wb + ky * k + kx];
            }
        }
    }
    y[o] = acc;
}
