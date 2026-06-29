// Generic 2D convolution with f16 weights and f32 activations.
//
// Used by:
//   * Gemma 4 vision patch embedding (kernel=16, stride=16, padding=0)
//   * Gemma 4 audio SSCP (kernel=3, stride=2, padding=1)
//
// Tensor layouts:
//   input:  f32 [in_C, in_H, in_W]                  — channel-first
//   weight: f16 [out_C, in_C, kH, kW]               — packed 2× per u32
//   output: f32 [out_H, out_W, out_C]               — channel-LAST so the result
//                                                     can be viewed as
//                                                     [n_outputs = out_H*out_W, out_C]
//                                                     by every downstream consumer
//                                                     (vision: patch-major; audio
//                                                     SSCP: time-frequency-major).
//
// Output shape is fixed by the caller via params.out_H / out_W; we don't recompute.
// Out-of-bounds input reads (when stride/padding produce them) yield 0 — matches
// llama.cpp's im2col behavior and Ollama's Conv2D.
//
// Dispatch one thread per output element. Workgroup size 64; total threads =
// out_C * out_H * out_W (caller does the ceiling-divide for workgroups).

struct Params {
    in_C:    u32,
    in_H:    u32,
    in_W:    u32,
    out_C:   u32,
    out_H:   u32,
    out_W:   u32,
    kH:      u32,
    kW:      u32,
    sH:      u32,
    sW:      u32,
    pH:      u32,
    pW:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

fn read_f16(idx: u32) -> f32 {
    let u32_idx: u32 = idx >> 1u;
    let lane:    u32 = idx & 1u;
    let pair: vec2<f32> = unpack2x16float(weight[u32_idx]);
    if (lane == 0u) { return pair.x; }
    return pair.y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total: u32 = params.out_C * params.out_H * params.out_W;
    let out_idx: u32 = gid.x;
    if (out_idx >= total) { return; }

    // Output is laid out [out_H, out_W, out_C], so out_idx = (oy * out_W + ox) * out_C + j.
    let j:    u32 = out_idx % params.out_C;
    let pidx: u32 = out_idx / params.out_C;
    let oy:   u32 = pidx / params.out_W;
    let ox:   u32 = pidx % params.out_W;

    var acc: f32 = 0.0;

    // Anchor in input space (signed because of padding).
    let iy_base: i32 = i32(oy * params.sH) - i32(params.pH);
    let ix_base: i32 = i32(ox * params.sW) - i32(params.pW);

    for (var c: u32 = 0u; c < params.in_C; c = c + 1u) {
        for (var ky: u32 = 0u; ky < params.kH; ky = ky + 1u) {
            let iy: i32 = iy_base + i32(ky);
            if (iy < 0 || iy >= i32(params.in_H)) { continue; }
            for (var kx: u32 = 0u; kx < params.kW; kx = kx + 1u) {
                let ix: i32 = ix_base + i32(kx);
                if (ix < 0 || ix >= i32(params.in_W)) { continue; }

                let in_idx: u32 = (c * params.in_H + u32(iy)) * params.in_W + u32(ix);
                let w_idx:  u32 = ((j * params.in_C + c) * params.kH + ky) * params.kW + kx;
                acc = acc + x[in_idx] * read_f16(w_idx);
            }
        }
    }

    y[out_idx] = acc;
}
