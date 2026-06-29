// 2D average pooling, kernel = stride (no overlap, no padding).
//
// Used by Gemma 4 vision tower for `visionPoolAndProject` — pools the patch grid
// by `nMerge × nMerge` (default 3×3) before the projector. With patches arriving
// in [hidden, n_patches] layout, we operate on the spatial reshape
// [n_patches_x, n_patches_y, hidden]; output is [merged_x, merged_y, hidden].
//
// Layout assumption (channel-LAST after the spatial reshape):
//   x: f32 [in_h, in_w, channels]    — ix-major within each row, then row, then channel
//   y: f32 [out_h, out_w, channels]
// out_h = in_h / k, out_w = in_w / k.
//
// One thread per output element (channel-major last).

struct Params {
    in_h:    u32,
    in_w:    u32,
    out_h:   u32,
    out_w:   u32,
    channels: u32,
    k:       u32,   // kernel = stride
    _p0:     u32,
    _p1:     u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total: u32 = params.out_h * params.out_w * params.channels;
    let out_idx: u32 = gid.x;
    if (out_idx >= total) { return; }

    // Decompose: y is [out_h, out_w, channels] with channels innermost.
    let cstride: u32 = params.channels;
    let row_stride_out: u32 = params.out_w * cstride;
    let oy: u32 = out_idx / row_stride_out;
    let rem: u32 = out_idx % row_stride_out;
    let ox: u32 = rem / cstride;
    let c:  u32 = rem % cstride;

    let iy0: u32 = oy * params.k;
    let ix0: u32 = ox * params.k;

    var acc: f32 = 0.0;
    let row_stride_in: u32 = params.in_w * cstride;
    for (var dy: u32 = 0u; dy < params.k; dy = dy + 1u) {
        for (var dx: u32 = 0u; dx < params.k; dx = dx + 1u) {
            let in_idx: u32 = (iy0 + dy) * row_stride_in + (ix0 + dx) * cstride + c;
            acc = acc + x[in_idx];
        }
    }
    y[out_idx] = acc / f32(params.k * params.k);
}
