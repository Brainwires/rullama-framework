// GroupNorm over a single image, channel-contiguous NCHW layout (N=1),
// one workgroup per group.
//
// x is laid out [C, H*W] (each channel's H*W spatial elements contiguous).
// Channels are split into `n_groups` contiguous groups of `chans_per_grp`
// channels each, so group g owns the contiguous block
//   x[g * chans_per_grp * hw  ..  (g+1) * chans_per_grp * hw).
// Mean/variance are taken over ALL elements of the group (channels × spatial);
// the optional affine is PER-CHANNEL (gamma/beta length C), not per-element:
//
//   y[c, s] = (x[c, s] - mean_g) / sqrt(var_g + eps) * gamma[c] + beta[c]
//
// This is the diffusion-UNet/VAE normalization (distinct from RMSNorm and from
// per-row LayerNorm). Reference oracle: reference::imagegen::ops::group_norm.

struct Params {
    n_groups:      u32,
    chans_per_grp: u32,
    hw:            u32,   // H * W
    eps:           f32,
    has_affine:    u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       gamma:  array<f32>;
@group(0) @binding(3) var<storage, read>       beta:   array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;

const WG: u32 = 64u;
var<workgroup> tile: array<f32, WG>;

fn reduce_sum(tid: u32) {
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            tile[tid] = tile[tid] + tile[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)           wg_id: vec3<u32>,
    @builtin(local_invocation_index) tid:   u32,
) {
    let g = wg_id.x;
    if (g >= params.n_groups) { return; }

    let grp_elems = params.chans_per_grp * params.hw; // elements in this group
    let base = g * grp_elems;
    let chan0 = g * params.chans_per_grp;             // first channel of group

    // local sum and sumsq over strided elements of the group block
    var ls: f32 = 0.0;
    var lss: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= grp_elems) { break; }
        let v = x[base + i];
        ls = ls + v;
        lss = lss + v * v;
        i = i + WG;
    }

    tile[tid] = ls;
    workgroupBarrier();
    reduce_sum(tid);
    let mean = tile[0] / f32(grp_elems);
    workgroupBarrier();

    tile[tid] = lss;
    workgroupBarrier();
    reduce_sum(tid);
    let var_g = tile[0] / f32(grp_elems) - mean * mean;
    let inv = 1.0 / sqrt(var_g + params.eps);

    var k: u32 = tid;
    loop {
        if (k >= grp_elems) { break; }
        var nv = (x[base + k] - mean) * inv;
        if (params.has_affine != 0u) {
            let c = chan0 + k / params.hw;  // absolute channel index
            nv = nv * gamma[c] + beta[c];
        }
        y[base + k] = nv;
        k = k + WG;
    }
}
