// AdaIN1d, channel-major [C, T]: per-channel InstanceNorm over time, then style
// modulation. y[c,t] = (1 + gamma[c]) * (x[c,t] - mean_c) / sqrt(var_c + eps) + beta[c].
// gamma/beta are precomputed = chunk(fc(style)) on the host (or a prior matmul).
// (The checkpoint's InstanceNorm affine weight/bias are absent → identity, so omitted.)
// One workgroup per channel.

struct Params { c: u32, t: u32, eps: f32, _p0: u32 }

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
        if (tid < stride) { tile[tid] = tile[tid] + tile[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)           wg_id: vec3<u32>,
    @builtin(local_invocation_index) tid:   u32,
) {
    let ch = wg_id.x;
    if (ch >= params.c) { return; }
    let n = params.t;
    let base = ch * n;

    var ls: f32 = 0.0;
    var lss: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= n) { break; }
        let v = x[base + i];
        ls = ls + v;
        lss = lss + v * v;
        i = i + WG;
    }
    tile[tid] = ls;
    workgroupBarrier();
    reduce_sum(tid);
    let mean = tile[0] / f32(n);
    workgroupBarrier();
    tile[tid] = lss;
    workgroupBarrier();
    reduce_sum(tid);
    let var_c = tile[0] / f32(n) - mean * mean;
    let inv = 1.0 / sqrt(var_c + params.eps);

    let g = 1.0 + gamma[ch];
    let b = beta[ch];
    var k: u32 = tid;
    loop {
        if (k >= n) { break; }
        y[base + k] = g * ((x[base + k] - mean) * inv) + b;
        k = k + WG;
    }
}
