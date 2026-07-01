// RMSNorm with optional weight: y = x / sqrt(mean(x²) + eps) * w
//
// One workgroup per RMSNorm. Threads cooperate to compute the sum-of-squares via
// shared-memory reduction, then each thread writes its slice of y.
//
// Layout: x[n], y[n], optional w[n]. With params.has_weight = 0 we use 1.0 instead
// of w[i] (lets us pass a dummy w buffer and toggle behavior at dispatch time).

struct Params {
    n:           u32,
    eps:         f32,
    has_weight:  u32,
    _pad:        u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       w:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

const WG: u32 = 64u;

var<workgroup> tile: array<f32, WG>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) tid: u32) {
    let n = params.n;

    // Each thread accumulates a strided slice.
    var local_sumsq: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= n) { break; }
        let v = x[i];
        local_sumsq = local_sumsq + v * v;
        i = i + WG;
    }
    tile[tid] = local_sumsq;
    workgroupBarrier();

    // Tree reduction in shared memory.
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            tile[tid] = tile[tid] + tile[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let mean_sq = tile[0] / f32(n);
    let inv_rms = 1.0 / sqrt(mean_sq + params.eps);

    // Each thread writes its strided slice of y.
    var k: u32 = tid;
    loop {
        if (k >= n) { break; }
        var scale: f32 = 1.0;
        if (params.has_weight != 0u) {
            scale = w[k];
        }
        y[k] = x[k] * inv_rms * scale;
        k = k + WG;
    }
}
