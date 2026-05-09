// Per-row RMSNorm: do N independent RMSNorms in parallel, one per workgroup.
//
// Layout: x[n_rows * row_dim] → y[n_rows * row_dim], weight w[row_dim] shared across rows
// (or omitted via has_weight=0). Used for per-head Q/K/V norm and per-layer PLE norm,
// where the existing per-row CPU loop forces 4–35 readbacks per token in the chained
// forward.
//
// Equivalent to looping `rmsnorm.wgsl` over each row but in a single dispatch so the
// caller never has to submit between rows.

struct Params {
    n_rows:     u32,
    row_dim:    u32,
    eps:        f32,
    has_weight: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       w:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

const WG: u32 = 64u;
var<workgroup> tile: array<f32, WG>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)         wg_id: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let row = wg_id.x;
    if (row >= params.n_rows) { return; }
    let n = params.row_dim;
    let base = row * n;

    var local_sumsq: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= n) { break; }
        let v = x[base + i];
        local_sumsq = local_sumsq + v * v;
        i = i + WG;
    }
    tile[tid] = local_sumsq;
    workgroupBarrier();

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

    var k: u32 = tid;
    loop {
        if (k >= n) { break; }
        var scale: f32 = 1.0;
        if (params.has_weight != 0u) {
            scale = w[k];
        }
        y[base + k] = x[base + k] * inv_rms * scale;
        k = k + WG;
    }
}
