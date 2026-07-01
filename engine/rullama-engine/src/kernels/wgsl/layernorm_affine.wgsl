// Per-row LayerNorm with affine (mean-subtraction + bias), one workgroup per row.
//
// y[r,c] = (x[r,c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]
// over the last dim (row_dim). Distinct from RMSNorm (which omits the mean and bias).
// Used by the Kokoro TTS port: ALBERT LayerNorm, the TextEncoder channel-axis norm
// (caller transposes), and (gamma=1,beta=0) the AdaLayerNorm / instance-norm inner norm.

struct Params {
    n_rows:     u32,
    row_dim:    u32,
    eps:        f32,
    has_affine: u32,
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
    let row = wg_id.x;
    if (row >= params.n_rows) { return; }
    let n = params.row_dim;
    let base = row * n;

    // local sum and sumsq over strided elements
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
    let var_r = tile[0] / f32(n) - mean * mean;
    let inv = 1.0 / sqrt(var_r + params.eps);

    var k: u32 = tid;
    loop {
        if (k >= n) { break; }
        var nv = (x[base + k] - mean) * inv;
        if (params.has_affine != 0u) {
            nv = nv * gamma[k] + beta[k];
        }
        y[base + k] = nv;
        k = k + WG;
    }
}
