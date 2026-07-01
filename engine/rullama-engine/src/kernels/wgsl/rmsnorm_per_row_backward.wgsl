// Per-row RMSNorm backward — mirrors `rmsnorm_per_row.wgsl` (forward).
//
// Forward:  y[r, i] = (x[r, i] / rms_r) * w[i]    (w is per-element, shared across rows;
//                                                  treated as 1 if has_weight = 0)
//           rms_r   = sqrt(mean(x[r, :]²) + eps)
// Backward: dx[r, j] = w[j]·dy[r, j]/rms_r - x[r, j]·s_r/(n·rms_r³)
//           where s_r = Σ_i dy[r, i]·w[i]·x[r, i].
//
// `w` is frozen (per-head shared weight; LoRA convention) — no `dw` output.
// One workgroup per row; rows independently reduce.

struct Params {
    n_rows:     u32,
    n:          u32,
    eps:        f32,
    has_weight: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       w:      array<f32>;
@group(0) @binding(3) var<storage, read>       dy:     array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:     array<f32>;

const WG: u32 = 64u;

var<workgroup> tile_sq:  array<f32, WG>;
var<workgroup> tile_dot: array<f32, WG>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let row = wid.x;
    if (row >= params.n_rows) { return; }
    let n = params.n;
    let has_w = params.has_weight != 0u;
    let row_off = row * n;

    var local_sq: f32 = 0.0;
    var local_dot: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= n) { break; }
        let xi  = x[row_off + i];
        let dyi = dy[row_off + i];
        let wi: f32 = select(1.0, w[i], has_w);
        local_sq  = local_sq  + xi * xi;
        local_dot = local_dot + dyi * wi * xi;
        i = i + WG;
    }
    tile_sq[tid]  = local_sq;
    tile_dot[tid] = local_dot;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            tile_sq[tid]  = tile_sq[tid]  + tile_sq[tid + stride];
            tile_dot[tid] = tile_dot[tid] + tile_dot[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let nf = f32(n);
    let inv_r = 1.0 / sqrt(tile_sq[0] / nf + params.eps);
    let s = tile_dot[0];
    let coef = s * inv_r * inv_r * inv_r / nf;

    var k: u32 = tid;
    loop {
        if (k >= n) { break; }
        let wk: f32 = select(1.0, w[k], has_w);
        dx[row_off + k] = wk * dy[row_off + k] * inv_r - x[row_off + k] * coef;
        k = k + WG;
    }
}
