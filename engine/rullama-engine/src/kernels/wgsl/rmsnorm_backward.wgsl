// RMSNorm backward.
//
// Forward:  y[i] = (x[i] / r) * w[i]   where r = sqrt(mean(x²) + eps)
//                                       (w treated as 1 if has_weight = 0)
// Backward: dx[j] = w[j]·dy[j]/r - x[j]·s/(n·r³)
//           where s = Σ_i dy[i]·w[i]·x[i].
//
// `w` is frozen (LoRA convention) — no `dw` output. One workgroup
// processes the whole vector: two parallel reductions (Σx², Σ dy·w·x),
// then each thread writes its strided slice of `dx`.

struct Params {
    n:          u32,
    eps:        f32,
    has_weight: u32,
    _pad:       u32,
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
fn main(@builtin(local_invocation_index) tid: u32) {
    let n = params.n;
    let has_w = params.has_weight != 0u;

    var local_sq: f32 = 0.0;
    var local_dot: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= n) { break; }
        let xi  = x[i];
        let dyi = dy[i];
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
        dx[k] = wk * dy[k] * inv_r - x[k] * coef;
        k = k + WG;
    }
}
