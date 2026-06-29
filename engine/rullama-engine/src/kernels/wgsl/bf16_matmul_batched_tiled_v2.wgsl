// Register-tiled batched bf16-weight matmul: y[b, j] = Σ_i x[b, i] * W[j, i].
//
// BF16 analogue of f16_matmul_batched_tiled_v2.wgsl. Same 16×16 output tile,
// 16-wide K chunks, 2×2 register sub-blocks per thread, 64 threads per
// workgroup. The only difference is the weight unpack — bf16 is the upper
// 16 bits of an f32, so `bitcast<f32>(bits << 16)` on the low/high halves.

struct Params {
    k:     u32,
    n:     u32,
    batch: u32,
    _pad:  u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;   // bf16 pairs per u32
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;
const THREADS_M: u32 = 8u;
const THREADS_N: u32 = 8u;

var<workgroup> x_tile: array<f32, 256>;
var<workgroup> w_tile: array<f32, 256>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid = lid.x;
    let wg_n = wg_id.x;
    let wg_m = wg_id.y;
    let tm = tid / THREADS_N;
    let tn = tid % THREADS_N;
    let half_k = params.k / 2u;

    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    let n_k_iters = (params.k + TILE_K - 1u) / TILE_K;

    for (var iter: u32 = 0u; iter < n_k_iters; iter = iter + 1u) {
        let k_offset = iter * TILE_K;

        for (var s: u32 = 0u; s < 4u; s = s + 1u) {
            let i = tid * 4u + s;
            let mx = i / TILE_K;
            let kx = i % TILE_K;
            let xr = wg_m * TILE_M + mx;
            let xk = k_offset + kx;
            if (xr < params.batch && xk < params.k) {
                x_tile[i] = x[xr * params.k + xk];
            } else {
                x_tile[i] = 0.0;
            }
        }

        for (var s: u32 = 0u; s < 2u; s = s + 1u) {
            let pid = tid * 2u + s;
            let nw = pid / 8u;
            let kp = pid % 8u;
            let wcol = wg_n * TILE_N + nw;
            let pi   = (k_offset / 2u) + kp;
            if (wcol < params.n && pi < half_k) {
                let packed = weight[wcol * half_k + pi];
                let lo = bf16_to_f32(packed & 0x0000FFFFu);
                let hi = bf16_to_f32(packed >> 16u);
                w_tile[nw * TILE_K + 2u * kp]      = lo;
                w_tile[nw * TILE_K + 2u * kp + 1u] = hi;
            } else {
                w_tile[nw * TILE_K + 2u * kp]      = 0.0;
                w_tile[nw * TILE_K + 2u * kp + 1u] = 0.0;
            }
        }

        workgroupBarrier();

        let m0 = tm * 2u;
        let m1 = m0 + 1u;
        let n0 = tn * 2u;
        let n1 = n0 + 1u;
        for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 1u) {
            let x0 = x_tile[m0 * TILE_K + kk];
            let x1 = x_tile[m1 * TILE_K + kk];
            let w0 = w_tile[n0 * TILE_K + kk];
            let w1 = w_tile[n1 * TILE_K + kk];
            acc00 = acc00 + x0 * w0;
            acc01 = acc01 + x0 * w1;
            acc10 = acc10 + x1 * w0;
            acc11 = acc11 + x1 * w1;
        }

        workgroupBarrier();
    }

    let m0 = wg_m * TILE_M + tm * 2u;
    let m1 = m0 + 1u;
    let n0 = wg_n * TILE_N + tn * 2u;
    let n1 = n0 + 1u;
    if (m0 < params.batch && n0 < params.n) { y[m0 * params.n + n0] = acc00; }
    if (m0 < params.batch && n1 < params.n) { y[m0 * params.n + n1] = acc01; }
    if (m1 < params.batch && n0 < params.n) { y[m1 * params.n + n0] = acc10; }
    if (m1 < params.batch && n1 < params.n) { y[m1 * params.n + n1] = acc11; }
}
