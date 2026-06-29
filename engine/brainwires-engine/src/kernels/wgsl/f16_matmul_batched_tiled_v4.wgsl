// v4 batched f16-weight matmul: 64×32 output tile, 8×4 register sub-block.
//
// Over v3 (32×32 / 4×4):
//   * 64×32 output tile = 2× the outputs per WG (2× fewer WG launches).
//   * Inner k-iter: 8 x's + 4 w's load → 32 MACs per thread, AI = 2.67 ops/load
//     (vs v3's 16 MACs / 8 loads = 2.0).
//   * x_tile 64×16 f32 = 4 KB, w_tile 32×16 f32 = 2 KB → 6 KB total LDS
//     (vs v3's 4 KB) — still well under any sensible occupancy limit.
//   * Per-thread regs: 32 accumulators + ~12 working = ~44 regs. Within GCN
//     wave-occupancy budget (256 regs/lane → 5+ waves/SIMD16).
//
// Dispatch: (n.div_ceil(32), batch.div_ceil(64), 1). WG=64 threads as 8×8.

struct Params { k: u32, n: u32, batch: u32, _pad: u32 }

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;   // f16 pairs per u32
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

const TILE_M: u32 = 64u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 16u;
const THREADS_M: u32 = 8u;   // 8 threads × 8 rows = 64
const THREADS_N: u32 = 8u;   // 8 threads × 4 cols = 32

var<workgroup> x_tile: array<f32, 1024>;  // TILE_M × TILE_K = 64 × 16
var<workgroup> w_tile: array<f32, 512>;   // TILE_N × TILE_K = 32 × 16

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid = lid.x;
    let wg_n = wg_id.x;
    let wg_m = wg_id.y;
    let tm = tid / THREADS_N;        // 0..8
    let tn = tid % THREADS_N;        // 0..8

    let half_k = params.k / 2u;

    // 8 rows × 4 cols register block → 32 accumulators per thread.
    var acc: array<f32, 32>;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) { acc[i] = 0.0; }

    let n_k_iters = (params.k + TILE_K - 1u) / TILE_K;

    for (var iter: u32 = 0u; iter < n_k_iters; iter = iter + 1u) {
        let k_offset = iter * TILE_K;

        // Load x_tile [TILE_M=64, TILE_K=16] = 1024 f32. 64 threads × 16 each.
        for (var s: u32 = 0u; s < 16u; s = s + 1u) {
            let i = tid * 16u + s;
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

        // Load w_tile [TILE_N=32, TILE_K=16] = 512 f32 from 256 packed u32s.
        // 64 threads × 4 packed reads each.
        for (var s: u32 = 0u; s < 4u; s = s + 1u) {
            let pid = tid * 4u + s;       // 0..256
            let nw = pid / 8u;             // 0..32
            let kp = pid % 8u;             // 0..8 pairs (covers 16 k)
            let wcol = wg_n * TILE_N + nw;
            let pi   = (k_offset / 2u) + kp;
            if (wcol < params.n && pi < half_k) {
                let packed = weight[wcol * half_k + pi];
                let pair = unpack2x16float(packed);
                w_tile[nw * TILE_K + 2u * kp]      = pair.x;
                w_tile[nw * TILE_K + 2u * kp + 1u] = pair.y;
            } else {
                w_tile[nw * TILE_K + 2u * kp]      = 0.0;
                w_tile[nw * TILE_K + 2u * kp + 1u] = 0.0;
            }
        }

        workgroupBarrier();

        // 8×4 register inner loop. Per kk: load 8 x's + 4 w's, do 32 MACs.
        let m0 = tm * 8u;
        let n0 = tn * 4u;
        for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 1u) {
            let x0 = x_tile[(m0     ) * TILE_K + kk];
            let x1 = x_tile[(m0 + 1u) * TILE_K + kk];
            let x2 = x_tile[(m0 + 2u) * TILE_K + kk];
            let x3 = x_tile[(m0 + 3u) * TILE_K + kk];
            let x4 = x_tile[(m0 + 4u) * TILE_K + kk];
            let x5 = x_tile[(m0 + 5u) * TILE_K + kk];
            let x6 = x_tile[(m0 + 6u) * TILE_K + kk];
            let x7 = x_tile[(m0 + 7u) * TILE_K + kk];
            let w0 = w_tile[(n0     ) * TILE_K + kk];
            let w1 = w_tile[(n0 + 1u) * TILE_K + kk];
            let w2 = w_tile[(n0 + 2u) * TILE_K + kk];
            let w3 = w_tile[(n0 + 3u) * TILE_K + kk];
            acc[ 0] = acc[ 0] + x0 * w0; acc[ 1] = acc[ 1] + x0 * w1;
            acc[ 2] = acc[ 2] + x0 * w2; acc[ 3] = acc[ 3] + x0 * w3;
            acc[ 4] = acc[ 4] + x1 * w0; acc[ 5] = acc[ 5] + x1 * w1;
            acc[ 6] = acc[ 6] + x1 * w2; acc[ 7] = acc[ 7] + x1 * w3;
            acc[ 8] = acc[ 8] + x2 * w0; acc[ 9] = acc[ 9] + x2 * w1;
            acc[10] = acc[10] + x2 * w2; acc[11] = acc[11] + x2 * w3;
            acc[12] = acc[12] + x3 * w0; acc[13] = acc[13] + x3 * w1;
            acc[14] = acc[14] + x3 * w2; acc[15] = acc[15] + x3 * w3;
            acc[16] = acc[16] + x4 * w0; acc[17] = acc[17] + x4 * w1;
            acc[18] = acc[18] + x4 * w2; acc[19] = acc[19] + x4 * w3;
            acc[20] = acc[20] + x5 * w0; acc[21] = acc[21] + x5 * w1;
            acc[22] = acc[22] + x5 * w2; acc[23] = acc[23] + x5 * w3;
            acc[24] = acc[24] + x6 * w0; acc[25] = acc[25] + x6 * w1;
            acc[26] = acc[26] + x6 * w2; acc[27] = acc[27] + x6 * w3;
            acc[28] = acc[28] + x7 * w0; acc[29] = acc[29] + x7 * w1;
            acc[30] = acc[30] + x7 * w2; acc[31] = acc[31] + x7 * w3;
        }

        workgroupBarrier();
    }

    // Write 8×4 outputs with bounds checks.
    let m_base = wg_m * TILE_M + tm * 8u;
    let n_base = wg_n * TILE_N + tn * 4u;
    for (var dm: u32 = 0u; dm < 8u; dm = dm + 1u) {
        for (var dn: u32 = 0u; dn < 4u; dn = dn + 1u) {
            let m = m_base + dm;
            let n = n_base + dn;
            if (m < params.batch && n < params.n) {
                y[m * params.n + n] = acc[dm * 4u + dn];
            }
        }
    }
}
