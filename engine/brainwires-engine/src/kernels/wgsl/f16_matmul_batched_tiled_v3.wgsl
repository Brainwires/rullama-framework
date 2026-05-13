// Register-tiled batched f16-weight matmul v3: 32×32 output tile with each
// thread holding a 4×4 register sub-block.
//
// Going from v2 (16×16 tile, 2×2/thread) to v3 (32×32 tile, 4×4/thread):
//   * 4× the outputs per workgroup → 4× fewer workgroup launches for the
//     same problem.
//   * Each shared-memory load amortises over 4× more arithmetic — the inner
//     loop reads 4 x's and 4 w's then does 16 MACs instead of 2/2/4.
//   * Workgroup-shared storage stays at 4 KB (x_tile 32×16 + w_tile 32×16).
//
// Workgroup is 64 threads (8 × 8 thread grid). Each thread owns a 4×4
// region of the output tile.
//
// Dispatch: (n.div_ceil(32), batch.div_ceil(32), 1).

struct Params {
    k:     u32,
    n:     u32,
    batch: u32,
    _pad:  u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;   // f16 pairs per u32
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 16u;
const THREADS_M: u32 = 8u;   // 8 threads × 4 rows = 32
const THREADS_N: u32 = 8u;   // 8 threads × 4 cols = 32

var<workgroup> x_tile: array<f32, 512>;   // TILE_M × TILE_K
var<workgroup> w_tile: array<f32, 512>;   // TILE_N × TILE_K

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid = lid.x;
    let wg_n = wg_id.x;
    let wg_m = wg_id.y;

    let tm = tid / THREADS_N;        // 0..THREADS_M
    let tn = tid % THREADS_N;        // 0..THREADS_N

    let half_k = params.k / 2u;

    // 4×4 register accumulators (16 outputs per thread).
    var acc: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    let n_k_iters = (params.k + TILE_K - 1u) / TILE_K;

    for (var iter: u32 = 0u; iter < n_k_iters; iter = iter + 1u) {
        let k_offset = iter * TILE_K;

        // Load x_tile [TILE_M=32, TILE_K=16] = 512 f32. 64 threads × 8 each.
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let i = tid * 8u + s;        // 0..512
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
            let pid = tid * 4u + s;      // 0..256
            let nw = pid / 8u;            // 0..32
            let kp = pid % 8u;            // 0..8 pairs (cover 16 k)
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

        // 4×4 register tile inner loop. Each kk reads 4 x's and 4 w's,
        // producing 16 MACs — 4× the v2 arithmetic intensity.
        let m0 = tm * 4u;
        let n0 = tn * 4u;
        for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 1u) {
            let x0 = x_tile[(m0     ) * TILE_K + kk];
            let x1 = x_tile[(m0 + 1u) * TILE_K + kk];
            let x2 = x_tile[(m0 + 2u) * TILE_K + kk];
            let x3 = x_tile[(m0 + 3u) * TILE_K + kk];
            let w0 = w_tile[(n0     ) * TILE_K + kk];
            let w1 = w_tile[(n0 + 1u) * TILE_K + kk];
            let w2 = w_tile[(n0 + 2u) * TILE_K + kk];
            let w3 = w_tile[(n0 + 3u) * TILE_K + kk];
            acc[ 0] = acc[ 0] + x0 * w0;
            acc[ 1] = acc[ 1] + x0 * w1;
            acc[ 2] = acc[ 2] + x0 * w2;
            acc[ 3] = acc[ 3] + x0 * w3;
            acc[ 4] = acc[ 4] + x1 * w0;
            acc[ 5] = acc[ 5] + x1 * w1;
            acc[ 6] = acc[ 6] + x1 * w2;
            acc[ 7] = acc[ 7] + x1 * w3;
            acc[ 8] = acc[ 8] + x2 * w0;
            acc[ 9] = acc[ 9] + x2 * w1;
            acc[10] = acc[10] + x2 * w2;
            acc[11] = acc[11] + x2 * w3;
            acc[12] = acc[12] + x3 * w0;
            acc[13] = acc[13] + x3 * w1;
            acc[14] = acc[14] + x3 * w2;
            acc[15] = acc[15] + x3 * w3;
        }

        workgroupBarrier();
    }

    // Write 4×4 outputs with bounds checks.
    let m_base = wg_m * TILE_M + tm * 4u;
    let n_base = wg_n * TILE_N + tn * 4u;
    for (var dm: u32 = 0u; dm < 4u; dm = dm + 1u) {
        for (var dn: u32 = 0u; dn < 4u; dn = dn + 1u) {
            let m = m_base + dm;
            let n = n_base + dn;
            if (m < params.batch && n < params.n) {
                y[m * params.n + n] = acc[dm * 4u + dn];
            }
        }
    }
}
