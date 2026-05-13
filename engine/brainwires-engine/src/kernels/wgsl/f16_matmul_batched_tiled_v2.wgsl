// Register-tiled batched f16-weight matmul: y[b, j] = Σ_i x[b, i] * W[j, i].
//
// Extends f16_matmul_batched_tiled.wgsl with a 2×2 register tile per thread.
// Each workgroup of 64 threads still cooperatively loads shared tiles, but
// the output tile is now 16×16 (256 outputs) instead of 8×8 (64) — so the
// shared-memory loads amortise across 4× more arithmetic.
//
// Tile dimensions:
//   TILE_M = 16  — output rows per workgroup
//   TILE_N = 16  — output cols per workgroup
//   TILE_K = 16  — k-axis chunk per outer iteration
//   THREAD_TILE = 2×2 — outputs per thread
//
// Workgroup is 64 threads (8 × 8) — each thread owns a 2×2 sub-block.
//
// Dispatch: `(n.div_ceil(TILE_N), batch.div_ceil(TILE_M), 1)`.
//
// Shared memory: 256 + 256 = 512 f32 = 2 KB per workgroup — well within
// any GPU's limit. The reduction in global loads vs the v1 kernel is
// ~2× (the 2×2 register tile reuses each shared-mem read 4 times instead
// of 1).

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

const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;
const THREADS_M: u32 = 8u;   // 8 threads × 2 rows = 16
const THREADS_N: u32 = 8u;   // 8 threads × 2 cols = 16

var<workgroup> x_tile: array<f32, 256>;   // TILE_M × TILE_K
var<workgroup> w_tile: array<f32, 256>;   // TILE_N × TILE_K (row-major over (n, k))

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid = lid.x;
    let wg_n = wg_id.x;
    let wg_m = wg_id.y;

    // Each thread covers a 2×2 output sub-block. tm/tn are thread coords in
    // the 8×8 thread grid; we multiply by 2 to get the row/col base.
    let tm = tid / THREADS_N;        // 0..THREADS_M
    let tn = tid % THREADS_N;        // 0..THREADS_N

    let half_k = params.k / 2u;

    // 2×2 accumulator block (m_offset × n_offset).
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    let n_k_iters = (params.k + TILE_K - 1u) / TILE_K;

    for (var iter: u32 = 0u; iter < n_k_iters; iter = iter + 1u) {
        let k_offset = iter * TILE_K;

        // Load x_tile [TILE_M=16, TILE_K=16] = 256 f32. 64 threads × 4 each.
        for (var s: u32 = 0u; s < 4u; s = s + 1u) {
            let i = tid * 4u + s;        // 0..256
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

        // Load w_tile [TILE_N=16, TILE_K=16] = 256 f32 from 128 packed u32s.
        // 64 threads × 2 packed reads each.
        for (var s: u32 = 0u; s < 2u; s = s + 1u) {
            let pid = tid * 2u + s;      // 0..128
            let nw = pid / 8u;            // 0..16
            let kp = pid % 8u;            // 0..8 (pairs cover 16 k)
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

        // 2×2 register tile inner loop. For each kk we read 2 x's and 2 w's
        // and produce 4 MACs — the cross product reuses each shared-memory
        // load twice, doubling arithmetic intensity vs the v1 kernel.
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

    // Write 2×2 outputs with bounds checks.
    let m0 = wg_m * TILE_M + tm * 2u;
    let m1 = m0 + 1u;
    let n0 = wg_n * TILE_N + tn * 2u;
    let n1 = n0 + 1u;
    if (m0 < params.batch && n0 < params.n) { y[m0 * params.n + n0] = acc00; }
    if (m0 < params.batch && n1 < params.n) { y[m0 * params.n + n1] = acc01; }
    if (m1 < params.batch && n0 < params.n) { y[m1 * params.n + n0] = acc10; }
    if (m1 < params.batch && n1 < params.n) { y[m1 * params.n + n1] = acc11; }
}
