// Tiled batched bf16-weight matmul: y[b, j] = Σ_i x[b, i] * W[j, i].
//
// Same I/O semantics as bf16_matmul_batched.wgsl, but uses workgroup-shared
// memory tiling — same structure as f16_matmul_batched_tiled.wgsl, swapping
// `unpack2x16float` for the bf16 unpack (`bitcast<f32>(bits << 16)` on the
// low/high halves).
//
// Tile dimensions:
//   TILE_M = 8   — output rows per workgroup
//   TILE_N = 8   — output cols per workgroup
//   TILE_K = 16  — k-axis chunk per outer iteration
//
// Workgroup is 64 threads = 8 × 8 — one thread per output element.
//
// Dispatch: `(n.div_ceil(TILE_N), batch.div_ceil(TILE_M), 1)`.

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

const TILE_M: u32 = 8u;
const TILE_N: u32 = 8u;
const TILE_K: u32 = 16u;

var<workgroup> x_tile: array<f32, 128>;   // TILE_M × TILE_K
var<workgroup> w_tile: array<f32, 128>;   // TILE_N × TILE_K

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

    let m_local = tid / TILE_N;
    let n_local = tid % TILE_N;
    let m_global = wg_m * TILE_M + m_local;
    let n_global = wg_n * TILE_N + n_local;

    let half_k = params.k / 2u;

    var acc: f32 = 0.0;
    let n_k_iters = (params.k + TILE_K - 1u) / TILE_K;

    for (var iter: u32 = 0u; iter < n_k_iters; iter = iter + 1u) {
        let k_offset = iter * TILE_K;

        // Load x_tile: 128 f32, 64 threads × 2.
        let i1 = tid;
        let mx1 = i1 / TILE_K;
        let kx1 = i1 % TILE_K;
        let xr1 = wg_m * TILE_M + mx1;
        let xk1 = k_offset + kx1;
        if (xr1 < params.batch && xk1 < params.k) {
            x_tile[i1] = x[xr1 * params.k + xk1];
        } else {
            x_tile[i1] = 0.0;
        }
        let i2 = tid + 64u;
        let mx2 = i2 / TILE_K;
        let kx2 = i2 % TILE_K;
        let xr2 = wg_m * TILE_M + mx2;
        let xk2 = k_offset + kx2;
        if (xr2 < params.batch && xk2 < params.k) {
            x_tile[i2] = x[xr2 * params.k + xk2];
        } else {
            x_tile[i2] = 0.0;
        }

        // Load w_tile: 128 f32 from 64 packed u32s. Each thread reads ONE
        // packed u32 (= two contiguous bf16) and writes them at adjacent
        // k-positions for its assigned (n_local, k_pair).
        let nw = tid / 8u;                  // 0..TILE_N
        let kp = tid % 8u;                  // 0..TILE_K/2  (8 pairs cover 16 k)
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

        workgroupBarrier();

        for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 1u) {
            acc = acc + x_tile[m_local * TILE_K + kk] * w_tile[n_local * TILE_K + kk];
        }

        workgroupBarrier();
    }

    if (m_global < params.batch && n_global < params.n) {
        y[m_global * params.n + n_global] = acc;
    }
}
