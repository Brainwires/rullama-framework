// Flash-attention-style bidirectional self-attention for the Gemma 4
// vision tower. Same I/O contract as vision_attention.wgsl:
//
//   q, k, v: f32 [n_patches, n_heads, head_dim]
//   out:     f32 [n_patches, n_heads, head_dim]
//
// Dispatch:  (n_patches, n_heads, 1)  — one workgroup per (query, head).
// Workgroup: 64 threads, head_dim = 64 → 1 thread per output dim.
//
// Replaces vision_attention.wgsl whose Phase E had each thread doing a
// 2304-iter sequential inner loop reading V from global memory (5 s per
// layer at 768×768 input on Radeon Pro 555). This kernel:
//
//   1. Loads q into workgroup-shared cache (64 elements, 4 bytes each).
//   2. Loops over patches in tiles of TILE_T = 32.
//   3. For each tile:
//      a. Cooperatively load K_tile (32 × 64 = 8 KB) into a shared buffer
//         (`kv_tile`).
//      b. Each thread computes one score = q · K_tile[t_local, *].
//      c. Tile-wide max-reduce → online softmax merge with the running
//         max `m` and normalizer `l`. Each thread rescales its output
//         accumulator `o` by `exp(m_old - m_new)`.
//      d. Reuse `kv_tile` for V — cooperative load of V_tile (32 × 64).
//      e. Each thread accumulates `o += Σ_t scores[t] · V_tile[t, tid]`.
//   4. After the last tile, normalize: `out[tid] = o / l`.
//
// Workgroup storage:
//   q_shared    (64 f32)      256 B
//   kv_tile     (32×64 f32)  8192 B  (re-used for K then V per tile)
//   tile_scores (64 f32)      256 B
//   rbuf        (64 f32)      256 B
//   --------------------------------
//   total                    8960 B  (fits in WebGPU's 16 KB minimum)
//
// Assumes head_dim ≤ 64. TILE_T = 32 chosen for high CU occupancy on
// AMD Radeon Pro 555 (64 KB LDS / CU). Bumping to TILE_T=64 cuts barrier
// count in half but doubles workgroup-shared size, dropping occupancy
// from ~7 to ~3 workgroups per CU — measured 2× regression on the Pro 555.

struct Params {
    head_dim:  u32,
    n_heads:   u32,
    n_patches: u32,
    _pad:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k:      array<f32>;
@group(0) @binding(3) var<storage, read>       v:      array<f32>;
@group(0) @binding(4) var<storage, read_write> out:    array<f32>;

const WG: u32 = 64u;
const HEAD_DIM_MAX: u32 = 64u;
const TILE_T: u32 = 32u;

var<workgroup> q_shared:    array<f32, HEAD_DIM_MAX>;
var<workgroup> kv_tile:     array<f32, 2048>;          // TILE_T × HEAD_DIM_MAX, reused for K then V
// Sized WG (not TILE_T) so threads with tid ≥ TILE_T can safely write -inf
// into their slot without OOB. Only the first TILE_T slots ever feed into
// the weighted sum.
var<workgroup> tile_scores: array<f32, WG>;
// `rbuf` and `sum_buf` are parallel reduction buffers — used together to do
// a fused max-and-sum (online-softmax-style) pair reduction in a single tree
// pass, saving ~6 barriers per tile over separate max + sum reductions.
var<workgroup> rbuf:        array<f32, WG>;
var<workgroup> sum_buf:     array<f32, WG>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let bq: u32 = wid.x;
    let qh: u32 = wid.y;
    if (bq >= params.n_patches || qh >= params.n_heads) { return; }

    let head_dim:  u32 = params.head_dim;
    let n_patches: u32 = params.n_patches;
    let n_heads:   u32 = params.n_heads;

    // --- Load q into workgroup shared memory (one element per thread). ---
    let q_off: u32 = (bq * n_heads + qh) * head_dim;
    if (tid < head_dim) {
        q_shared[tid] = q[q_off + tid];
    }
    workgroupBarrier();

    // Online softmax state. `o` is one element of the output vector per thread.
    var m: f32 = -1.0e30;
    var l: f32 = 0.0;
    var o: f32 = 0.0;

    let n_tiles = (n_patches + TILE_T - 1u) / TILE_T;
    for (var tile: u32 = 0u; tile < n_tiles; tile = tile + 1u) {
        let t0 = tile * TILE_T;
        let tile_size = min(TILE_T, n_patches - t0);

        // --- Load K_tile into kv_tile (tile_size × head_dim). ---
        // Each thread loads (tile_size * head_dim + WG - 1) / WG slots.
        let total_k = tile_size * head_dim;
        var lk = tid;
        loop {
            if (lk >= total_k) { break; }
            let t_local = lk / head_dim;
            let d_local = lk % head_dim;
            let g_off = ((t0 + t_local) * n_heads + qh) * head_dim + d_local;
            kv_tile[lk] = k[g_off];
            lk = lk + WG;
        }
        workgroupBarrier();

        // --- Score per t_local. Thread tid handles t_local = tid (or none if past). ---
        // Inner dot product is vectorized to 4-wide vec4 dot products — most
        // GPU back-ends issue these as single fused MAC instructions, halving
        // loop overhead and improving register reuse.
        var s_t: f32 = -1.0e30;
        if (tid < tile_size) {
            var sum: f32 = 0.0;
            let row_off = tid * head_dim;
            let n_vec = head_dim / 4u;
            for (var dv: u32 = 0u; dv < n_vec; dv = dv + 1u) {
                let dv4 = dv * 4u;
                let qv = vec4<f32>(
                    q_shared[dv4],     q_shared[dv4 + 1u],
                    q_shared[dv4 + 2u], q_shared[dv4 + 3u],
                );
                let kv = vec4<f32>(
                    kv_tile[row_off + dv4],     kv_tile[row_off + dv4 + 1u],
                    kv_tile[row_off + dv4 + 2u], kv_tile[row_off + dv4 + 3u],
                );
                sum = sum + dot(qv, kv);
            }
            // Tail for non-multiples of 4 (head_dim=64 is exact, so this is dead
            // code for the Gemma 4 vision shape but kept for safety).
            for (var d: u32 = n_vec * 4u; d < head_dim; d = d + 1u) {
                sum = sum + q_shared[d] * kv_tile[row_off + d];
            }
            s_t = sum;
        }

        // --- Fused max-and-sum reduction (combines what would otherwise be
        // two separate ~6-barrier tree reductions into one). Invariant: each
        // pair (rbuf[i], sum_buf[i]) represents the running max and the
        // associated sum-of-exp(s - max) over the values folded into slot i.
        // For a leaf (singleton valid entry): (s_t, 1.0). For an invalid
        // entry: (-inf, 0). Combine: m_n = max(m_a, m_b); l_n = l_a*exp(m_a-m_n)
        // + l_b*exp(m_b-m_n). ---
        rbuf[tid] = s_t;
        sum_buf[tid] = select(0.0, 1.0, tid < tile_size);
        workgroupBarrier();

        var stride: u32 = WG / 2u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) {
                let m_a = rbuf[tid];
                let m_b = rbuf[tid + stride];
                let l_a = sum_buf[tid];
                let l_b = sum_buf[tid + stride];
                let m_n = max(m_a, m_b);
                rbuf[tid] = m_n;
                sum_buf[tid] = l_a * exp(m_a - m_n) + l_b * exp(m_b - m_n);
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_m = rbuf[0];
        let tile_l = sum_buf[0];   // = Σ_t exp(s_t - tile_m)

        // --- Online softmax merge. ---
        let m_new = max(m, tile_m);
        let alpha = exp(m - m_new);

        // Compute this thread's normalized p_t (used in the V-weighted sum below).
        var p_t: f32 = 0.0;
        if (tid < tile_size) {
            p_t = exp(s_t - m_new);
        }
        tile_scores[tid] = p_t;
        workgroupBarrier();

        // Tile's contribution to the running normaliser, expressed in the new
        // max basis: tile_l × exp(tile_m - m_new).
        l = l * alpha + tile_l * exp(tile_m - m_new);
        m = m_new;

        // Rescale the running output accumulator BEFORE summing the new tile's
        // V contribution (so partial sums all share the new max).
        o = o * alpha;

        // --- Reuse kv_tile for V. Load V_tile. ---
        workgroupBarrier();   // ensure all threads finished reading kv_tile (K)
        var lv = tid;
        loop {
            if (lv >= total_k) { break; }
            let t_local = lv / head_dim;
            let d_local = lv % head_dim;
            let g_off = ((t0 + t_local) * n_heads + qh) * head_dim + d_local;
            kv_tile[lv] = v[g_off];
            lv = lv + WG;
        }
        workgroupBarrier();

        // --- Weighted sum: each thread adds its column of V scaled by scores.
        // Unroll by 4 in the t_local loop so the back-end can issue 4 strided
        // V reads ahead of the FMAs (memory parallelism) and fold the work
        // into one fused vec4 dot. tile_size is TILE_T=32 in the common case
        // so we take the fast path; tail handles non-multiples of 4. ---
        if (tid < head_dim) {
            var contrib: f32 = 0.0;
            let n_vec = tile_size / 4u;
            for (var tv: u32 = 0u; tv < n_vec; tv = tv + 1u) {
                let t0_l = tv * 4u;
                let sv = vec4<f32>(
                    tile_scores[t0_l],      tile_scores[t0_l + 1u],
                    tile_scores[t0_l + 2u], tile_scores[t0_l + 3u],
                );
                let vv = vec4<f32>(
                    kv_tile[t0_l * head_dim + tid],
                    kv_tile[(t0_l + 1u) * head_dim + tid],
                    kv_tile[(t0_l + 2u) * head_dim + tid],
                    kv_tile[(t0_l + 3u) * head_dim + tid],
                );
                contrib = contrib + dot(sv, vv);
            }
            for (var t_local: u32 = n_vec * 4u; t_local < tile_size; t_local = t_local + 1u) {
                contrib = contrib + tile_scores[t_local] * kv_tile[t_local * head_dim + tid];
            }
            o = o + contrib;
        }
        workgroupBarrier();
    }

    // --- Normalize and write out. ---
    if (tid < head_dim) {
        let out_off = (bq * n_heads + qh) * head_dim + tid;
        out[out_off] = o / l;
    }
}
