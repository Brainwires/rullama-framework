// Multi-query flash vision attention. Each workgroup processes Q = 4 query
// patches × 1 head, sharing one K/V load across all 4 queries — this cuts
// workgroup launch overhead 4× and K/V global-memory bandwidth 4× over the
// single-query flash kernel.
//
// I/O contract identical to vision_attention.wgsl:
//   q, k, v: f32 [n_patches, n_heads, head_dim]
//   out:     f32 [n_patches, n_heads, head_dim]
//
// Dispatch:  (ceil(n_patches / Q), n_heads, 1)
// Workgroup: 64 threads.
//
// Per-thread state (registers):
//   m[Q] — running max for each of the Q queries this WG owns
//   l[Q] — running normaliser for each query
//   o[Q] — output accumulator (only meaningful for tid < head_dim)
//
// Workgroup storage:
//   q_shared    (Q × HEAD_DIM_MAX)         1 KB
//   kv_tile     (TILE_T × HEAD_DIM_MAX)    8 KB   (reused for K then V)
//   tile_scores (Q × WG)                   1 KB   (per-query softmaxed scores)
//   rbuf        (WG)                     256 B
//   sum_buf     (WG)                     256 B
//   --------------------------------
//   total                              ~10.5 KB   (fits 16 KB minimum)

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
const Q_PER_WG: u32 = 4u;

var<workgroup> q_shared:    array<f32, 256>;     // Q_PER_WG × HEAD_DIM_MAX
var<workgroup> kv_tile:     array<f32, 2048>;    // TILE_T × HEAD_DIM_MAX
var<workgroup> tile_scores: array<f32, 256>;     // Q_PER_WG × WG
var<workgroup> rbuf:        array<f32, WG>;
var<workgroup> sum_buf:     array<f32, WG>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let qh: u32 = wid.y;
    if (qh >= params.n_heads) { return; }

    let head_dim:  u32 = params.head_dim;
    let n_patches: u32 = params.n_patches;
    let n_heads:   u32 = params.n_heads;

    let bq_base: u32 = wid.x * Q_PER_WG;
    // How many of the Q_PER_WG queries are valid for this workgroup.
    let q_count: u32 = min(Q_PER_WG, n_patches - bq_base);
    if (q_count == 0u) { return; }

    // --- Load Q query vectors into shared memory. ---
    // q_shared[q * head_dim + d] holds the d-th element of the q-th query.
    // 4 × 64 = 256 elements; 64 threads × 4 entries each.
    for (var i: u32 = 0u; i < Q_PER_WG; i = i + 1u) {
        let bq = bq_base + i;
        if (bq < n_patches && tid < head_dim) {
            let q_off = (bq * n_heads + qh) * head_dim + tid;
            q_shared[i * head_dim + tid] = q[q_off];
        }
    }
    workgroupBarrier();

    // Per-query online-softmax state in registers.
    var m0: f32 = -1.0e30; var l0: f32 = 0.0; var o0: f32 = 0.0;
    var m1: f32 = -1.0e30; var l1: f32 = 0.0; var o1: f32 = 0.0;
    var m2: f32 = -1.0e30; var l2: f32 = 0.0; var o2: f32 = 0.0;
    var m3: f32 = -1.0e30; var l3: f32 = 0.0; var o3: f32 = 0.0;

    let n_tiles = (n_patches + TILE_T - 1u) / TILE_T;
    for (var tile: u32 = 0u; tile < n_tiles; tile = tile + 1u) {
        let t0 = tile * TILE_T;
        let tile_size = min(TILE_T, n_patches - t0);

        // --- Load K_tile cooperatively. ---
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

        // --- Score + reduce + softmax merge for each of the Q queries. ---
        for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
            if (q_idx >= q_count) { break; }

            // Per-tile score for this query (one per thread, valid if tid<tile_size).
            var s_t: f32 = -1.0e30;
            if (tid < tile_size) {
                var sum: f32 = 0.0;
                let row_off = tid * head_dim;
                let q_row_off = q_idx * head_dim;
                let n_vec = head_dim / 4u;
                for (var dv: u32 = 0u; dv < n_vec; dv = dv + 1u) {
                    let dv4 = dv * 4u;
                    let qv = vec4<f32>(
                        q_shared[q_row_off + dv4],
                        q_shared[q_row_off + dv4 + 1u],
                        q_shared[q_row_off + dv4 + 2u],
                        q_shared[q_row_off + dv4 + 3u],
                    );
                    let kv = vec4<f32>(
                        kv_tile[row_off + dv4],
                        kv_tile[row_off + dv4 + 1u],
                        kv_tile[row_off + dv4 + 2u],
                        kv_tile[row_off + dv4 + 3u],
                    );
                    sum = sum + dot(qv, kv);
                }
                for (var d: u32 = n_vec * 4u; d < head_dim; d = d + 1u) {
                    sum = sum + q_shared[q_row_off + d] * kv_tile[row_off + d];
                }
                s_t = sum;
            }

            // Fused max-and-sum reduction.
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
            let tile_l = sum_buf[0];

            // Pull the current query's (m, l, o) into temporaries, merge,
            // store back. The 4-query branch is unrolled to keep state in
            // registers (no per-query indexed register file in WGSL).
            var m_cur: f32 = 0.0;
            var l_cur: f32 = 0.0;
            var o_cur: f32 = 0.0;
            switch q_idx {
                case 0u: { m_cur = m0; l_cur = l0; o_cur = o0; }
                case 1u: { m_cur = m1; l_cur = l1; o_cur = o1; }
                case 2u: { m_cur = m2; l_cur = l2; o_cur = o2; }
                default:  { m_cur = m3; l_cur = l3; o_cur = o3; }
            }
            let m_new = max(m_cur, tile_m);
            let alpha = exp(m_cur - m_new);

            var p_t: f32 = 0.0;
            if (tid < tile_size) {
                p_t = exp(s_t - m_new);
            }
            tile_scores[q_idx * WG + tid] = p_t;

            l_cur = l_cur * alpha + tile_l * exp(tile_m - m_new);
            m_cur = m_new;
            o_cur = o_cur * alpha;
            switch q_idx {
                case 0u: { m0 = m_cur; l0 = l_cur; o0 = o_cur; }
                case 1u: { m1 = m_cur; l1 = l_cur; o1 = o_cur; }
                case 2u: { m2 = m_cur; l2 = l_cur; o2 = o_cur; }
                default:  { m3 = m_cur; l3 = l_cur; o3 = o_cur; }
            }
        }

        workgroupBarrier();

        // --- Reuse kv_tile for V. ---
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

        // --- Per-query weighted sum: o[q, tid] += Σ_t p_t * V[t, tid]. ---
        if (tid < head_dim) {
            for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
                if (q_idx >= q_count) { break; }
                let s_off = q_idx * WG;
                var contrib: f32 = 0.0;
                let n_vec = tile_size / 4u;
                for (var tv: u32 = 0u; tv < n_vec; tv = tv + 1u) {
                    let t0_l = tv * 4u;
                    let sv = vec4<f32>(
                        tile_scores[s_off + t0_l],      tile_scores[s_off + t0_l + 1u],
                        tile_scores[s_off + t0_l + 2u], tile_scores[s_off + t0_l + 3u],
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
                    contrib = contrib + tile_scores[s_off + t_local] * kv_tile[t_local * head_dim + tid];
                }
                switch q_idx {
                    case 0u: { o0 = o0 + contrib; }
                    case 1u: { o1 = o1 + contrib; }
                    case 2u: { o2 = o2 + contrib; }
                    default:  { o3 = o3 + contrib; }
                }
            }
        }
        workgroupBarrier();
    }

    // --- Normalize and write. ---
    if (tid < head_dim) {
        for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
            if (q_idx >= q_count) { break; }
            let bq = bq_base + q_idx;
            let out_off = (bq * n_heads + qh) * head_dim + tid;
            var o_val: f32 = 0.0;
            var l_val: f32 = 0.0;
            switch q_idx {
                case 0u: { o_val = o0; l_val = l0; }
                case 1u: { o_val = o1; l_val = l1; }
                case 2u: { o_val = o2; l_val = l2; }
                default:  { o_val = o3; l_val = l3; }
            }
            out[out_off] = o_val / l_val;
        }
    }
}
