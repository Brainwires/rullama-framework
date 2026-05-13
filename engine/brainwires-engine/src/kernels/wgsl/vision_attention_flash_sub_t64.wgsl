// TILE_T=64, Q=12 subgroup-collapsed flash attention.
//
// Two simultaneous wins over the TILE_T=32 / Q=8 subgroup variant:
//   • TILE_T=64: every one of the 64 lanes participates in scoring (no
//     `if (tid < tile_size)` masking), and the outer tile loop runs 36 iters
//     instead of 72. K/V are global-loaded half as many times.
//   • Q=12: 1.5× more queries amortise the K/V load. With subgroup reductions
//     replacing barrier trees the per-query overhead is essentially `score +
//     subgroupMax + subgroupAdd + merge`, so going from Q=8 to Q=12 costs
//     little.
//
// Workgroup storage:
//   q_shared    (12 × 64)   3 KB
//   kv_tile     (64 × 64)  16 KB    (TILE_T × HEAD_DIM_MAX)
//   tile_scores (12 × 64)   3 KB
//   --------------------------
//   total                 ~22 KB     (> WebGPU spec minimum 16 KB)
//
// **Requires** the device's `max_compute_workgroup_storage_size` ≥ 22528.
// `WgpuCtx::new` opportunistically raises that limit when the adapter exposes
// it (Pro 555 / Metal: 32 KB). Routing in `vision_attention_chained` only
// chooses this kernel when `has_subgroups` is set.

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
const TILE_T: u32 = 64u;
const Q_PER_WG: u32 = 8u;

var<workgroup> q_shared:    array<f32, 512>;    // Q_PER_WG × HEAD_DIM_MAX = 8 × 64
var<workgroup> kv_tile:     array<f32, 4096>;   // TILE_T × HEAD_DIM_MAX = 64 × 64
var<workgroup> tile_scores: array<f32, 512>;    // Q_PER_WG × WG = 8 × 64

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
    let q_count: u32 = min(Q_PER_WG, n_patches - bq_base);
    if (q_count == 0u) { return; }

    // Load Q vectors (one channel per lane).
    for (var i: u32 = 0u; i < Q_PER_WG; i = i + 1u) {
        let bq = bq_base + i;
        if (bq < n_patches && tid < head_dim) {
            let q_off = (bq * n_heads + qh) * head_dim + tid;
            q_shared[i * head_dim + tid] = q[q_off];
        }
    }
    workgroupBarrier();

    var m_arr: array<f32, 8>;
    var l_arr: array<f32, 8>;
    var o_arr: array<f32, 8>;
    for (var i: u32 = 0u; i < Q_PER_WG; i = i + 1u) {
        m_arr[i] = -1.0e30;
        l_arr[i] = 0.0;
        o_arr[i] = 0.0;
    }

    let n_tiles = (n_patches + TILE_T - 1u) / TILE_T;
    for (var tile: u32 = 0u; tile < n_tiles; tile = tile + 1u) {
        let t0 = tile * TILE_T;
        let tile_size = min(TILE_T, n_patches - t0);

        // Load K tile cooperatively. tile_size × head_dim = up to 4096 f32 /
        // 64 lanes = up to 64 per lane.
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

        // For each query, compute its score against every K row in the tile.
        // Lane tid owns K row index `tid` (so all 64 lanes work — no masking).
        let in_tile = tid < tile_size;
        for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
            if (q_idx >= q_count) { break; }

            var s_t: f32 = -1.0e30;
            if (in_tile) {
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

            // Subgroup max + sum (one subgroup == one WG on AMD GCN).
            let tile_m = subgroupMax(s_t);
            var p_t: f32 = 0.0;
            if (in_tile) {
                p_t = exp(s_t - tile_m);
            }
            let tile_l = subgroupAdd(p_t);

            let m_cur = m_arr[q_idx];
            let l_cur = l_arr[q_idx];
            let o_cur = o_arr[q_idx];
            let m_new = max(m_cur, tile_m);
            let alpha = exp(m_cur - m_new);
            let beta  = exp(tile_m - m_new);
            tile_scores[q_idx * WG + tid] = p_t * beta;

            m_arr[q_idx] = m_new;
            l_arr[q_idx] = l_cur * alpha + tile_l * beta;
            o_arr[q_idx] = o_cur * alpha;
        }

        workgroupBarrier();

        // Reuse kv_tile for V (same 64 × 64 footprint).
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
                o_arr[q_idx] = o_arr[q_idx] + contrib;
            }
        }
        workgroupBarrier();
    }

    if (tid < head_dim) {
        for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
            if (q_idx >= q_count) { break; }
            let bq = bq_base + q_idx;
            let out_off = (bq * n_heads + qh) * head_dim + tid;
            out[out_off] = o_arr[q_idx] / l_arr[q_idx];
        }
    }
}
