// Subgroup-collapsed flash vision attention. Same numerics as the Q=8 variant
// but the tree-reduction barriers are replaced by `subgroupMax` /
// `subgroupAdd`. On AMD GCN a wave is 64 lanes, so the entire 64-thread
// workgroup is one subgroup — every per-query reduction over TILE_T=32 K
// positions collapses to two intrinsics with zero barriers between them.
//
// Per tile the Q=8 (barrier) variant runs roughly:
//     load_K_barrier
//     × 8 queries: [score, ~6 reduction barriers, merge, ~6 reduction barriers]
//     load_V_barrier
//     × 8 queries: weighted-sum
// = ~56 workgroup barriers per tile, 72 tiles = ~4 k barriers per WG.
// Subgroup variant cuts that to ~3 barriers per tile (K load, V load, end).
//
// Workgroup storage (subgroup variant):
//   q_shared    (8 × 64)   2 KB
//   kv_tile     (32 × 64)  8 KB
//   tile_scores (8 × 64)   2 KB   (write-out for V-apply phase)
//   --------------------------
//   total                ~12 KB
//
// Requires `Features::SUBGROUP`. Routed only when WgpuCtx::has_subgroups.
// (No `enable subgroups;` directive — naga recognises subgroupAdd/Max as
// builtins gated by the device feature, not by a WGSL enable.)

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
const Q_PER_WG: u32 = 8u;

var<workgroup> q_shared:    array<f32, 512>;    // Q_PER_WG × HEAD_DIM_MAX
var<workgroup> kv_tile:     array<f32, 2048>;   // TILE_T × HEAD_DIM_MAX
var<workgroup> tile_scores: array<f32, 512>;    // Q_PER_WG × WG

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

    // Load Q vectors. tid in [0, head_dim) covers the channel axis.
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

        // Load K tile into shared memory.
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

        // For each query, compute its score against this tile.
        // Lanes [0, tile_size) hold meaningful scores, lanes [tile_size, WG)
        // hold a sentinel that won't affect max / sum.
        for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
            if (q_idx >= q_count) { break; }

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

            // Subgroup max. WG=64 == 1 subgroup on AMD GCN; this is a single
            // intrinsic with no barrier. tile_m is broadcast to all lanes.
            let tile_m = subgroupMax(s_t);

            // Per-lane probability against this tile's max.
            var p_t: f32 = 0.0;
            if (tid < tile_size) {
                p_t = exp(s_t - tile_m);
            }
            // Subgroup sum → tile_l broadcast to all lanes.
            let tile_l = subgroupAdd(p_t);

            // Merge with running (m, l, o). All lanes execute identically since
            // tile_m / tile_l are subgroup-uniform.
            let m_cur = m_arr[q_idx];
            let l_cur = l_arr[q_idx];
            let o_cur = o_arr[q_idx];
            let m_new = max(m_cur, tile_m);
            let alpha = exp(m_cur - m_new);
            // p_t was computed against tile_m; rescale to m_new for V-apply.
            let beta = exp(tile_m - m_new);
            let p_scaled = p_t * beta;
            tile_scores[q_idx * WG + tid] = p_scaled;

            m_arr[q_idx] = m_new;
            l_arr[q_idx] = l_cur * alpha + tile_l * beta;
            o_arr[q_idx] = o_cur * alpha;
        }

        // Cross-subgroup ordering before the K → V tile overwrite. (Within a
        // single subgroup the prior subgroupAdd already serialises lanes; but
        // we still need a workgroup barrier so other subgroups, if any, see
        // the tile_scores writes and the kv_tile slot is reusable for V.)
        workgroupBarrier();

        // Reuse kv_tile for V.
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
