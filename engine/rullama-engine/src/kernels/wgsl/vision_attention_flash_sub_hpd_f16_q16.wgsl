// HPD f16-LDS subgroup flash attention with Q=16 queries per WG.
//
// **NOT ROUTED BY DEFAULT** — kept in tree as a reference variant.
// On AMD Radeon Pro 555 / Metal: 995 ms/iter vs HPD-f16-Q8's 874 ms/iter.
// The slowdown is register pressure: Q=16 doubles the per-thread (m, l, o)
// state from 24 → 48 f32 regs, plus loop temporaries, which on GCN's 256-
// reg/lane budget cuts wave concurrency roughly in half. The K/V-bandwidth
// saving from re-using one tile load for 16 queries (vs 8) doesn't recover
// what occupancy loses.
//
// Workgroup storage (f16): 8 KB total
//   q_shared    (16 × 64)  2 KB
//   kv_tile     (32 × 64)  4 KB
//   tile_scores (16 × 64)  2 KB

enable f16;

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
const Q_PER_WG: u32 = 16u;

var<workgroup> q_shared:    array<f16, 1024>;   // Q_PER_WG × HEAD_DIM_MAX
var<workgroup> kv_tile:     array<f16, 2048>;   // TILE_T × HEAD_DIM_MAX
var<workgroup> tile_scores: array<f16, 1024>;   // Q_PER_WG × WG

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let qh: u32 = wid.y;
    if (qh >= params.n_heads) { return; }

    let head_dim:  u32 = params.head_dim;
    let n_patches: u32 = params.n_patches;
    let head_base: u32 = qh * n_patches * head_dim;

    let bq_base: u32 = wid.x * Q_PER_WG;
    let q_count: u32 = min(Q_PER_WG, n_patches - bq_base);
    if (q_count == 0u) { return; }

    for (var i: u32 = 0u; i < Q_PER_WG; i = i + 1u) {
        let bq = bq_base + i;
        if (bq < n_patches && tid < head_dim) {
            let q_off = head_base + bq * head_dim + tid;
            q_shared[i * head_dim + tid] = f16(q[q_off]);
        }
    }
    workgroupBarrier();

    var m_arr: array<f32, 16>;
    var l_arr: array<f32, 16>;
    var o_arr: array<f32, 16>;
    for (var i: u32 = 0u; i < Q_PER_WG; i = i + 1u) {
        m_arr[i] = -1.0e30;
        l_arr[i] = 0.0;
        o_arr[i] = 0.0;
    }

    let n_tiles = (n_patches + TILE_T - 1u) / TILE_T;
    for (var tile: u32 = 0u; tile < n_tiles; tile = tile + 1u) {
        let t0 = tile * TILE_T;
        let tile_size = min(TILE_T, n_patches - t0);
        let total_k = tile_size * head_dim;
        let k_base = head_base + t0 * head_dim;

        var lk = tid;
        loop {
            if (lk >= total_k) { break; }
            kv_tile[lk] = f16(k[k_base + lk]);
            lk = lk + WG;
        }
        workgroupBarrier();

        for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
            if (q_idx >= q_count) { break; }

            var s_t: f32 = -1.0e30;
            if (tid < tile_size) {
                var sum: f32 = 0.0;
                let row_off = tid * head_dim;
                let q_row_off = q_idx * head_dim;
                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    sum = sum + f32(q_shared[q_row_off + d]) * f32(kv_tile[row_off + d]);
                }
                s_t = sum;
            }

            let tile_m = subgroupMax(s_t);
            var p_t: f32 = 0.0;
            if (tid < tile_size) {
                p_t = exp(s_t - tile_m);
            }
            let tile_l = subgroupAdd(p_t);

            let m_cur = m_arr[q_idx];
            let l_cur = l_arr[q_idx];
            let o_cur = o_arr[q_idx];
            let m_new = max(m_cur, tile_m);
            let alpha = exp(m_cur - m_new);
            let beta  = exp(tile_m - m_new);
            tile_scores[q_idx * WG + tid] = f16(p_t * beta);

            m_arr[q_idx] = m_new;
            l_arr[q_idx] = l_cur * alpha + tile_l * beta;
            o_arr[q_idx] = o_cur * alpha;
        }

        workgroupBarrier();

        var lv = tid;
        loop {
            if (lv >= total_k) { break; }
            kv_tile[lv] = f16(v[k_base + lv]);
            lv = lv + WG;
        }
        workgroupBarrier();

        if (tid < head_dim) {
            for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
                if (q_idx >= q_count) { break; }
                let s_off = q_idx * WG;
                var contrib: f32 = 0.0;
                for (var t_local: u32 = 0u; t_local < tile_size; t_local = t_local + 1u) {
                    contrib = contrib +
                        f32(tile_scores[s_off + t_local]) *
                        f32(kv_tile[t_local * head_dim + tid]);
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
            let out_off = head_base + bq * head_dim + tid;
            out[out_off] = o_arr[q_idx] / l_arr[q_idx];
        }
    }
}
