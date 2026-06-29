// HPD + f16-LDS flash attention with @workgroup_size(32), Q=8.
//
// Targets Apple Silicon (subgroup_size = 32) where our default WG=64 kernels
// run as 2 subgroups per WG, and every workgroup-barrier costs a cross-
// subgroup sync. Shrinking the WG to one subgroup makes barriers intra-
// wave — on Apple GPUs these are essentially free.
//
// Trade-offs vs WG=64:
//   * 2× more workgroup launches (288 query-groups × 12 heads = 3456 WGs
//     becomes 6912 WGs at n_patches=2304).
//   * Each lane does 2× more K-tile load work (64 vs 32 elements/lane).
//   * Barrier tree is one level shorter (5 vs 6 reduction levels).
//   * V-apply: 32 lanes × head_dim=64 → 2 output channels per lane.
//   * LDS footprint shrinks ~50% (tile_scores Q*WG halves) — better
//     occupancy on small-LDS adapters.
//
// Same buffer layout + uniform layout as the WG=64 HPD-f16 variant; the
// dispatcher just changes dispatchWorkgroups WG_X.

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

const WG: u32 = 32u;
const HEAD_DIM_MAX: u32 = 64u;
const TILE_T: u32 = 32u;
const Q_PER_WG: u32 = 8u;

var<workgroup> q_shared:    array<f16, 512>;    // Q_PER_WG × HEAD_DIM_MAX = 8 × 64
var<workgroup> kv_tile:     array<f16, 2048>;   // TILE_T × HEAD_DIM_MAX = 32 × 64
var<workgroup> tile_scores: array<f16, 256>;    // Q_PER_WG × WG = 8 × 32
var<workgroup> rbuf:        array<f32, WG>;
var<workgroup> sum_buf:     array<f32, WG>;

@compute @workgroup_size(32)
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

    // Load Q: each lane covers 2 channels (head_dim=64, WG=32).
    for (var i: u32 = 0u; i < Q_PER_WG; i = i + 1u) {
        let bq = bq_base + i;
        if (bq < n_patches) {
            // Channel 0..32
            if (tid < head_dim) {
                q_shared[i * head_dim + tid] = f16(q[head_base + bq * head_dim + tid]);
            }
            // Channel 32..64 (when head_dim > WG)
            let d2 = tid + WG;
            if (d2 < head_dim) {
                q_shared[i * head_dim + d2] = f16(q[head_base + bq * head_dim + d2]);
            }
        }
    }
    workgroupBarrier();

    var m_arr: array<f32, 8>;
    var l_arr: array<f32, 8>;
    var o_arr0: array<f32, 8>;  // Each lane owns channel `tid` of output…
    var o_arr1: array<f32, 8>;  // …and channel `tid + WG` (covers full head_dim).
    for (var i: u32 = 0u; i < Q_PER_WG; i = i + 1u) {
        m_arr[i] = -1.0e30;
        l_arr[i] = 0.0;
        o_arr0[i] = 0.0;
        o_arr1[i] = 0.0;
    }

    let n_tiles = (n_patches + TILE_T - 1u) / TILE_T;
    for (var tile: u32 = 0u; tile < n_tiles; tile = tile + 1u) {
        let t0 = tile * TILE_T;
        let tile_size = min(TILE_T, n_patches - t0);
        let total_k = tile_size * head_dim;
        let k_base = head_base + t0 * head_dim;

        // K tile load: 32 lanes × ~64 elements each.
        var lk = tid;
        loop {
            if (lk >= total_k) { break; }
            kv_tile[lk] = f16(k[k_base + lk]);
            lk = lk + WG;
        }
        workgroupBarrier();

        for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
            if (q_idx >= q_count) { break; }

            // Each lane scores K row index `tid` (lanes [0, tile_size) only).
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

            // 5-level barrier tree (log2 32). Intra-wave on Apple → cheap.
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
                    rbuf[tid]    = m_n;
                    sum_buf[tid] = l_a * exp(m_a - m_n) + l_b * exp(m_b - m_n);
                }
                workgroupBarrier();
                stride = stride / 2u;
            }
            let tile_m = rbuf[0];
            let tile_l = sum_buf[0];

            let m_cur = m_arr[q_idx];
            let l_cur = l_arr[q_idx];
            let m_new = max(m_cur, tile_m);
            let alpha = exp(m_cur - m_new);

            var p_t: f32 = 0.0;
            if (tid < tile_size) {
                p_t = exp(s_t - m_new);
            }
            tile_scores[q_idx * WG + tid] = f16(p_t);

            m_arr[q_idx] = m_new;
            l_arr[q_idx] = l_cur * alpha + tile_l * exp(tile_m - m_new);
            o_arr0[q_idx] = o_arr0[q_idx] * alpha;
            o_arr1[q_idx] = o_arr1[q_idx] * alpha;
        }

        workgroupBarrier();

        // V tile load — same cooperative pattern.
        var lv = tid;
        loop {
            if (lv >= total_k) { break; }
            kv_tile[lv] = f16(v[k_base + lv]);
            lv = lv + WG;
        }
        workgroupBarrier();

        // V-apply: lane `tid` accumulates output channels `tid` AND `tid+WG`.
        let d0 = tid;
        let d1 = tid + WG;
        let d0_ok = d0 < head_dim;
        let d1_ok = d1 < head_dim;
        for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
            if (q_idx >= q_count) { break; }
            let s_off = q_idx * WG;
            var c0: f32 = 0.0;
            var c1: f32 = 0.0;
            for (var t_local: u32 = 0u; t_local < tile_size; t_local = t_local + 1u) {
                let sv = f32(tile_scores[s_off + t_local]);
                if (d0_ok) { c0 = c0 + sv * f32(kv_tile[t_local * head_dim + d0]); }
                if (d1_ok) { c1 = c1 + sv * f32(kv_tile[t_local * head_dim + d1]); }
            }
            o_arr0[q_idx] = o_arr0[q_idx] + c0;
            o_arr1[q_idx] = o_arr1[q_idx] + c1;
        }
        workgroupBarrier();
    }

    // Write 2 output channels per lane.
    for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
        if (q_idx >= q_count) { break; }
        let bq = bq_base + q_idx;
        let inv_l = 1.0 / l_arr[q_idx];
        let d0 = tid;
        let d1 = tid + WG;
        if (d0 < head_dim) {
            out[head_base + bq * head_dim + d0] = o_arr0[q_idx] * inv_l;
        }
        if (d1 < head_dim) {
            out[head_base + bq * head_dim + d1] = o_arr1[q_idx] * inv_l;
        }
    }
}
