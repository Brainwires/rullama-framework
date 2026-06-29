// HPD + f16-LDS flash attention WITHOUT subgroups, Q=4 queries per WG.
//
// Tuned for mobile GPUs (Apple A-series, ARM Mali, Adreno) where
// per-lane register files are smaller than AMD GCN's 256 regs. Halves
// per-thread (m, l, o) state from 24 → 12 f32 regs, freeing register
// budget that the compiler can use for inner-loop arithmetic. Trade-
// off: 2× more workgroup launches (576 vs 288 query-groups for
// n_patches=2304) and 2× more K/V global loads. On Apple A18 + WebGPU
// the register-occupancy win comes out ahead.
//
// Same algorithm + barrier-tree reduction as the Q=8 variant.

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
const Q_PER_WG: u32 = 4u;

var<workgroup> q_shared:    array<f16, 256>;    // Q_PER_WG × HEAD_DIM_MAX = 4 × 64
var<workgroup> kv_tile:     array<f16, 2048>;   // TILE_T × HEAD_DIM_MAX
var<workgroup> tile_scores: array<f16, 256>;    // Q_PER_WG × WG
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

    var m_arr: array<f32, 4>;
    var l_arr: array<f32, 4>;
    var o_arr: array<f32, 4>;
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
            let o_cur = o_arr[q_idx];
            let m_new = max(m_cur, tile_m);
            let alpha = exp(m_cur - m_new);

            var p_t: f32 = 0.0;
            if (tid < tile_size) {
                p_t = exp(s_t - m_new);
            }
            tile_scores[q_idx * WG + tid] = f16(p_t);

            m_arr[q_idx] = m_new;
            l_arr[q_idx] = l_cur * alpha + tile_l * exp(tile_m - m_new);
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
