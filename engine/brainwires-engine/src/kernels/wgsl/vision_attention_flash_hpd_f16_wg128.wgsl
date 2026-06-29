// HPD + f16-LDS flash attention with @workgroup_size(128).
//
// Hypothesis: at WG=64 the scoring phase only uses tile_size=32 threads
// while 32 are idle; the V-apply phase uses 64 (head_dim) threads but the
// scoring loop runs 8 sequential query iterations. With WG=128 we can
// potentially overlap multiple queries' scoring (TILE_T*Q parallelism)
// instead of serializing them, exposing more in-flight work to the
// scheduler.
//
// Layout: 128 lanes = 4 subgroups on Apple (subgroup=32). Each lane scores
// one (query, K-position) pair. With Q=8, TILE_T=32: 256 active scoring
// slots — but we only have 128 lanes. So each lane handles 2 (q, k) pairs.
//
// LDS: q_shared 1 KB + kv_tile 4 KB + tile_scores Q*WG=4 KB + rbuf+sum 1 KB
//      ≈ 10 KB. Fits comfortably in 32 KB.

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

const WG: u32 = 128u;
const HEAD_DIM_MAX: u32 = 64u;
const TILE_T: u32 = 32u;
const Q_PER_WG: u32 = 8u;

var<workgroup> q_shared:    array<f16, 512>;    // 8 × 64
var<workgroup> kv_tile:     array<f16, 2048>;   // 32 × 64
var<workgroup> tile_scores: array<f16, 1024>;   // 8 × 128
// Reduction is per-query (32 tile positions). Use a single 32-wide reduction
// scratch indexed by tile-position; only lanes 0..32 participate.
var<workgroup> rbuf:        array<f32, 32>;
var<workgroup> sum_buf:     array<f32, 32>;
var<workgroup> tile_m_lds:  array<f32, 8>;      // broadcast per-query tile max
var<workgroup> tile_l_lds:  array<f32, 8>;      // broadcast per-query tile sum

@compute @workgroup_size(128)
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

    // Load Q. With WG=128, all of q_shared (8×64=512 elems) is loaded by
    // first 8 lanes per query, or share across 128 lanes (each lane loads 4).
    for (var s: u32 = 0u; s < 4u; s = s + 1u) {
        let i = tid * 4u + s;            // 0..512
        let qi = i / head_dim;            // 0..8 (which query)
        let d  = i % head_dim;            // 0..64
        if (qi < Q_PER_WG) {
            let bq = bq_base + qi;
            if (bq < n_patches) {
                q_shared[i] = f16(q[head_base + bq * head_dim + d]);
            } else {
                q_shared[i] = f16(0.0);
            }
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
        let total_k = tile_size * head_dim;
        let k_base = head_base + t0 * head_dim;

        // K tile load: 128 lanes × ~16 each.
        var lk = tid;
        loop {
            if (lk >= total_k) { break; }
            kv_tile[lk] = f16(k[k_base + lk]);
            lk = lk + WG;
        }
        workgroupBarrier();

        // Score: each lane handles ONE (q_idx, k_pos) pair.
        // Map: lane = q_idx * TILE_T + k_pos for the first 8*32=256 lanes.
        // We have 128 lanes, so each lane handles 2 (q, k) pairs.
        for (var q_idx: u32 = 0u; q_idx < Q_PER_WG; q_idx = q_idx + 1u) {
            if (q_idx >= q_count) { break; }

            // Lanes 0..tile_size compute scores for this (q_idx, tid).
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

            // Reduce within first 32 lanes via rbuf/sum_buf. Other 96 lanes wait.
            if (tid < 32u) {
                rbuf[tid] = s_t;
                sum_buf[tid] = select(0.0, 1.0, tid < tile_size);
            }
            workgroupBarrier();
            var stride: u32 = 16u;
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
            if (tid == 0u) {
                tile_m_lds[q_idx] = rbuf[0];
                tile_l_lds[q_idx] = sum_buf[0];
            }
            workgroupBarrier();
            let tile_m = tile_m_lds[q_idx];
            let tile_l = tile_l_lds[q_idx];

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

        // V-apply: lanes 0..head_dim accumulate output channel `tid`.
        // Lanes [head_dim, WG) are idle this phase.
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
