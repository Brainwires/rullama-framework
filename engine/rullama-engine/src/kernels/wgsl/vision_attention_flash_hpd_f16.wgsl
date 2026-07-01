// HPD + f16-LDS flash attention WITHOUT subgroups.
//
// Same head-major buffer layout and f16 workgroup storage as
// `vision_attention_flash_sub_hpd_f16.wgsl`, but uses a barrier tree for
// the per-tile (max, sum) reductions instead of `subgroupMax`/`subgroupAdd`.
//
// This is the **portable subgroup-free fast path**. Targets devices that
// expose `SHADER_F16` but where `subgroup_max_size < 64` (Apple Silicon,
// NVIDIA, Intel — where a 64-thread WG would split into 2+ subgroups and
// our subgroup-collapsed kernels would silently lose cross-subgroup
// reductions).
//
// Vs the original `vision_attention_flash_q8.wgsl`, this variant retains
// two of the three big wins:
//   * Head-major Q/K/V access → contiguous LDS loads per head.
//   * f16 LDS storage → ~2× wave concurrency on memory-latency-bound paths,
//     and packed-FP16 MADs on hardware that supports them (Apple AMX,
//     RDNA WMMA, NVIDIA tensor pipes).
//
// Microbench expectation: ~15-25% over Q=8 on Apple Silicon. Routed in
// preference to Q=8 whenever `has_f16` is true but `has_subgroups` is not.
//
// Requires `Features::SHADER_F16`.

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
const Q_PER_WG: u32 = 8u;

var<workgroup> q_shared:    array<f16, 512>;
var<workgroup> kv_tile:     array<f16, 2048>;
var<workgroup> tile_scores: array<f16, 512>;
// Tree-reduction scratch.
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

            // Fused max + sum-of-exp tree reduction.
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
