// Attention backward — pass 1: produces `d_scores` (staged for pass 2) and `d_q`.
//
// One workgroup per query head `h`. Each WG, sharing one `d_probs`/`d_scores`
// staging array via workgroup memory:
//
//   1. d_probs[j]   = Σ_d  d_out[h, d] · v_hist[j, kv, d]
//   2. sum_pd       = Σ_j  probs[h, j] · d_probs[j]
//   3. d_scores[j]  = probs[h, j] · (d_probs[j] - sum_pd)
//   4. d_q[h, d]    = Σ_j  d_scores[j] · k_hist[j, kv, d]
//
// `d_scores` is also written to a global `[n_heads, history_len]` buffer
// so pass 2 (`attention_backward_dkv.wgsl`) can consume it without redoing
// the math. Forward attention's score is the raw dot product (Gemma 4
// absorbs the inverse-√d factor into the q RMSNorm), so the backward
// arithmetic is un-scaled as well.

struct Params {
    head_dim:     u32,
    n_heads:      u32,
    n_kv_heads:   u32,
    heads_per_kv: u32,
    history_len:  u32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}

// Pass 1 doesn't need `q` — d_q depends only on d_scores and k_hist, and
// d_scores depends only on probs, v_hist, and d_out. `q` lives on the
// pass-2 bind group.
@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read>       k_hist:   array<f32>;
@group(0) @binding(2) var<storage, read>       v_hist:   array<f32>;
@group(0) @binding(3) var<storage, read>       probs:    array<f32>;
@group(0) @binding(4) var<storage, read>       d_out:    array<f32>;
@group(0) @binding(5) var<storage, read_write> d_scores: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_q:      array<f32>;

const WG: u32 = 64u;
// Mirrors the forward attention kernel's MAX_HISTORY — keeps workgroup
// memory at 16 KB (4096 × 4 B) and accommodates the longest sequence
// rullama currently supports.
const MAX_HISTORY: u32 = 4096u;

var<workgroup> ds_buf: array<f32, MAX_HISTORY>;
var<workgroup> rbuf:   array<f32, WG>;

fn block_sum_reduce(tid: u32) -> f32 {
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            rbuf[tid] = rbuf[tid] + rbuf[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return rbuf[0];
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let h = wid.x;
    if (h >= params.n_heads) { return; }
    let kv = h / params.heads_per_kv;
    let head_dim = params.head_dim;
    let history_len = params.history_len;
    let q_off = h * head_dim;
    let p_off = h * history_len;

    // ---- Phase A: d_probs[j] into ds_buf, accumulate partial sum_pd ----
    var local_sum_pd: f32 = 0.0;
    var j_a: u32 = tid;
    loop {
        if (j_a >= history_len) { break; }
        let v_off = (j_a * params.n_kv_heads + kv) * head_dim;
        var dp: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dp = dp + d_out[q_off + d] * v_hist[v_off + d];
        }
        ds_buf[j_a] = dp;
        local_sum_pd = local_sum_pd + probs[p_off + j_a] * dp;
        j_a = j_a + WG;
    }
    rbuf[tid] = local_sum_pd;
    workgroupBarrier();
    let sum_pd = block_sum_reduce(tid);
    workgroupBarrier();

    // ---- Phase B: d_scores[j] = probs[h, j] · (d_probs[j] - sum_pd) ----
    var j_b: u32 = tid;
    loop {
        if (j_b >= history_len) { break; }
        let dp = ds_buf[j_b];
        let ds = probs[p_off + j_b] * (dp - sum_pd);
        ds_buf[j_b] = ds;
        d_scores[p_off + j_b] = ds;
        j_b = j_b + WG;
    }
    workgroupBarrier();

    // ---- Phase C: d_q[h, d] = Σ_j d_scores[j] · k_hist[j, kv, d] ----
    var d_c: u32 = tid;
    loop {
        if (d_c >= head_dim) { break; }
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < history_len; j = j + 1u) {
            let k_off = (j * params.n_kv_heads + kv) * head_dim;
            acc = acc + ds_buf[j] * k_hist[k_off + d_c];
        }
        d_q[q_off + d_c] = acc;
        d_c = d_c + WG;
    }
}
