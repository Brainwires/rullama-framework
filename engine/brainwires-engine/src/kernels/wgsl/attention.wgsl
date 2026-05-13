// Softmax attention over a KV history. Single-batch, supports GQA (heads_per_kv > 1)
// and an optional sliding-window mask (window=0 → global causal-only).
//
// One workgroup per query head. Each workgroup:
//   1. Computes raw scores[t] = q · K[t, kvh] for all t (or -∞ if masked).
//   2. Workgroup-reduce max.
//   3. exp(scores - max), workgroup-reduce sum.
//   4. Normalize scores (in shared memory).
//   5. Walk V history weighted by normalized scores → out[qh, :].
//
// Layout: q[n_heads, head_dim], K[history_len, n_kv_heads, head_dim], same for V,
// out[n_heads, head_dim]. Sequence index is the slow axis of K/V.

struct Params {
    head_dim:     u32,
    n_heads:      u32,
    n_kv_heads:   u32,
    heads_per_kv: u32,
    pos:          u32,        // current logical position (last valid index in history)
    history_len:  u32,        // how many K/V entries are present
    window:       u32,        // SWA window size; 0 for global
    _pad:         u32,
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       q:       array<f32>;
@group(0) @binding(2) var<storage, read>       k_hist:  array<f32>;
@group(0) @binding(3) var<storage, read>       v_hist:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out:     array<f32>;

const WG: u32 = 64u;
// 4096 covers the text path (max ctx 4096) + the vision tower's pre-pool patch
// counts (≈ 2304 for a 768×768 image). Stays under Apple's 32 KB workgroup-mem
// limit (4096 f32 = 16 KB).
const MAX_HISTORY: u32 = 4096u;
const NEG_INF: f32 = -1.0e30;

var<workgroup> scores: array<f32, MAX_HISTORY>;
var<workgroup> rbuf:   array<f32, WG>;

fn block_max_reduce(tid: u32) -> f32 {
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            rbuf[tid] = max(rbuf[tid], rbuf[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return rbuf[0];
}

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
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let qh = wid.x;
    if (qh >= params.n_heads) { return; }
    let kvh = qh / params.heads_per_kv;
    let head_dim = params.head_dim;
    let history_len = params.history_len;
    let pos = params.pos;

    // Earliest valid position for this attention.
    var earliest: u32 = 0u;
    if (params.window != 0u) {
        if (pos + 1u >= params.window) {
            earliest = pos + 1u - params.window;
        }
    }

    let q_off = qh * head_dim;

    // ---- Phase A: compute raw scores ----
    var t: u32 = tid;
    loop {
        if (t >= history_len) { break; }
        if (t < earliest || t > pos) {
            scores[t] = NEG_INF;
        } else {
            let k_off = (t * params.n_kv_heads + kvh) * head_dim;
            var s: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                s = s + q[q_off + d] * k_hist[k_off + d];
            }
            scores[t] = s;
        }
        t = t + WG;
    }
    workgroupBarrier();

    // ---- Phase B: max reduction ----
    var local_max: f32 = NEG_INF;
    var t1: u32 = tid;
    loop {
        if (t1 >= history_len) { break; }
        local_max = max(local_max, scores[t1]);
        t1 = t1 + WG;
    }
    rbuf[tid] = local_max;
    workgroupBarrier();
    let m = block_max_reduce(tid);

    // ---- Phase C: exp(score - max) and sum ----
    var local_sum: f32 = 0.0;
    var t2: u32 = tid;
    loop {
        if (t2 >= history_len) { break; }
        let s = scores[t2];
        // Avoid exp on a -∞ mask: leave it at 0.
        var e: f32 = 0.0;
        if (s != NEG_INF) {
            e = exp(s - m);
        }
        scores[t2] = e;
        local_sum = local_sum + e;
        t2 = t2 + WG;
    }
    rbuf[tid] = local_sum;
    workgroupBarrier();
    let total = block_sum_reduce(tid);
    workgroupBarrier();

    // ---- Phase D: normalize ----
    let inv = 1.0 / total;
    var t3: u32 = tid;
    loop {
        if (t3 >= history_len) { break; }
        scores[t3] = scores[t3] * inv;
        t3 = t3 + WG;
    }
    workgroupBarrier();

    // ---- Phase E: weighted sum of V over history ----
    var d: u32 = tid;
    loop {
        if (d >= head_dim) { break; }
        var acc: f32 = 0.0;
        for (var tt: u32 = 0u; tt < history_len; tt = tt + 1u) {
            let w = scores[tt];
            if (w == 0.0) { continue; }
            let v_off = (tt * params.n_kv_heads + kvh) * head_dim;
            acc = acc + w * v_hist[v_off + d];
        }
        out[qh * head_dim + d] = acc;
        d = d + WG;
    }
}
