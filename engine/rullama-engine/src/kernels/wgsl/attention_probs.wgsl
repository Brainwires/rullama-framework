// Compute softmax attention probabilities for a single query position.
//
// Mirrors the forward attention kernel's Phase A–D (scores → max → exp
// → normalize), then writes probs[h, j] to a global buffer instead of
// applying them against V. Used by the training backward pass to
// reconstruct attention probs without modifying the forward kernel.
//
// Layout: q[n_heads, head_dim] (query at the current position),
// k_hist[history_len, n_kv_heads, head_dim], probs[n_heads, history_len].
// Sequence index is the slow axis of k_hist.
//
// As with the forward attention kernel, scores are un-scaled — Gemma 4
// absorbs the 1/√d factor into the q_norm RMSNorm weights, so backward
// arithmetic stays un-scaled too.

struct Params {
    head_dim:     u32,
    n_heads:      u32,
    n_kv_heads:   u32,
    heads_per_kv: u32,
    pos:          u32,
    history_len:  u32,
    window:       u32,
    _pad:         u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k_hist: array<f32>;
@group(0) @binding(3) var<storage, read_write> probs:  array<f32>;

const WG: u32 = 64u;
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

    var earliest: u32 = 0u;
    if (params.window != 0u) {
        if (pos + 1u >= params.window) {
            earliest = pos + 1u - params.window;
        }
    }

    let q_off = qh * head_dim;
    let p_off = qh * history_len;

    // ---- Phase A: raw scores ----
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

    // ---- Phase C: exp + sum ----
    var local_sum: f32 = 0.0;
    var t2: u32 = tid;
    loop {
        if (t2 >= history_len) { break; }
        let s = scores[t2];
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

    // ---- Phase D: normalize and write probs ----
    let inv = 1.0 / total;
    var t3: u32 = tid;
    loop {
        if (t3 >= history_len) { break; }
        probs[p_off + t3] = scores[t3] * inv;
        t3 = t3 + WG;
    }
}
