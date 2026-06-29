// Bidirectional batched self-attention for the Gemma 4 vision tower.
//
// Differs from the text attention.wgsl on three points:
//   * No causal/sliding-window mask — every patch attends to every other patch.
//   * No GQA: n_kv_heads == n_heads in vision.
//   * Batched queries: each (batch_query, head) gets its own workgroup, total
//     dispatch = (n_patches, n_heads, 1).
//   * Score scale = 1.0 (Ollama explicitly: model_vision.go::Forward → nn.Attention(... 1.0, nil)).
//
// Layout (patch-major):
//   q, k, v: f32 [n_patches, n_heads, head_dim]  — flat, fastest = head_dim
//   out:     f32 [n_patches, n_heads, head_dim]
//
// Workgroup state: scores[t] for t in 0..n_patches; max 4096.

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
const MAX_PATCHES: u32 = 4096u;

var<workgroup> scores: array<f32, MAX_PATCHES>;
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
fn main(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let bq: u32 = wid.x;
    let qh: u32 = wid.y;
    if (bq >= params.n_patches || qh >= params.n_heads) { return; }

    let head_dim: u32 = params.head_dim;
    let n_patches: u32 = params.n_patches;
    let q_off: u32 = (bq * params.n_heads + qh) * head_dim;

    // ---- Phase A: raw scores q · K[t, qh] for t in 0..n_patches ----
    var t: u32 = tid;
    loop {
        if (t >= n_patches) { break; }
        let k_off = (t * params.n_heads + qh) * head_dim;
        var s: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            s = s + q[q_off + d] * k[k_off + d];
        }
        scores[t] = s;
        t = t + WG;
    }
    workgroupBarrier();

    // ---- Phase B: max reduction ----
    var local_max: f32 = -1.0e30;
    var t1: u32 = tid;
    loop {
        if (t1 >= n_patches) { break; }
        local_max = max(local_max, scores[t1]);
        t1 = t1 + WG;
    }
    rbuf[tid] = local_max;
    workgroupBarrier();
    let m = block_max_reduce(tid);

    // ---- Phase C: exp(s - m), sum ----
    var local_sum: f32 = 0.0;
    var t2: u32 = tid;
    loop {
        if (t2 >= n_patches) { break; }
        let e = exp(scores[t2] - m);
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
        if (t3 >= n_patches) { break; }
        scores[t3] = scores[t3] * inv;
        t3 = t3 + WG;
    }
    workgroupBarrier();

    // ---- Phase E: weighted V ----
    let out_off = (bq * params.n_heads + qh) * head_dim;
    var d: u32 = tid;
    loop {
        if (d >= head_dim) { break; }
        var acc: f32 = 0.0;
        for (var tt: u32 = 0u; tt < n_patches; tt = tt + 1u) {
            let v_off = (tt * params.n_heads + qh) * head_dim;
            acc = acc + scores[tt] * v[v_off + d];
        }
        out[out_off + d] = acc;
        d = d + WG;
    }
}
