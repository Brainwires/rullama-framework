// DiffusionGemma non-autoregressive region-masked attention.
//
// Mirrors `reference/diffusion/forward.rs::masked_attention` (the tested CPU
// oracle) + the region mask in `reference/diffusion/mask.rs::allowed`. Unlike
// rullama's causal+KV-cache attention, this is a full-sequence single pass:
// every query position attends to every key position permitted by the region
// mask (canvas queries bidirectional; prompt queries causal+SWA-windowed,
// never canvas). Score scale is 1.0 (Gemma 4 folds it into the Q-norm).
//
// One workgroup per (query position, query head). GQA: query head qh reads kv
// head qh / (n_heads / n_kv_heads). Layout: q [n, n_heads, head_dim] row-major,
// k/v [n, n_kv_heads, head_dim].

struct Params {
    n_tokens:   u32,
    n_heads:    u32,
    n_kv_heads: u32,
    head_dim:   u32,
    prompt_len: u32,
    n_swa:      u32,
    swa_layer:  u32, // 0/1
    _pad:       u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       q: array<f32>;
@group(0) @binding(2) var<storage, read>       k: array<f32>;
@group(0) @binding(3) var<storage, read>       v: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;

const WG: u32 = 128u;
const MAX_HEAD_DIM: u32 = 512u;
const MAX_TOKENS: u32 = 1280u; // 256 canvas + headroom for prompt

var<workgroup> q_cache: array<f32, MAX_HEAD_DIM>;
var<workgroup> scores:  array<f32, MAX_TOKENS>;
var<workgroup> red:     array<f32, WG>;

// Region mask (== reference/diffusion/mask.rs::allowed).
fn allowed(qi: u32, kj: u32, p: u32, n_swa: u32, swa: bool) -> bool {
    let q_canvas = qi >= p;
    let k_canvas = kj >= p;
    if (q_canvas) {
        if (swa) {
            return k_canvas || (kj + n_swa > p); // last n_swa-1 prompt + all canvas
        }
        return true;
    }
    // prompt query: causal over earlier prompt, never canvas
    if (k_canvas || kj > qi) { return false; }
    if (swa) { return (qi - kj) < n_swa; }
    return true;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let n  = params.n_tokens;
    let hd = params.head_dim;
    let qi = wid.x;          // query position
    let qh = wid.y;          // query head
    if (qi >= n || qh >= params.n_heads) { return; }
    let heads_per_kv = max(params.n_heads / params.n_kv_heads, 1u);
    let kvh = qh / heads_per_kv;
    let swa = params.swa_layer != 0u;

    // 1. cache the query slice.
    var d: u32 = tid;
    loop {
        if (d >= hd) { break; }
        q_cache[d] = q[(qi * params.n_heads + qh) * hd + d];
        d = d + WG;
    }
    workgroupBarrier();

    // 2. scores[kj] = (q · k[kj]) for allowed kj, else -inf.
    var kj: u32 = tid;
    loop {
        if (kj >= n) { break; }
        if (allowed(qi, kj, params.prompt_len, params.n_swa, swa)) {
            let kbase = (kj * params.n_kv_heads + kvh) * hd;
            var acc: f32 = 0.0;
            for (var i: u32 = 0u; i < hd; i = i + 1u) {
                acc = acc + q_cache[i] * k[kbase + i];
            }
            scores[kj] = acc; // scale 1.0
        } else {
            scores[kj] = -3.4e38; // -inf
        }
        kj = kj + WG;
    }
    workgroupBarrier();

    // 3. max-reduce.
    var local_max: f32 = -3.4e38;
    var m: u32 = tid;
    loop { if (m >= n) { break; } local_max = max(local_max, scores[m]); m = m + WG; }
    red[tid] = local_max;
    workgroupBarrier();
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { red[tid] = max(red[tid], red[tid + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let smax = red[0];
    workgroupBarrier();

    // 4. exp + sum-reduce.
    var local_sum: f32 = 0.0;
    var e: u32 = tid;
    loop {
        if (e >= n) { break; }
        let ex = exp(scores[e] - smax);
        scores[e] = ex;
        local_sum = local_sum + ex;
        e = e + WG;
    }
    red[tid] = local_sum;
    workgroupBarrier();
    stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv_sum = 1.0 / red[0];
    workgroupBarrier();

    // 5. out[d] = Σ_kj prob[kj] · v[kj][d].
    let obase = (qi * params.n_heads + qh) * hd;
    var od: u32 = tid;
    loop {
        if (od >= hd) { break; }
        var acc: f32 = 0.0;
        for (var t: u32 = 0u; t < n; t = t + 1u) {
            let vw = scores[t] * inv_sum;
            if (vw != 0.0) {
                acc = acc + vw * v[(t * params.n_kv_heads + kvh) * hd + od];
            }
        }
        o[obase + od] = acc;
        od = od + WG;
    }
}
