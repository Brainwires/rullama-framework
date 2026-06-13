// Gemma 4 MoE router — one fused dispatch (one workgroup):
//
//   1. unweighted RMSNorm of the raw post-attention hidden state x
//   2. ×1/√d_model, ×optional learned per-channel scale (ffn_gate_inp.scale)
//   3. expert scores = routerᵀ · normed_x   (router = ffn_gate_inp.weight,
//      GGUF [d_model, n_experts] row-major: element (i, e) at e*d_model + i)
//   4. softmax over ALL experts → top-k by probability → renormalize the k
//      selected weights to sum to 1
//
// Mirrors Ollama's TextRouter.Forward + TextMoEBlock weight selection
// (model/models/gemma4/model_text.go) and `route`/`softmax_topk_renorm` in
// reference/moe.rs (the CPU oracle this kernel is parity-tested against).
//
// Outputs land GPU-resident (no CPU readback per token): expert_ids[k] +
// expert_weights[k], consumed by moe_expert_matmul_* (which index the stacked
// expert weight buffer by ids[slot] on-GPU, MulmatID-style).

struct Params {
    d_model:   u32,
    n_experts: u32,
    top_k:     u32,
    eps:       f32,
    has_scale: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       x:       array<f32>;
@group(0) @binding(2) var<storage, read>       scale:   array<f32>;
@group(0) @binding(3) var<storage, read>       router:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_w:   array<f32>;

const WG: u32 = 128u;
const MAX_EXPERTS: u32 = 256u;
const MAX_TOPK: u32 = 16u;

var<workgroup> tile:   array<f32, WG>;
var<workgroup> scores: array<f32, MAX_EXPERTS>;
var<workgroup> inv_rms_shared: f32;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) tid: u32) {
    let n = params.d_model;
    let n_exp = params.n_experts;

    // ---- 1. sum-of-squares reduction for the unweighted RMSNorm ----
    var local_sumsq: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= n) { break; }
        let v = x[i];
        local_sumsq = local_sumsq + v * v;
        i = i + WG;
    }
    tile[tid] = local_sumsq;
    workgroupBarrier();
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            tile[tid] = tile[tid] + tile[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        inv_rms_shared = 1.0 / sqrt(tile[0] / f32(n) + params.eps);
    }
    workgroupBarrier();

    // ---- 2+3. per-expert dot of the normed/scaled hidden state ----
    let inv_rms = inv_rms_shared;
    let inv_sqrt_d = 1.0 / sqrt(f32(n));
    var e: u32 = tid;
    loop {
        if (e >= n_exp) { break; }
        var acc: f32 = 0.0;
        let row = e * n;
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            var v = x[j] * inv_rms * inv_sqrt_d;
            if (params.has_scale != 0u) {
                v = v * scale[j];
            }
            acc = acc + v * router[row + j];
        }
        scores[e] = acc;
        e = e + WG;
    }
    workgroupBarrier();

    // ---- 4. softmax → top-k → renorm (serial in thread 0; n_experts ≤ 256
    //         and k ≤ 16, so this is trivial against the dots above) ----
    if (tid == 0u) {
        var m: f32 = scores[0];
        for (var v: u32 = 1u; v < n_exp; v = v + 1u) {
            m = max(m, scores[v]);
        }
        var z: f32 = 0.0;
        for (var v: u32 = 0u; v < n_exp; v = v + 1u) {
            let p = exp(scores[v] - m);
            scores[v] = p;
            z = z + p;
        }
        // probabilities
        for (var v: u32 = 0u; v < n_exp; v = v + 1u) {
            scores[v] = scores[v] / z;
        }
        // iterative top-k selection (ties → lower index, matching the CPU
        // oracle's stable sort)
        let k = min(params.top_k, min(n_exp, MAX_TOPK));
        var sum_sel: f32 = 0.0;
        for (var s: u32 = 0u; s < k; s = s + 1u) {
            var best: u32 = 0u;
            var best_p: f32 = -1.0;
            for (var v: u32 = 0u; v < n_exp; v = v + 1u) {
                if (scores[v] > best_p) {
                    best_p = scores[v];
                    best = v;
                }
            }
            out_ids[s] = best;
            out_w[s] = best_p;
            sum_sel = sum_sel + best_p;
            scores[best] = -2.0; // remove from contention
        }
        for (var s: u32 = 0u; s < k; s = s + 1u) {
            out_w[s] = out_w[s] / sum_sel;
        }
    }
}
