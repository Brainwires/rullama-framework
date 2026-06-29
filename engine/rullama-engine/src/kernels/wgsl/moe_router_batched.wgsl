// Batched Gemma 4 MoE router: one workgroup per token POSITION, each routing
// that position independently (vs the single-position moe_router.wgsl). For the
// DiffusionGemma 256-token canvas this is one dispatch instead of 256.
//
// Per position p: unweighted RMSNorm(x[p]) → ×1/√d → ×optional scale → expert
// scores (routerᵀ·x) → softmax over all experts → top-k → renorm. Writes
// out_ids[p*top_k + s] (u32) + out_w[p*top_k + s] (f32). Identical math to
// moe_router.wgsl (parity-tested there); this just adds the position index +
// per-position input/output offsets.

struct Params {
    n_pos:     u32,
    d_model:   u32,
    n_experts: u32,
    top_k:     u32,
    eps:       f32,
    has_scale: u32,
    _pad0:     u32,
    _pad1:     u32,
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       x:       array<f32>; // [n_pos, d_model]
@group(0) @binding(2) var<storage, read>       scale:   array<f32>; // [d_model] (shared)
@group(0) @binding(3) var<storage, read>       router:  array<f32>; // [n_experts, d_model]
@group(0) @binding(4) var<storage, read_write> out_ids: array<u32>; // [n_pos, top_k]
@group(0) @binding(5) var<storage, read_write> out_w:   array<f32>; // [n_pos, top_k]

const WG: u32 = 64u;
const MAX_EXPERTS: u32 = 256u;
const MAX_TOPK: u32 = 16u;

var<workgroup> tile:   array<f32, WG>;
var<workgroup> scores: array<f32, MAX_EXPERTS>;
var<workgroup> inv_rms_shared: f32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let pos = wid.x;
    if (pos >= params.n_pos) { return; }
    let n = params.d_model;
    let n_exp = params.n_experts;
    let x_off = pos * n;

    // 1. sum-of-squares for the unweighted RMSNorm.
    var local_sumsq: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= n) { break; }
        let val = x[x_off + i];
        local_sumsq = local_sumsq + val * val;
        i = i + WG;
    }
    tile[tid] = local_sumsq;
    workgroupBarrier();
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { tile[tid] = tile[tid] + tile[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        inv_rms_shared = 1.0 / sqrt(tile[0] / f32(n) + params.eps);
    }
    workgroupBarrier();

    // 2+3. per-expert dot of the normed/scaled hidden state.
    let inv_rms = inv_rms_shared;
    let inv_sqrt_d = 1.0 / sqrt(f32(n));
    var e: u32 = tid;
    loop {
        if (e >= n_exp) { break; }
        var acc: f32 = 0.0;
        let row = e * n;
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            var val = x[x_off + j] * inv_rms * inv_sqrt_d;
            if (params.has_scale != 0u) { val = val * scale[j]; }
            acc = acc + val * router[row + j];
        }
        scores[e] = acc;
        e = e + WG;
    }
    workgroupBarrier();

    // 4. softmax → top-k → renorm (thread 0; n_exp ≤ 256, k ≤ 16).
    if (tid == 0u) {
        var m: f32 = scores[0];
        for (var vv: u32 = 1u; vv < n_exp; vv = vv + 1u) { m = max(m, scores[vv]); }
        var z: f32 = 0.0;
        for (var vv: u32 = 0u; vv < n_exp; vv = vv + 1u) {
            let pr = exp(scores[vv] - m);
            scores[vv] = pr;
            z = z + pr;
        }
        for (var vv: u32 = 0u; vv < n_exp; vv = vv + 1u) { scores[vv] = scores[vv] / z; }
        let k = min(params.top_k, min(n_exp, MAX_TOPK));
        let o_off = pos * params.top_k;
        var sum_sel: f32 = 0.0;
        for (var s: u32 = 0u; s < k; s = s + 1u) {
            var best: u32 = 0u;
            var best_p: f32 = -1.0;
            for (var vv: u32 = 0u; vv < n_exp; vv = vv + 1u) {
                if (scores[vv] > best_p) { best_p = scores[vv]; best = vv; }
            }
            out_ids[o_off + s] = best;
            out_w[o_off + s] = best_p;
            sum_sel = sum_sel + best_p;
            scores[best] = -2.0;
        }
        for (var s: u32 = 0u; s < k; s = s + 1u) {
            out_w[o_off + s] = out_w[o_off + s] / sum_sel;
        }
    }
}
