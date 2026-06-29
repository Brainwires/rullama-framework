// Conformer block-local attention (Gemma 4 audio).
//
// Mirrors the CPU oracle in `src/multimodal/audio.rs::forward_attention`,
// specifically the inner loop over (chunk, query, head) that consumes the
// already-projected Q/K/V plus the projected positional-bias vectors.
//
// One workgroup per (padded query position, head). Each workgroup computes
// the `head_dim` outputs for that (q, h) by:
//   1. Caching the query slice into workgroup memory.
//   2. For each of `context_size` context positions: parallel-reducing the
//      content-content (q · k) and content-position (q · pos_proj) dot
//      products, applying the `tanh(score/cap) * cap` softcap, masking
//      out causally-invalid context positions.
//   3. Computing softmax over the context dimension (single thread for
//      simplicity — context_size is small, typically 24).
//   4. Computing the weighted V sum, writing one head_dim slice to attn_out.
//
// The kernel assumes Q is already per-dim-scaled and K is already
// k-scale-multiplied (per the CPU oracle layout). Positional bias is
// likewise pre-projected through `linear_pos`.
//
// Inputs:
//   q_pad     : [padded_len, hidden]                        — padded queries
//   k_padded  : [pad_left + padded_len + pad_right, hidden] — padded keys
//   v_padded  : same shape as k_padded                      — padded values
//   pos_proj  : [max_span, hidden]                          — projected positions
// Output:
//   attn_out  : [padded_len, hidden]
//
// Notes:
//   * `head_dim` is fixed at 128 (Gemma 4 audio: hidden=1024, n_heads=8).
//   * `context_size` = max_past + chunk_size + max_future (typically 24).
//   * `max_span` = max_past + max_future + 1 (typically 13).

struct Params {
    seq:          u32,
    padded_len:   u32,
    hidden:       u32,
    n_heads:      u32,
    head_dim:     u32,
    chunk_size:   u32,
    context_size: u32,
    max_span:     u32,
    max_past:     u32,
    max_future:   u32,
    pad_left:     u32,
    logit_cap:    f32,
}

@group(0) @binding(0) var<uniform>             p:        Params;
@group(0) @binding(1) var<storage, read>       q_pad:    array<f32>;
@group(0) @binding(2) var<storage, read>       k_padded: array<f32>;
@group(0) @binding(3) var<storage, read>       v_padded: array<f32>;
@group(0) @binding(4) var<storage, read>       pos_proj: array<f32>;
@group(0) @binding(5) var<storage, read_write> attn_out: array<f32>;

const HEAD_DIM:    u32 = 128u;   // Gemma 4 audio head_dim — hard-coded.
const MAX_CONTEXT: u32 = 32u;    // Headroom over context_size = 24.
const NEG_LARGE:   f32 = -1e30;

var<workgroup> sh_q:      array<f32, HEAD_DIM>;
var<workgroup> sh_red:    array<f32, HEAD_DIM>;
var<workgroup> sh_logits: array<f32, MAX_CONTEXT>;

// Tree reduction across a workgroup of HEAD_DIM threads. `val` is each
// thread's contribution; result is broadcast (every thread reads sh_red[0]).
fn workgroup_reduce(val: f32, tid: u32) -> f32 {
    sh_red[tid] = val;
    workgroupBarrier();
    var stride: u32 = HEAD_DIM / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_red[tid] = sh_red[tid] + sh_red[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return sh_red[0];
}

// Dispatch:
//   workgroups = (padded_len, n_heads)
//   workgroup_size = (HEAD_DIM,)
@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let row = wg_id.x;          // 0..padded_len
    let h   = wg_id.y;          // 0..n_heads
    let tid = lid.x;            // 0..head_dim
    // No `if (tid >= p.head_dim) return` guard: HEAD_DIM and head_dim are
    // both 128 (Gemma 4 audio) and the dispatch matches @workgroup_size(128).
    // An early return ahead of workgroupBarrier() trips Safari/Tint's
    // uniformity analysis ("workgroupBarrier must only be called from
    // uniform control flow"), since the validator can't prove the guard
    // is dead.

    let u = row / p.chunk_size;     // chunk index
    let r = row % p.chunk_size;     // position within chunk

    // Cache q_pad[row, h, :] into workgroup memory.
    let q_off = row * p.hidden + h * p.head_dim;
    sh_q[tid] = q_pad[q_off + tid];
    workgroupBarrier();

    let q_val = sh_q[tid];

    // Phase 1: compute logits[c] for c in 0..context_size.
    for (var c: u32 = 0u; c < p.context_size; c = c + 1u) {
        // Causal-valid mask. `actual_t` is the absolute sequence position
        // being attended to (negative when in the left zero-pad region).
        let actual_t_signed = i32(u * p.chunk_size) + i32(c) - i32(p.pad_left);
        let valid_seq    = (actual_t_signed >= 0) && (actual_t_signed < i32(p.seq));
        let causal_ok    = (c >= r) && (c <= r + p.max_past + p.max_future);
        let invalid      = !(valid_seq && causal_ok);

        // Content-content score: q · k_padded[k_off..k_off + head_dim].
        let k_off = (u * p.chunk_size + c) * p.hidden + h * p.head_dim;
        let k_val = k_padded[k_off + tid];
        let ac    = workgroup_reduce(q_val * k_val, tid);

        // Content-position score: q · pos_proj[p_signed * hidden + h*head_dim..]
        // where p_signed = max_past + r - c. May lie outside [0, max_span);
        // in that case bd is 0.
        let p_signed_i = i32(p.max_past) + i32(r) - i32(c);
        var bd_partial: f32 = 0.0;
        if (p_signed_i >= 0 && p_signed_i < i32(p.max_span)) {
            let pos_off = u32(p_signed_i) * p.hidden + h * p.head_dim;
            bd_partial = q_val * pos_proj[pos_off + tid];
        }
        let bd = workgroup_reduce(bd_partial, tid);

        if (tid == 0u) {
            if (invalid) {
                sh_logits[c] = NEG_LARGE;
            } else {
                let raw   = ac + bd;
                let score = tanh(raw / p.logit_cap) * p.logit_cap;
                sh_logits[c] = score;
            }
        }
        workgroupBarrier();
    }

    // Phase 2: softmax over the context dimension. Single thread.
    if (tid == 0u) {
        var max_logit: f32 = NEG_LARGE;
        for (var c: u32 = 0u; c < p.context_size; c = c + 1u) {
            if (sh_logits[c] > max_logit) {
                max_logit = sh_logits[c];
            }
        }
        var sum_exp: f32 = 0.0;
        for (var c: u32 = 0u; c < p.context_size; c = c + 1u) {
            if (sh_logits[c] <= NEG_LARGE * 0.5) {
                sh_logits[c] = 0.0;
            } else {
                let e = exp(sh_logits[c] - max_logit);
                sh_logits[c] = e;
                sum_exp = sum_exp + e;
            }
        }
        let inv = select(0.0, 1.0 / sum_exp, sum_exp > 0.0);
        for (var c: u32 = 0u; c < p.context_size; c = c + 1u) {
            sh_logits[c] = sh_logits[c] * inv;
        }
    }
    workgroupBarrier();

    // Phase 3: weighted V sum. Each thread computes one output dim.
    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < p.context_size; c = c + 1u) {
        let w = sh_logits[c];
        if (w != 0.0) {
            let v_off = (u * p.chunk_size + c) * p.hidden + h * p.head_dim;
            acc = acc + w * v_padded[v_off + tid];
        }
    }
    let out_off = row * p.hidden + h * p.head_dim;
    attn_out[out_off + tid] = acc;
}
