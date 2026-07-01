// Attention backward — pass 2: produces `d_k_hist` and `d_v_hist`.
//
// Reads `d_scores` (from pass 1), `probs`, `q`, and `d_out`. One workgroup
// per (kv-head, history-position) pair; threads parallelize over head_dim.
//
//   d_k_hist[j, kv, d] = Σ_{h ∈ kv}  d_scores[h, j] · q[h, d]
//   d_v_hist[j, kv, d] = Σ_{h ∈ kv}  probs[h, j]    · d_out[h, d]
//
// No atomics — each (j, kv, d) tuple is owned by exactly one thread in
// exactly one workgroup, so writes don't race.

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

@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read>       q:        array<f32>;
@group(0) @binding(2) var<storage, read>       probs:    array<f32>;
@group(0) @binding(3) var<storage, read>       d_out:    array<f32>;
@group(0) @binding(4) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_k_hist: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_v_hist: array<f32>;

const WG: u32 = 64u;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let kv = wid.x;
    let j  = wid.y;
    if (kv >= params.n_kv_heads || j >= params.history_len) { return; }

    let head_dim = params.head_dim;
    let h_start = kv * params.heads_per_kv;
    let h_end = h_start + params.heads_per_kv;
    let kv_off = (j * params.n_kv_heads + kv) * head_dim;

    var d: u32 = tid;
    loop {
        if (d >= head_dim) { break; }
        var acc_dk: f32 = 0.0;
        var acc_dv: f32 = 0.0;
        for (var h: u32 = h_start; h < h_end; h = h + 1u) {
            let ds = d_scores[h * params.history_len + j];
            let pj = probs[h * params.history_len + j];
            let qv = q[h * head_dim + d];
            let dv = d_out[h * head_dim + d];
            acc_dk = acc_dk + ds * qv;
            acc_dv = acc_dv + pj * dv;
        }
        d_k_hist[kv_off + d] = acc_dk;
        d_v_hist[kv_off + d] = acc_dv;
        d = d + WG;
    }
}
