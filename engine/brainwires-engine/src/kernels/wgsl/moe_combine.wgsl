// MoE weighted combine: sum the k per-slot expert down-projections into one
// vector, applying the router weight and the optional per-expert down scale:
//
//   y[i] = Σ_s expert_weights[s] · down_scale[expert_ids[s]] · slots[s*d + i]
//
// `slots` is the concatenated per-slot down outputs [k, d_model] (slot-major).
// down_scale indexes by the ORIGINAL expert id (ffn_down_exps.scale is
// [n_experts]); has_down_scale=0 uses 1.0. Mirrors the combine in
// reference/moe.rs::moe_experts.

struct Params {
    d_model:        u32,
    top_k:          u32,
    has_down_scale: u32,
    _pad0:          u32,
}

@group(0) @binding(0) var<uniform>             params:     Params;
@group(0) @binding(1) var<storage, read>       slots:      array<f32>;
@group(0) @binding(2) var<storage, read>       ids:        array<u32>;
@group(0) @binding(3) var<storage, read>       weights:    array<f32>;
@group(0) @binding(4) var<storage, read>       down_scale: array<f32>;
@group(0) @binding(5) var<storage, read_write> y:          array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.d_model) { return; }
    var acc: f32 = 0.0;
    for (var s: u32 = 0u; s < params.top_k; s = s + 1u) {
        var sc: f32 = 1.0;
        if (params.has_down_scale != 0u) {
            sc = down_scale[ids[s]];
        }
        acc = acc + weights[s] * sc * slots[s * params.d_model + i];
    }
    y[i] = acc;
}
