// Batched MoE weighted combine for the DiffusionGemma canvas: per position,
// sum the top_k expert down-projections, applying the router weight and the
// optional per-expert down scale.
//
//   y[pos, i] = Σ_s weights[pos*top_k+s] · down_scale[ids[pos*top_k+s]]
//                    · slots[(pos*top_k+s)*d_model + i]
//
// `slots` is the batched expert down output [n_pos*top_k, d_model]. One thread
// per (position, i). Same math as moe_combine.wgsl, batched over positions.

struct Params {
    n_pos:          u32,
    d_model:        u32,
    top_k:          u32,
    has_down_scale: u32,
}

@group(0) @binding(0) var<uniform>             params:     Params;
@group(0) @binding(1) var<storage, read>       slots:      array<f32>;
@group(0) @binding(2) var<storage, read>       ids:        array<u32>;
@group(0) @binding(3) var<storage, read>       weights:    array<f32>;
@group(0) @binding(4) var<storage, read>       down_scale: array<f32>;
@group(0) @binding(5) var<storage, read_write> y:          array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = gid.x;
    let pos = wid.y;
    if (i >= params.d_model || pos >= params.n_pos) { return; }
    var acc: f32 = 0.0;
    let base = pos * params.top_k;
    for (var s: u32 = 0u; s < params.top_k; s = s + 1u) {
        let ps = base + s;
        var sc: f32 = 1.0;
        if (params.has_down_scale != 0u) { sc = down_scale[ids[ps]]; }
        acc = acc + weights[ps] * sc * slots[ps * params.d_model + i];
    }
    y[pos * params.d_model + i] = acc;
}
