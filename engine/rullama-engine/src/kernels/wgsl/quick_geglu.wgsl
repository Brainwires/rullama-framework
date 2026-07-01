// Quick GeGLU split: y[i] = quick_gelu(gate[i]) * up[i].
//
// QuickGELU = x * sigmoid(1.702 * x). Used by the Gemma 4 vision tower MLP
// (model_vision.go::VisionMLP.Forward → ggml_geglu_quick_split). Distinct from
// the text path's exact erf-based GELU (geglu.wgsl) — the ViT was trained with
// QuickGELU and using the exact GELU here breaks parity.

struct Params {
    n:  u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       gate: array<f32>;
@group(0) @binding(2) var<storage, read>       up:   array<f32>;
@group(0) @binding(3) var<storage, read_write> y:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 4194240u + gid.x;
    if (i >= params.n) { return; }
    let g = gate[i];
    let s = 1.0 / (1.0 + exp(-1.702 * g));
    y[i] = (g * s) * up[i];
}
