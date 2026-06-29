// Batched GeGLU over the two halves of each row of a fused gate_up buffer.
//   act[ps, i] = gelu_tanh(gu[ps, i]) * gu[ps, i + n_ff]
// gu is [rows, 2*n_ff], act is [rows, n_ff]. One thread per (row, i). Same
// clamped tanh-GELU as moe_geglu_halves.wgsl; for the diffusion canvas, rows
// = n_pos*top_k.

struct Params {
    rows: u32,
    n_ff: u32,
    _p0:  u32,
    _p1:  u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       gu: array<f32>;
@group(0) @binding(2) var<storage, read_write> y:  array<f32>;

const SQRT_2_OVER_PI: f32 = 0.79788456;
const GELU_COEF_A: f32    = 0.044715;

fn gelu_tanh(g: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * g * (1.0 + GELU_COEF_A * g * g);
    let safe = clamp(inner, -10.0, 10.0);
    return 0.5 * g * (1.0 + tanh(safe));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = gid.x;
    let row = wid.y;
    if (i >= params.n_ff || row >= params.rows) { return; }
    let gu_base = row * 2u * params.n_ff;
    y[row * params.n_ff + i] = gelu_tanh(gu[gu_base + i]) * gu[gu_base + i + params.n_ff];
}
