// GeGLU over the two halves of ONE fused gate_up vector:
//   act[i] = gelu_tanh(gu[i]) * gu[i + n_ff]
//
// The fused `ffn_gate_up_exps` expert matmul produces [2*n_ff] per token —
// gate in rows 0..n_ff, up in rows n_ff..2n_ff (matching Ollama's
// gateUp.Slice on dim 0). Same clamped tanh-GELU as geglu.wgsl.

struct Params {
    n_ff: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       gu:  array<f32>;
@group(0) @binding(2) var<storage, read_write> y:   array<f32>;

const SQRT_2_OVER_PI: f32 = 0.79788456;
const GELU_COEF_A: f32    = 0.044715;

fn gelu_tanh(g: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * g * (1.0 + GELU_COEF_A * g * g);
    let safe = clamp(inner, -10.0, 10.0);
    return 0.5 * g * (1.0 + tanh(safe));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_ff) { return; }
    y[i] = gelu_tanh(gu[i]) * gu[i + params.n_ff];
}
