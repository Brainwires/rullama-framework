// GeGLU split: y[i] = gelu(gate[i]) * up[i], elementwise.
//
// Mirrors `geglu_split` in `src/reference/ops.rs`: tanh-approximation GELU,
// matching ggml's `ggml_geglu_split` (the variant Ollama's `Tensor.GELU` calls).
// ggml has a separate `ggml_geglu_erf_split` for the exact erf form; we use the
// tanh form here for parity with Ollama.

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       gate: array<f32>;
@group(0) @binding(2) var<storage, read>       up:   array<f32>;
@group(0) @binding(3) var<storage, read_write> y:    array<f32>;

const SQRT_2_OVER_PI: f32 = 0.79788456;
const GELU_COEF_A: f32    = 0.044715;

fn gelu_tanh(g: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * g * (1.0 + GELU_COEF_A * g * g);
    // Clamp |inner| to a safe range — WGSL's `tanh` can yield NaN for very
    // large |inner| on some backends (Metal computes via `exp(2x)` which
    // overflows). tanh(±10) already rounds to ±1 in f32, so clamping is
    // numerically lossless within f32 precision.
    let safe = clamp(inner, -10.0, 10.0);
    return 0.5 * g * (1.0 + tanh(safe));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    y[i] = gelu_tanh(gate[i]) * up[i];
}
