// GeGLU backward.
//
// Forward (tanh-approximation, matching ggml's `ggml_geglu_split`):
//   gelu(g) = 0.5 · g · (1 + tanh(φ(g)))
//   φ(g)    = √(2/π) · g · (1 + α·g²),   α = 0.044715
//   gelu'(g) = 0.5 · (1 + tanh(φ)) + 0.5 · g · (1 - tanh²(φ)) · φ'(g)
//   φ'(g)   = √(2/π) · (1 + 3·α·g²)
// Backward: d_gate[i] = dy[i] · gelu'(gate[i]) · up[i]
//           d_up[i]   = dy[i] · gelu(gate[i])

struct Params {
    n:     u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       gate:   array<f32>;
@group(0) @binding(2) var<storage, read>       up:     array<f32>;
@group(0) @binding(3) var<storage, read>       dy:     array<f32>;
@group(0) @binding(4) var<storage, read_write> d_gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_up:   array<f32>;

const SQRT_2_OVER_PI: f32 = 0.79788456;
const GELU_COEF_A:    f32 = 0.044715;

// Clamp |inner| before tanh — mirrors the forward kernel (geglu.wgsl).
// Metal computes tanh via exp(2x) which overflows for large |inner|, returning
// NaN; gate values can reach ±40 in trained checkpoints, producing inner ≈ 2300.
// tanh(±10) already rounds to ±1 in f32 so the clamp is numerically lossless.
fn gelu(g: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * g * (1.0 + GELU_COEF_A * g * g);
    let safe  = clamp(inner, -10.0, 10.0);
    return 0.5 * g * (1.0 + tanh(safe));
}

fn gelu_prime(g: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * g * (1.0 + GELU_COEF_A * g * g);
    let safe  = clamp(inner, -10.0, 10.0);
    let t     = tanh(safe);
    let dphi  = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEF_A * g * g);
    return 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * dphi;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let g    = gate[i];
    let u    = up[i];
    let dy_i = dy[i];
    d_gate[i] = dy_i * gelu_prime(g) * u;
    d_up[i]   = dy_i * gelu(g);
}
