// GeGLU backward.
//
// Forward:  y[i] = gelu(gate[i]) * up[i]
// Backward: d_gate[i] = dy[i] · gelu'(gate[i]) · up[i]
//           d_up[i]   = dy[i] · gelu(gate[i])
//
// gelu(g)    = 0.5 · g · (1 + erf(g / √2))
// gelu'(g)   = 0.5 · (1 + erf(g / √2)) + g · √(2/π) · exp(-g²/2) / 2
//            = 0.5 · phi + 0.5 · g · sqrt2pi · exp(-g²/2)
//
// `erf_approx` matches `gelu_split` in `reference/ops.rs` exactly
// (Abramowitz & Stegun 7.1.26), so the backward arithmetic stays in
// sync with the forward.

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

const SQRT_HALF:       f32 = 0.70710677;   // 1/√2
const SQRT_2_OVER_PI:  f32 = 0.79788456;   // √(2/π)
const A1: f32 =  0.254829592;
const A2: f32 = -0.284496736;
const A3: f32 =  1.421413741;
const A4: f32 = -1.453152027;
const A5: f32 =  1.061405429;
const P:  f32 =  0.3275911;

fn erf_approx(xx: f32) -> f32 {
    var sign: f32 = 1.0;
    var x: f32 = xx;
    if (x < 0.0) { sign = -1.0; x = -x; }
    let t = 1.0 / (1.0 + P * x);
    let v = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * exp(-x * x);
    return sign * v;
}

fn gelu(g: f32) -> f32 {
    return 0.5 * g * (1.0 + erf_approx(g * SQRT_HALF));
}

fn gelu_prime(g: f32) -> f32 {
    let phi  = 1.0 + erf_approx(g * SQRT_HALF);
    let dphi = SQRT_2_OVER_PI * exp(-0.5 * g * g);
    return 0.5 * phi + 0.5 * g * dphi;
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
