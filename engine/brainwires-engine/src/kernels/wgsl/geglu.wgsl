// GeGLU split: y[i] = gelu(gate[i]) * up[i], elementwise.
//
// Mirrors `geglu_split` in `src/reference/ops.rs` exactly: erf-based GELU using
// Abramowitz & Stegun 7.1.26, so f32 outputs are bit-identical to the CPU oracle
// modulo accumulation-order differences (none here — pure elementwise).

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

const SQRT_HALF: f32 = 0.70710677;
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
    let t  = 1.0 / (1.0 + P * x);
    let v  = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * exp(-x * x);
    return sign * v;
}

fn gelu_exact(g: f32) -> f32 {
    return 0.5 * g * (1.0 + erf_approx(g * SQRT_HALF));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    y[i] = gelu_exact(gate[i]) * up[i];
}
