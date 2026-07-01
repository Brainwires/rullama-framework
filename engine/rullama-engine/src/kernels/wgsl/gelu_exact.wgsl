// Elementwise exact GELU, in-place. y = y·½(1+erf(y/√2)).
// erf via Abramowitz-Stegun 7.1.26 (~1e-7) — matches the f32 diffusion oracle's gelu_exact.
// Used by the StyleTTS2 style-diffusion denoiser (FFN + to_time/to_features/to_mapping).
// NOT the tanh `gelu_new` the ALBERT path uses.

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if (i >= params.n) { return; }
    let x = y[i];
    let z = x * 0.70710677;           // x/√2
    let az = abs(z);
    let t = 1.0 / (1.0 + 0.3275911 * az);
    let poly = ((((1.0614054 * t - 1.4531520) * t + 1.4214137) * t - 0.28449674) * t + 0.25482960) * t;
    let y_ = 1.0 - poly * exp(-az * az);
    let erf = select(-y_, y_, z >= 0.0);
    y[i] = x * 0.5 * (1.0 + erf);
}
