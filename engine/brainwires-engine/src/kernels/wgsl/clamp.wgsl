// In-place elementwise clamp: x[i] = max(lo, min(hi, x[i])).
//
// Used for ClippableLinear in the Gemma 4 vision tower — the F16 weight storage
// path overflows on some inputs without input/output clamping (see Ollama's
// model_vision.go::ClippableLinear). Each linear has its own (input_min, input_max,
// output_min, output_max) tuple loaded from `v.clamp_data`.

struct Params {
    n:  u32,
    lo: f32,
    hi: f32,
    _p: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> x:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 2D dispatch: gid.y * (65535 * 64) + gid.x. The constant covers a full
    // x-row so callers can dispatch (min(wg_x, 65535), wg_y, 1) without
    // passing the actual stride. WebGPU mandates max_compute_workgroups_per_dimension >= 65535.
    let i = gid.y * 4194240u + gid.x;
    if (i >= params.n) { return; }
    x[i] = clamp(x[i], params.lo, params.hi);
}
