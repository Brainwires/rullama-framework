// 2-D transpose: in[rows, cols] (row-major) → out[cols, rows]. One thread per element.
// Used to bridge channel-major [C,T] conv activations and row-major [T,C] matmul / LN
// inputs in the Kokoro GPU forward (e.g. the TextEncoder channel-axis LayerNorm).

struct Params { rows: u32, cols: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.rows * params.cols) { return; }
    let r = idx / params.cols;
    let c = idx % params.cols;
    y[c * params.rows + r] = x[idx];
}
