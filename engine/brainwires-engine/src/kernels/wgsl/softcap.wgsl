// Logit softcap: y[i] = cap * tanh(x[i] / cap), elementwise.
//
// One thread per element. Used at the very end of the forward pass to clamp logits
// (Gemma 4 cap = 30.0).

struct Params {
    n:    u32,
    cap:  f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let cap = params.cap;
    if (cap <= 0.0) {
        y[i] = x[i];
        return;
    }
    y[i] = cap * tanh(x[i] / cap);
}
