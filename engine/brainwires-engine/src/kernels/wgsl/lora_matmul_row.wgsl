// Tiny f32 row-major matmul: y = scale * W @ x  (or y += scale * W @ x).
//
// W shape: [n, k] row-major, so W[j, i] = w[j * k + i].
// Used as the building block for LoRA forward correction:
//   z = A @ x          (W=A shape [r, k], scale=1, accumulate=0)
//   y += s · B @ z     (W=B shape [n, r], scale=alpha/r, accumulate=1)
//
// Naive — one thread per output row, loop over k. Adequate for LoRA's
// small dimensions (r ≤ 64; n, k ≤ a few thousand).

struct Params {
    k:          u32,
    n:          u32,
    accumulate: u32,
    _pad:       u32,
    scale:      f32,
    _pad2:      u32,
    _pad3:      u32,
    _pad4:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       w:      array<f32>;
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.n) { return; }
    var acc: f32 = 0.0;
    let row_off = j * params.k;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        acc = acc + w[row_off + i] * x[i];
    }
    let v = params.scale * acc;
    if (params.accumulate != 0u) {
        y[j] = y[j] + v;
    } else {
        y[j] = v;
    }
}
