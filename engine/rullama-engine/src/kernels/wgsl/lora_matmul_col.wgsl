// Tiny f32 column-axis matmul: y = scale * Wᵀ @ x  (or y += scale · Wᵀ @ x).
//
// W shape: [n, r] row-major (same physical layout as `lora_matmul_row`'s W,
// but here we iterate by column to compute the transposed product).
//   W[j, p] = w[j * r + p]
//   (Wᵀ @ x)[p] = Σ_j W[j, p] · x[j]
//
// Used by LoRA backward:
//   u  = Bᵀ @ dy           (W=B shape [n, r], scale=1, accumulate=0)
//   dx += s · Aᵀ @ u       (W=A shape [r, k]…wait, A is [r, k] row-major:
//                           A[p, k] = a[p * k + k_idx]. Transposed (Aᵀ)[k, p]
//                           means iterating over p for each output k:
//                           (Aᵀ @ u)[k] = Σ_p A[p, k] · u[p] = Σ_p a[p*k + k_idx] · u[p]
//                           — a strided read, not a contiguous row. The
//                           dispatcher passes A as-is with r=`n` and
//                           k_out=`k_dim` to reuse this kernel:
//                           Wᵀ-with-outer-dim-r → output length r=r_dim.
//                           See dispatch::lora_matmul_col_chained docs.)

struct Params {
    outer:      u32,   // number of "j" iterations (= n in B^T @ dy; = r in A^T @ u)
    inner:      u32,   // length of output y (= r in B^T @ dy; = k in A^T @ u)
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
    let p = gid.x;
    if (p >= params.inner) { return; }
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < params.outer; j = j + 1u) {
        acc = acc + w[j * params.inner + p] * x[j];
    }
    let v = params.scale * acc;
    if (params.accumulate != 0u) {
        y[p] = y[p] + v;
    } else {
        y[p] = v;
    }
}
