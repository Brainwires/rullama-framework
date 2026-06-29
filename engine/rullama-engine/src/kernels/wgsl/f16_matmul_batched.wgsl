// Batched f16-weight matmul: y[b, j] = Σ_i x[b, i] * W[j, i].
//
// 2D dispatch: gid.x indexes the output column (j), gid.y indexes the batch
// (b). Caller dispatches `(n.div_ceil(64), batch, 1)`. This keeps each
// dimension under the 65535 wgpu workgroup-per-dim limit even at the larger
// vision shapes (batch up to ~2560 patches, n up to 3072 ffn).

struct Params {
    k:     u32,
    n:     u32,
    batch: u32,
    _pad:  u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;   // f16 pairs per u32
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j: u32 = gid.x;
    let b: u32 = gid.y;
    if (j >= params.n || b >= params.batch) { return; }

    let half_k: u32 = params.k / 2u;
    let row_start: u32 = j * half_k;
    let x_off: u32 = b * params.k;

    var acc: f32 = 0.0;
    for (var p: u32 = 0u; p < half_k; p = p + 1u) {
        let packed: u32 = weight[row_start + p];
        let pair: vec2<f32> = unpack2x16float(packed);
        acc = acc + x[x_off + p * 2u] * pair.x + x[x_off + p * 2u + 1u] * pair.y;
    }
    y[b * params.n + j] = acc;
}
