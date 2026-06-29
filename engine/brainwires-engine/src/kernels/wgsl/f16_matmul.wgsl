// f16 weight matmul: y[j] = Σ_i x[i] * W[j, i]
//
// Weight layout: row-major, each row j has length params.k. F16 elements packed two
// per u32 in little-endian bit order. We require params.k % 2 == 0 (true for every
// shape in Gemma 4 — d_model=1536, ffn_inter=6144, head_dim*n_heads=2048 are all even).
//
// One thread per output element. Each thread reads x once (cached in registers via
// the loop) and walks the j-th row of W in order.

struct Params {
    k: u32,        // input dim
    n: u32,        // output dim (= number of output elements)
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;   // f16 pairs packed per u32
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j: u32 = gid.x;
    if (j >= params.n) { return; }

    let half_k: u32 = params.k / 2u;
    let row_start: u32 = j * half_k;

    var acc: f32 = 0.0;
    for (var p: u32 = 0u; p < half_k; p = p + 1u) {
        let packed: u32 = weight[row_start + p];
        let pair: vec2<f32> = unpack2x16float(packed);
        acc = acc + x[p * 2u] * pair.x + x[p * 2u + 1u] * pair.y;
    }
    y[j] = acc;
}
