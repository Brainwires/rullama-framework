// Backward of `y = matmul_q4_0(W, x)` with respect to x.
//
// Forward (q4_0_dequant_matmul.wgsl): y[j] = Σ_i x[i] * dequant(W)[j, i]
// Backward:                            dx[i] = Σ_j dy[j] * dequant(W)[j, i]
//
// W is row-major [n, k] with each row packed into k/32 Q4_0 blocks of 18 bytes
// (32 elements each). One workgroup per block-row of k (32 output elements);
// each thread owns one i within that block-row, loops over j ∈ [j_start, j_end),
// reads its own nibble from the (j, block_row) Q4_0 block, and accumulates.
//
// W is frozen (LoRA convention) — there is no weight gradient. The j_start/j_end/
// accumulate params support the vocab-axis tiling used by the output-proj
// backward; non-tiled callers pass j_start=0, j_end=n, accumulate=0.

struct Params {
    k: u32,
    n: u32,
    j_start: u32,
    j_end: u32,
    accumulate: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;
@group(0) @binding(2) var<storage, read>       dy:     array<f32>;
@group(0) @binding(3) var<storage, read_write> dx:     array<f32>;

const BLOCK_ELEMS: u32 = 32u;
const BLOCK_BYTES: u32 = 18u;

fn read_byte(byte_off: u32) -> u32 {
    let u32_idx = byte_off >> 2u;
    let shift   = (byte_off & 3u) << 3u;
    return (weight[u32_idx] >> shift) & 0xFFu;
}

fn read_f16_as_f32(byte_off: u32) -> f32 {
    let lo = read_byte(byte_off);
    let hi = read_byte(byte_off + 1u);
    let packed: u32 = lo | (hi << 8u);
    return unpack2x16float(packed).x;
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let block_row: u32 = wg.x;
    let tid:       u32 = lid.x;
    let i:         u32 = block_row * BLOCK_ELEMS + tid;
    if (i >= params.k) { return; }

    let n_blocks:  u32 = params.k / BLOCK_ELEMS;
    let row_bytes: u32 = n_blocks * BLOCK_BYTES;

    // Q4_0 packs the first 16 elements in the low nibbles of qs[0..16] and the
    // next 16 in the high nibbles. tid → which byte + which nibble (constant
    // for the life of this thread).
    let nibble_hi: bool = tid >= 16u;
    let qs_idx: u32 = select(tid, tid - 16u, nibble_hi);
    let qs_local_off: u32 = 2u + qs_idx;

    var acc: f32 = 0.0;
    for (var j: u32 = params.j_start; j < params.j_end; j = j + 1u) {
        let block_off: u32 = j * row_bytes + block_row * BLOCK_BYTES;
        let d: f32 = read_f16_as_f32(block_off + 0u);
        let q = read_byte(block_off + qs_local_off);
        let nibble: f32 = select(f32(q & 0xFu), f32(q >> 4u), nibble_hi);
        let w_ij: f32 = (nibble - 8.0) * d;
        acc = acc + w_ij * dy[j];
    }

    if (params.accumulate == 0u) {
        dx[i] = acc;
    } else {
        dx[i] = dx[i] + acc;
    }
}
