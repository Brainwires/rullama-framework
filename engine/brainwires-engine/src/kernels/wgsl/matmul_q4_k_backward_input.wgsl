// Backward of `y = matmul_q4_k(W, x)` with respect to x.
//
// Forward (q4_k_dequant_matmul.wgsl): y[j] = Σ_i x[i] * dequant(W)[j, i]
// Backward:                            dx[i] = Σ_j dy[j] * dequant(W)[j, i]
//
// W is stored as in the forward kernel: row-major [n, k] with each
// "row" packed into k/256 Q4_K super-blocks of 144 bytes (256 elements
// each). For backward, we want one output element per thread along the
// k axis, summing over the n axis — the access pattern is column-strided
// through the storage.
//
// Layout strategy: one workgroup per block-row of k (a contiguous group
// of 256 output elements). Each thread within the workgroup handles one
// i value within that block-row, loops over j ∈ [0, n), reads its own
// element from the (j, block_row) Q4_K block, and accumulates into a
// thread-local sum. After the loop, each thread writes one dx[i].
//
// W is frozen (LoRA convention) — there is no weight gradient.

struct Params {
    k: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;
@group(0) @binding(2) var<storage, read>       dy:     array<f32>;
@group(0) @binding(3) var<storage, read_write> dx:     array<f32>;

const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 144u;

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

@compute @workgroup_size(256)
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

    // tid → (chunk c, position p within chunk, is index into scales, low/high
    // nibble flag, qs offset within block). All constant for the life of
    // this thread.
    let c: u32 = tid / 64u;
    let p: u32 = tid % 64u;
    let nibble_hi: bool = p >= 32u;
    let is_idx: u32 = 2u * c + select(0u, 1u, nibble_hi);
    let l: u32 = p % 32u;
    let qs_local_off: u32 = 16u + c * 32u + l;

    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < params.n; j = j + 1u) {
        let block_off: u32 = j * row_bytes + block_row * BLOCK_BYTES;

        let d:    f32 = read_f16_as_f32(block_off + 0u);
        let dmin: f32 = read_f16_as_f32(block_off + 2u);

        // Inline `get_scale_min_k4(is_idx, scales[12])` for this thread's
        // (scale, min) pair only.
        var sc: u32;
        var mn: u32;
        if (is_idx < 4u) {
            sc = read_byte(block_off + 4u + is_idx) & 63u;
            mn = read_byte(block_off + 4u + is_idx + 4u) & 63u;
        } else {
            let b_45  = read_byte(block_off + 4u + is_idx + 4u);
            let b_lo  = read_byte(block_off + 4u + (is_idx - 4u));
            let b_self = read_byte(block_off + 4u + is_idx);
            sc = (b_45 & 0xFu) | (((b_lo >> 6u) & 3u) << 4u);
            mn = ((b_45 >> 4u) & 0xFu) | (((b_self >> 6u) & 3u) << 4u);
        }
        let scale: f32 = f32(sc);
        let min_v: f32 = f32(mn);

        let q = read_byte(block_off + qs_local_off);
        let nibble: f32 = select(f32(q & 0xFu), f32(q >> 4u), nibble_hi);

        let w_ij: f32 = d * scale * nibble - dmin * min_v;
        acc = acc + w_ij * dy[j];
    }

    dx[i] = acc;
}
