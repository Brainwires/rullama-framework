// Backward of `y = matmul_q6_k(W, x)` with respect to x.
//
// Forward (q6_k_dequant_matmul.wgsl): y[j] = Σ_i x[i] * dequant(W)[j, i]
// Backward:                            dx[i] = Σ_j dy[j] * dequant(W)[j, i]
//
// W is stored row-major [n, k] with each row packed into k/256 Q6_K
// super-blocks of 210 bytes (256 elements each). Used by the training
// backward pass for the tied embedding (Gemma 4 ships token_embd as
// Q6_K), so this kernel is sized for vocab × d_model dispatches.
//
// One workgroup per block-row of k (256 contiguous output elements).
// Each thread within the workgroup handles one i value within that
// block-row, loops over j ∈ [0, n), and accumulates dy[j] · W[j, i]
// dequantized on the fly. W is frozen — no weight gradient.

// **Vocab-axis tiling (Patch 6).** `j_start..j_end` bounds the sum-axis
// loop so the kernel can be dispatched as N tiles, each consuming only
// `(j_end-j_start)/n` of the dequantized weight working set per
// command buffer. Non-tiled callers pass `j_start=0, j_end=n,
// accumulate=0` and behave exactly as before. Tiled callers set
// `accumulate=1` for tiles 1..N (tile 0 still writes) to add into the
// running `dx` instead of overwriting. This breaks up iOS Safari's
// per-dispatch Metal heap working set for the big head outproj matmul
// (vocab=262144) — a single dispatch was bringing ~400 MB of f32
// dequant through Metal's execution path; 8 tiles bring ~50 MB each.
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

const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 210u;

fn read_byte(byte_off: u32) -> u32 {
    let u32_idx = byte_off >> 2u;
    let shift   = (byte_off & 3u) << 3u;
    return (weight[u32_idx] >> shift) & 0xFFu;
}

fn read_i8_as_f32(byte_off: u32) -> f32 {
    let b = read_byte(byte_off);
    if (b >= 128u) {
        return f32(i32(b) - 256);
    }
    return f32(i32(b));
}

fn read_f16_as_f32(byte_off: u32) -> f32 {
    let lo = read_byte(byte_off);
    let hi = read_byte(byte_off + 1u);
    let packed: u32 = lo | (hi << 8u);
    return unpack2x16float(packed).x;
}

/// Dequantize a single element at index `i_in_block` (0..256) of the
/// Q6_K super-block starting at `block_off`. Mirrors the per-element
/// math of `q6_k_dequant_matmul.wgsl`'s inner loop.
fn dequant_q6_at(block_off: u32, i_in_block: u32) -> f32 {
    let d = read_f16_as_f32(block_off + 208u);
    let half       = i_in_block / 128u;     // 0 or 1
    let e_in_half  = i_in_block % 128u;     // 0..128
    let group      = e_in_half / 32u;       // 0..4 → q1, q2, q3, q4 in the forward
    let l          = e_in_half % 32u;       // 0..32
    let is         = l / 16u;               // 0 or 1
    let sc_off     = half * 8u;
    let scale_idx  = sc_off + is + group * 2u;
    let scale      = read_i8_as_f32(block_off + 192u + scale_idx);

    // q1, q3 use ql at offset `l`; q2, q4 use ql at offset `l + 32`.
    let ql_low_position: bool = (group == 0u) || (group == 2u);
    let ql_l_offset      = select(l + 32u, l, ql_low_position);
    let ql_off           = block_off + half * 64u + ql_l_offset;
    let qh_off           = block_off + 128u + half * 32u + l;

    let ql_b: u32 = read_byte(ql_off);
    let qh_b: u32 = read_byte(qh_off);
    let ql_high_nibble: bool = group >= 2u;
    let ql_nibble: u32 = select(ql_b & 0xFu, ql_b >> 4u, ql_high_nibble);
    let qh_bits:   u32 = (qh_b >> (group * 2u)) & 3u;
    let q: i32 = i32(ql_nibble | (qh_bits << 4u)) - 32;
    return d * scale * f32(q);
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

    var acc: f32 = 0.0;
    for (var j: u32 = params.j_start; j < params.j_end; j = j + 1u) {
        let block_off: u32 = j * row_bytes + block_row * BLOCK_BYTES;
        let w_ji: f32      = dequant_q6_at(block_off, tid);
        acc = acc + w_ji * dy[j];
    }

    if (params.accumulate == 0u) {
        dx[i] = acc;
    } else {
        dx[i] = dx[i] + acc;
    }
}
