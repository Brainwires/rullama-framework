// Fused Q4_0 dequant + matmul: y[j] = Σ_i x[i] * dequant(W[j, i])
//
// W is laid out [k, n] in GGUF (row-major). Each row contains k/32 Q4_0 blocks;
// we require k % 32 == 0. Q4_0 is the legacy ggml quant Google ships QAT Gemma in.
//
// Q4_0 block layout (18 bytes / 32 elems):
//   bytes [ 0.. 2): d  — f16 block scale
//   bytes [ 2..18): qs[16] — 32 × 4-bit quants, two per byte
//
// Dequant (mirrors dequantize_row_q4_0 in ggml-quants.c):
//   value[l]      = (f32(qs[l] & 0xF) - 8) * d   for l in 0..16
//   value[l + 16] = (f32(qs[l] >> 4)  - 8) * d
//
// Same bind layout + byte-addressing helpers as q4_k_dequant_matmul.wgsl, so the
// matmul_chained_inner dispatcher binds it identically. 18-byte blocks aren't
// u32-aligned, but read_byte/read_f16_as_f32 compute byte offsets manually.

struct Params {
    k: u32,
    n: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j: u32 = gid.x;
    if (j >= params.n) { return; }

    let n_blocks: u32 = params.k / BLOCK_ELEMS;
    let row_bytes: u32 = n_blocks * BLOCK_BYTES;
    let row_byte_off: u32 = j * row_bytes;

    var acc: f32 = 0.0;

    for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
        let block_off: u32 = row_byte_off + b * BLOCK_BYTES;
        let d: f32 = read_f16_as_f32(block_off + 0u);
        let qs_off: u32 = block_off + 2u;
        let elem_base: u32 = b * BLOCK_ELEMS;

        for (var l: u32 = 0u; l < 16u; l = l + 1u) {
            let q = read_byte(qs_off + l);
            let v_lo: f32 = (f32(q & 0xFu) - 8.0) * d;
            let v_hi: f32 = (f32(q >> 4u) - 8.0) * d;

            acc = acc + x[elem_base + l]       * v_lo;
            acc = acc + x[elem_base + l + 16u] * v_hi;
        }
    }

    y[j] = acc;
}
