// Fused Q5_0 dequant + matmul: y[j] = Σ_i x[i] * dequant(W[j, i])
//
// W is laid out [k, n] in GGUF (row-major). Each row contains k/32 Q5_0 blocks;
// we require k % 32 == 0. Q5_0 shows up in DiffusionGemma's Q4_K_M build
// (ffn_down / ffn_down_exps / self_cond_down on some layers).
//
// Q5_0 block layout (22 bytes / 32 elems):
//   bytes [ 0.. 2): d  — f16 block scale
//   bytes [ 2.. 6): qh — LE u32, the 5th bit of each quant
//   bytes [ 6..22): qs[16] — 32 × 4-bit low nibbles, two per byte
//
// Dequant (mirrors dequantize_row_q5_0 in ggml-quants.c):
//   xh_0 = ((qh >> l) << 4) & 0x10        (bit l → element l)
//   xh_1 = (qh >> (l + 12)) & 0x10        (bit l+16 → element l+16)
//   value[l]      = f32(((qs[l] & 0xF) | xh_0) - 16) * d
//   value[l + 16] = f32(((qs[l] >> 4)  | xh_1) - 16) * d
//
// Same bind layout + byte-addressing helpers as q4_0_dequant_matmul.wgsl, so the
// matmul_chained_inner dispatcher binds it identically. 22-byte blocks aren't
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
const BLOCK_BYTES: u32 = 22u;

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

fn read_u32_le(byte_off: u32) -> u32 {
    return read_byte(byte_off)
        | (read_byte(byte_off + 1u) << 8u)
        | (read_byte(byte_off + 2u) << 16u)
        | (read_byte(byte_off + 3u) << 24u);
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
        let qh: u32 = read_u32_le(block_off + 2u);
        let qs_off: u32 = block_off + 6u;
        let elem_base: u32 = b * BLOCK_ELEMS;

        for (var l: u32 = 0u; l < 16u; l = l + 1u) {
            let q = read_byte(qs_off + l);
            let xh_0: u32 = ((qh >> l) << 4u) & 0x10u;
            let xh_1: u32 = (qh >> (l + 12u)) & 0x10u;
            let v_lo: f32 = (f32((q & 0xFu) | xh_0) - 16.0) * d;
            let v_hi: f32 = (f32((q >> 4u) | xh_1) - 16.0) * d;

            acc = acc + x[elem_base + l]       * v_lo;
            acc = acc + x[elem_base + l + 16u] * v_hi;
        }
    }

    y[j] = acc;
}
