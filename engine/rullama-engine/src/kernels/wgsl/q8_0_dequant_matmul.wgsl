// Fused Q8_0 dequant + matmul: y[j] = Σ_i x[i] * dequant(W[j, i])
//
// W is laid out [k, n] in GGUF (row-major). Each row contains k/32 Q8_0 blocks;
// we require k % 32 == 0. Q8_0 is the 8-bit legacy ggml quant the `-it-q8_0`
// Ollama tags ship.
//
// Q8_0 block layout (34 bytes / 32 elems):
//   bytes [ 0.. 2): d  — f16 block scale
//   bytes [ 2..34): qs[32] — 32 × signed int8 quants
//
// Dequant (mirrors dequantize_row_q8_0 in ggml-quants.c):
//   value[l] = f32(i8(qs[l])) * d
//
// Same bind layout + byte-addressing helpers as q4_0_dequant_matmul.wgsl, so the
// matmul_chained_inner dispatcher binds it identically. 34-byte blocks aren't
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
const BLOCK_BYTES: u32 = 34u;

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

// Sign-extend an 8-bit value read as u32 → f32.
fn i8_to_f32(q: u32) -> f32 {
    return f32(i32(q << 24u) >> 24u);
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

        for (var l: u32 = 0u; l < 32u; l = l + 1u) {
            let q = read_byte(qs_off + l);
            acc = acc + x[elem_base + l] * i8_to_f32(q) * d;
        }
    }

    y[j] = acc;
}
