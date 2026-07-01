// Fused Q4_K dequant + matmul: y[j] = Σ_i x[i] * dequant(W[j, i])
//
// W is laid out [k, n] in GGUF (row-major). Each row contains k/256 Q4_K super-blocks;
// we require k % 256 == 0.
//
// Q4_K block layout (144 bytes / 256 elems):
//   bytes [  0..  2): d     — f16 super-block scale (for quantized scales)
//   bytes [  2..  4): dmin  — f16 super-block scale (for quantized mins)
//   bytes [  4.. 16): scales[12] — packed 6-bit (scale, min) per sub-block ×8
//   bytes [ 16..144): qs[128]    — 4-bit weights (256 nibbles, two per byte)
//
// Dequant per 64-element chunk (4 chunks per block, sub-blocks indexed 0..8):
//   chunk c uses sub-block indices is=2c (low nibbles) and is+1 (high nibbles).
//   value[j+l]      = d * scale[is]    * (qs[c*32 + l] & 0xF) - dmin * min[is]
//   value[j+l+32]   = d * scale[is+1]  * (qs[c*32 + l] >> 4)  - dmin * min[is+1]
//
// `get_scale_min_k4(j, scales)` decodes the packed scales into 6-bit (scale, min) pairs.

struct Params {
    k: u32,
    n: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

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
        let d:    f32 = read_f16_as_f32(block_off + 0u);
        let dmin: f32 = read_f16_as_f32(block_off + 2u);

        // Pre-load 12 packed scale bytes once.
        var sb: array<u32, 12>;
        for (var s: u32 = 0u; s < 12u; s = s + 1u) {
            sb[s] = read_byte(block_off + 4u + s);
        }

        // Decode 8 (scale, min) pairs from the 12 packed bytes.
        // get_scale_min_k4(j, q):
        //   if j < 4: scale = q[j] & 63;            min = q[j+4] & 63;
        //   else:     scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        //             min   = (q[j+4] >> 4)  | ((q[j-0] >> 6) << 4);
        var scales: array<f32, 8>;
        var mins:   array<f32, 8>;
        for (var jj: u32 = 0u; jj < 8u; jj = jj + 1u) {
            var sc: u32;
            var mn: u32;
            if (jj < 4u) {
                sc = sb[jj] & 63u;
                mn = sb[jj + 4u] & 63u;
            } else {
                sc = (sb[jj + 4u] & 0xFu) | (((sb[jj - 4u] >> 6u) & 3u) << 4u);
                mn = ((sb[jj + 4u] >> 4u) & 0xFu) | (((sb[jj] >> 6u) & 3u) << 4u);
            }
            scales[jj] = f32(sc);
            mins[jj]   = f32(mn);
        }

        // Process 4 chunks of 64 elements each.
        let qs_off: u32 = block_off + 16u;
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            let is_lo: u32 = 2u * c;
            let is_hi: u32 = is_lo + 1u;
            let chunk_qs_off: u32 = qs_off + c * 32u;
            let elem_base: u32 = b * BLOCK_ELEMS + c * 64u;

            let s_lo = scales[is_lo];
            let m_lo = mins[is_lo];
            let s_hi = scales[is_hi];
            let m_hi = mins[is_hi];

            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                let q = read_byte(chunk_qs_off + l);
                let q_lo: f32 = f32(q & 0xFu);
                let q_hi: f32 = f32(q >> 4u);

                let v_lo: f32 = d * s_lo * q_lo - dmin * m_lo;
                let v_hi: f32 = d * s_hi * q_hi - dmin * m_hi;

                acc = acc + x[elem_base + l]       * v_lo;
                acc = acc + x[elem_base + l + 32u] * v_hi;
            }
        }
    }

    y[j] = acc;
}
