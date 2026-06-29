// MulmatID-style Q4_K expert matmul: y[j] = Σ_i x[i] * dequant(W[ids[slot]][j, i])
//
// W is a stacked 3-D expert tensor (`ffn_*_exps.weight`, GGUF dims
// [k, n, n_experts]) resident as ONE buffer; the expert to use is read from
// the GPU-resident `ids` buffer written by moe_router.wgsl — the CPU never
// learns the selection (no readback per token). `slice_blocks` is the number
// of Q4_K blocks in one expert's [k, n] slice = (k/256)*n.
//
// Dequant math is identical to q4_k_dequant_matmul.wgsl (see there for the
// block layout); only the row addressing adds the expert base offset.

struct Params {
    k:            u32,
    n:            u32,
    slot:         u32,
    slice_blocks: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;
@group(0) @binding(2) var<storage, read>       ids:    array<u32>;
@group(0) @binding(3) var<storage, read>       x:      array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;

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
    let expert: u32 = ids[params.slot];
    let row_byte_off: u32 = (expert * params.slice_blocks + j * n_blocks) * BLOCK_BYTES;

    var acc: f32 = 0.0;

    for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
        let block_off: u32 = row_byte_off + b * BLOCK_BYTES;
        let d:    f32 = read_f16_as_f32(block_off + 0u);
        let dmin: f32 = read_f16_as_f32(block_off + 2u);

        var sb: array<u32, 12>;
        for (var s: u32 = 0u; s < 12u; s = s + 1u) {
            sb[s] = read_byte(block_off + 4u + s);
        }

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
                let v_lo: f32 = d * s_lo * f32(q & 0xFu) - dmin * m_lo;
                let v_hi: f32 = d * s_hi * f32(q >> 4u) - dmin * m_hi;

                acc = acc + x[elem_base + l]       * v_lo;
                acc = acc + x[elem_base + l + 32u] * v_hi;
            }
        }
    }

    y[j] = acc;
}
