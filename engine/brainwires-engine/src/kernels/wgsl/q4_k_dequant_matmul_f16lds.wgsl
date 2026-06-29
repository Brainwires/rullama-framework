// Q4_K dequant + matmul with f16 LDS x and f16 inner-loop arithmetic.
//
// Same dequant arithmetic as q4_k_dequant_matmul_tiled.wgsl, but:
//   * `x_tile` stored as f16 (halves LDS bytes for the per-WG x cache).
//   * Inner mul `x_tile[k] * v` runs in f16 with the f32 accumulator
//     untouched. On GCN 1.2+ / Apple Silicon naga emits packed FP16
//     fused MADs (v_pk_fma_f16), matching the win we got for the batched
//     matmuls (128 → 168 GFLOPS, ~30%).
//
// Numerical envelope: Q4_K dequantised values are at most d * 63 * 15 ≈
// d × 945 with d typically O(0.1)-O(10) — comfortably under f16's 65504
// ceiling. x is post-norm so also bounded. Accumulator stays f32 for the
// 2560-term sum.
//
// Requires `Features::SHADER_F16`.

enable f16;

struct Params {
    k: u32,
    n: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

const WG: u32 = 64u;
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 144u;

var<workgroup> x_tile: array<f16, 256>;

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
fn main(
    @builtin(workgroup_id)         wg_id: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let j: u32 = wg_id.x * WG + tid;
    let is_active: bool = j < params.n;

    let n_blocks: u32 = params.k / BLOCK_ELEMS;
    let row_bytes: u32 = n_blocks * BLOCK_BYTES;
    let row_byte_off: u32 = j * row_bytes;

    var acc: f32 = 0.0;

    for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
        let x_base: u32 = b * BLOCK_ELEMS;
        // Cooperative x load: 4 elements per thread, f32 → f16 in LDS.
        x_tile[tid           ] = f16(x[x_base + tid           ]);
        x_tile[tid + WG      ] = f16(x[x_base + tid + WG      ]);
        x_tile[tid + WG * 2u ] = f16(x[x_base + tid + WG * 2u ]);
        x_tile[tid + WG * 3u ] = f16(x[x_base + tid + WG * 3u ]);
        workgroupBarrier();

        if (is_active) {
            let block_off: u32 = row_byte_off + b * BLOCK_BYTES;
            let d_f32:    f32 = read_f16_as_f32(block_off + 0u);
            let dmin_f32: f32 = read_f16_as_f32(block_off + 2u);
            let d:    f16 = f16(d_f32);
            let dmin: f16 = f16(dmin_f32);

            var sb: array<u32, 12>;
            for (var s: u32 = 0u; s < 12u; s = s + 1u) {
                sb[s] = read_byte(block_off + 4u + s);
            }
            var scales: array<f16, 8>;
            var mins:   array<f16, 8>;
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
                scales[jj] = f16(f32(sc));
                mins[jj]   = f16(f32(mn));
            }

            let qs_off: u32 = block_off + 16u;
            for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                let is_lo: u32 = 2u * c;
                let is_hi: u32 = is_lo + 1u;
                let chunk_qs_off: u32 = qs_off + c * 32u;
                let s_lo = scales[is_lo];
                let m_lo = mins[is_lo];
                let s_hi = scales[is_hi];
                let m_hi = mins[is_hi];
                let elem_base: u32 = c * 64u;

                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let q = read_byte(chunk_qs_off + l);
                    let q_lo: f16 = f16(f32(q & 0xFu));
                    let q_hi: f16 = f16(f32(q >> 4u));

                    let v_lo: f16 = d * s_lo * q_lo - dmin * m_lo;
                    let v_hi: f16 = d * s_hi * q_hi - dmin * m_hi;

                    acc = acc + f32(x_tile[elem_base + l       ] * v_lo);
                    acc = acc + f32(x_tile[elem_base + l + 32u ] * v_hi);
                }
            }
        }

        workgroupBarrier();
    }

    if (is_active) {
        y[j] = acc;
    }
}
