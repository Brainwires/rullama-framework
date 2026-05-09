// Tiled Q6_K dequant + matmul. See q4_k_dequant_matmul_tiled.wgsl for the tiling
// rationale. Per-thread weight access is unchanged from q6_k_dequant_matmul.wgsl;
// the only difference is that 256-element x blocks are loaded into shared memory
// once per block and reused across all 64 outputs in the workgroup.

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
const BLOCK_BYTES: u32 = 210u;

var<workgroup> x_tile: array<f32, 256>;

fn read_byte(byte_off: u32) -> u32 {
    let u32_idx = byte_off >> 2u;
    let shift   = (byte_off & 3u) << 3u;
    return (weight[u32_idx] >> shift) & 0xFFu;
}

fn read_i8_as_f32(byte_off: u32) -> f32 {
    let b = read_byte(byte_off);
    if (b >= 128u) { return f32(i32(b) - 256); }
    return f32(i32(b));
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
        x_tile[tid           ] = x[x_base + tid           ];
        x_tile[tid + WG      ] = x[x_base + tid + WG      ];
        x_tile[tid + WG * 2u ] = x[x_base + tid + WG * 2u ];
        x_tile[tid + WG * 3u ] = x[x_base + tid + WG * 3u ];
        workgroupBarrier();

        if (is_active) {
            let block_off: u32 = row_byte_off + b * BLOCK_BYTES;
            let d: f32 = read_f16_as_f32(block_off + 208u);

            var sc: array<f32, 16>;
            for (var s: u32 = 0u; s < 16u; s = s + 1u) {
                sc[s] = read_i8_as_f32(block_off + 192u + s);
            }

            for (var p: u32 = 0u; p < 2u; p = p + 1u) {
                let ql_off: u32 = block_off + p * 64u;
                let qh_off: u32 = block_off + 128u + p * 32u;
                let sc_off: u32 = p * 8u;
                let elem_base: u32 = p * 128u;

                for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                    let is: u32 = l >> 4u;

                    let ql_l    = read_byte(ql_off + l);
                    let ql_l_32 = read_byte(ql_off + l + 32u);
                    let qh_l    = read_byte(qh_off + l);

                    let q1: f32 = f32(i32((ql_l    & 0xFu) | ((qh_l & 3u) << 4u)) - 32);
                    let q2: f32 = f32(i32((ql_l_32 & 0xFu) | (((qh_l >> 2u) & 3u) << 4u)) - 32);
                    let q3: f32 = f32(i32((ql_l    >>  4u) | (((qh_l >> 4u) & 3u) << 4u)) - 32);
                    let q4: f32 = f32(i32((ql_l_32 >>  4u) | (((qh_l >> 6u) & 3u) << 4u)) - 32);

                    let s0: f32 = sc[sc_off + is + 0u];
                    let s2: f32 = sc[sc_off + is + 2u];
                    let s4: f32 = sc[sc_off + is + 4u];
                    let s6: f32 = sc[sc_off + is + 6u];

                    acc = acc + x_tile[elem_base + l       ] * d * s0 * q1;
                    acc = acc + x_tile[elem_base + l + 32u ] * d * s2 * q2;
                    acc = acc + x_tile[elem_base + l + 64u ] * d * s4 * q3;
                    acc = acc + x_tile[elem_base + l + 96u ] * d * s6 * q4;
                }
            }
        }

        workgroupBarrier();
    }

    if (is_active) {
        y[j] = acc;
    }
}
