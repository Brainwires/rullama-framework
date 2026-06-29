// Batched MulmatID Q8_0 expert matmul (DiffusionGemma canvas). Same
// position/slot indexing as moe_expert_matmul_batched_q4_k.wgsl; dequant ==
// moe_expert_matmul_q8_0.wgsl. Used for ffn_down_exps (Q8_0 in 26b-a4b /
// diffusion-gemma). slice_blocks = (k/32)*n.

struct Params {
    k:            u32,
    n:            u32,
    top_k:        u32,
    slice_blocks: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;
@group(0) @binding(2) var<storage, read>       ids:    array<u32>;
@group(0) @binding(3) var<storage, read>       x:      array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;

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
    return unpack2x16float(lo | (hi << 8u)).x;
}
fn i8_to_f32(q: u32) -> f32 { return f32(i32(q << 24u) >> 24u); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let j: u32 = gid.x;
    if (j >= params.n) { return; }
    let ps = wid.y;
    let pos = ps / params.top_k;
    let expert = ids[ps];
    let x_off = pos * params.k;

    let n_blocks: u32 = params.k / BLOCK_ELEMS;
    let row_byte_off: u32 = (expert * params.slice_blocks + j * n_blocks) * BLOCK_BYTES;

    var acc: f32 = 0.0;
    for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
        let block_off: u32 = row_byte_off + b * BLOCK_BYTES;
        let d: f32 = read_f16_as_f32(block_off + 0u);
        let qs_off: u32 = block_off + 2u;
        let eb: u32 = b * BLOCK_ELEMS;
        for (var l: u32 = 0u; l < 32u; l = l + 1u) {
            acc = acc + x[x_off + eb + l] * i8_to_f32(read_byte(qs_off + l)) * d;
        }
    }
    y[ps * params.n + j] = acc;
}
