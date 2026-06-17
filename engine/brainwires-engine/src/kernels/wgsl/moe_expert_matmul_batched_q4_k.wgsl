// Batched MulmatID Q4_K expert matmul for the DiffusionGemma canvas: every
// (position, slot) applies its OWN selected expert, in one dispatch.
//
//   y[ps, j] = Σ_i x[pos, i] · dequant(W[ids[ps]][j, i])
//
// where ps = pos*top_k + slot indexes the flattened (position, slot) grid,
// pos = ps / top_k. W is the stacked 3-D `ffn_*_exps.weight` resident as one
// buffer; the expert id is read from the GPU-resident `ids` (the batched
// router's output). Dequant math == moe_expert_matmul_q4_k.wgsl; this adds the
// position/slot indexing.
//
// Dispatch: workgroups (ceil(n/64), n_pos*top_k, 1); thread global x = output
// row j, workgroup y = ps.

struct Params {
    k:            u32,
    n:            u32,
    top_k:        u32,
    slice_blocks: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;
@group(0) @binding(2) var<storage, read>       ids:    array<u32>; // [n_pos*top_k]
@group(0) @binding(3) var<storage, read>       x:      array<f32>; // [n_pos, k]
@group(0) @binding(4) var<storage, read_write> y:      array<f32>; // [n_pos*top_k, n]

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
    return unpack2x16float(lo | (hi << 8u)).x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let j: u32 = gid.x;
    if (j >= params.n) { return; }
    let ps = wid.y;                       // pos*top_k + slot
    let pos = ps / params.top_k;
    let expert = ids[ps];
    let x_off = pos * params.k;

    let n_blocks: u32 = params.k / BLOCK_ELEMS;
    let row_byte_off: u32 = (expert * params.slice_blocks + j * n_blocks) * BLOCK_BYTES;

    var acc: f32 = 0.0;
    for (var b: u32 = 0u; b < n_blocks; b = b + 1u) {
        let block_off: u32 = row_byte_off + b * BLOCK_BYTES;
        let d:    f32 = read_f16_as_f32(block_off + 0u);
        let dmin: f32 = read_f16_as_f32(block_off + 2u);
        var sb: array<u32, 12>;
        for (var s: u32 = 0u; s < 12u; s = s + 1u) { sb[s] = read_byte(block_off + 4u + s); }
        var scales: array<f32, 8>;
        var mins:   array<f32, 8>;
        for (var jj: u32 = 0u; jj < 8u; jj = jj + 1u) {
            var sc: u32; var mn: u32;
            if (jj < 4u) {
                sc = sb[jj] & 63u;
                mn = sb[jj + 4u] & 63u;
            } else {
                sc = (sb[jj + 4u] & 0xFu) | (((sb[jj - 4u] >> 6u) & 3u) << 4u);
                mn = ((sb[jj + 4u] >> 4u) & 0xFu) | (((sb[jj] >> 6u) & 3u) << 4u);
            }
            scales[jj] = f32(sc); mins[jj] = f32(mn);
        }
        let qs_off: u32 = block_off + 16u;
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            let is_lo = 2u * c; let is_hi = is_lo + 1u;
            let chunk = qs_off + c * 32u;
            let eb = b * BLOCK_ELEMS + c * 64u;
            let s_lo = scales[is_lo]; let m_lo = mins[is_lo];
            let s_hi = scales[is_hi]; let m_hi = mins[is_hi];
            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                let q = read_byte(chunk + l);
                let v_lo = d * s_lo * f32(q & 0xFu) - dmin * m_lo;
                let v_hi = d * s_hi * f32(q >> 4u) - dmin * m_hi;
                acc = acc + x[x_off + eb + l]       * v_lo;
                acc = acc + x[x_off + eb + l + 32u] * v_hi;
            }
        }
    }
    y[ps * params.n + j] = acc;
}
