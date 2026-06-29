// f16-weight variant of conv_transpose1d.wgsl. Identical math; weight tensor is
// stored f16 (2 halves per u32, write_storage_f16 layout) and unpacked via
// unpack2x16float. Activations / bias / output stay f32. Covers the ISTFTNet
// `ups` and the StyleTTS2 depthwise pool upsample for the f16 clone variant.
// Pairs with conv_transpose1d.wgsl (f16-rounded) as its parity oracle.

struct Params {
    cin:      u32,
    tin:      u32,
    cout:     u32,
    tout:     u32,
    k:        u32,
    stride:   u32,
    pad:      u32,
    groups:   u32,
    has_bias: u32,
    _p0:      u32,
    _p1:      u32,
    _p2:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       w:      array<u32>;  // 2×f16 per u32
@group(0) @binding(3) var<storage, read>       bias:   array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;

fn wf16(e: u32) -> f32 {
    let pair = unpack2x16float(w[e >> 1u]);
    return select(pair.x, pair.y, (e & 1u) == 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    if (idx >= params.cout * params.tout) { return; }
    let co = idx / params.tout;
    let to = idx % params.tout;

    let cout_g = params.cout / params.groups;
    let cin_g = params.cin / params.groups;
    let ci_base = (co / cout_g) * cin_g;
    let co_local = co % cout_g;

    var acc: f32 = 0.0;
    if (params.has_bias != 0u) { acc = bias[co]; }

    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        let num = i32(to + params.pad) - i32(kk);
        if (num >= 0 && (u32(num) % params.stride) == 0u) {
            let i = u32(num) / params.stride;
            if (i < params.tin) {
                for (var icg: u32 = 0u; icg < cin_g; icg = icg + 1u) {
                    let ci = ci_base + icg;
                    let widx = (ci * cout_g + co_local) * params.k + kk;
                    acc = acc + x[ci * params.tin + i] * wf16(widx);
                }
            }
        }
    }
    y[idx] = acc;
}
