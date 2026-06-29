// General ConvTranspose1d, channel-major [C, T]. Gather form (one thread per output
// element, no atomics). Covers the ISTFTNet `ups` (groups=1) and the StyleTTS2 pool
// upsample (depthwise, groups=channels). weight w[Cin, Cout/groups, K] row-major.
// tout = (tin-1)*stride - 2*pad + (K-1) + output_padding + 1.

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
@group(0) @binding(2) var<storage, read>       w:      array<f32>;
@group(0) @binding(3) var<storage, read>       bias:   array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;

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
                    acc = acc + x[ci * params.tin + i] * w[widx];
                }
            }
        }
    }
    y[idx] = acc;
}
