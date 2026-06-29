// f16-weight variant of conv1d.wgsl. Identical math + indexing, but the weight
// tensor is stored f16 (2 halves packed per u32, the layout write_storage_f16
// produces) and unpacked on-the-fly with `unpack2x16float`. Activations (x),
// bias, and output stay f32. This halves the resident weight footprint of the
// StyleTTS2 generator / decoder / style-encoder convolutions — the bulk of the
// model — for the memory-tight f16 clone variant.
//
// Pairs with conv1d.wgsl as its CPU/f32 oracle (GPU-vs-GPU parity test).

struct Params {
    cin:      u32,
    tin:      u32,
    cout:     u32,
    tout:     u32,
    k:        u32,
    stride:   u32,
    pad:      u32,
    dilation: u32,
    groups:   u32,
    has_bias: u32,
    _p0:      u32,
    _p1:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       w:      array<u32>;  // 2×f16 per u32
@group(0) @binding(3) var<storage, read>       bias:   array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;

// f16 weight element `e` — low half of u32 = even index, high half = odd.
fn wf16(e: u32) -> f32 {
    let pair = unpack2x16float(w[e >> 1u]);
    return select(pair.x, pair.y, (e & 1u) == 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * nwg.x * 64u + gid.x;
    let total = params.cout * params.tout;
    if (idx >= total) { return; }

    let co = idx / params.tout;
    let to = idx % params.tout;
    let cout_g = params.cout / params.groups;
    let cin_g = params.cin / params.groups;
    let ci_base = (co / cout_g) * cin_g;
    let wbase = co * cin_g * params.k;

    var acc: f32 = 0.0;
    if (params.has_bias != 0u) { acc = bias[co]; }

    for (var icg: u32 = 0u; icg < cin_g; icg = icg + 1u) {
        let ci = ci_base + icg;
        let wrow = wbase + icg * params.k;
        let xrow = ci * params.tin;
        for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
            let ipos = i32(to * params.stride + kk * params.dilation) - i32(params.pad);
            if (ipos >= 0 && ipos < i32(params.tin)) {
                acc = acc + wf16(wrow + kk) * x[xrow + u32(ipos)];
            }
        }
    }
    y[idx] = acc;
}
