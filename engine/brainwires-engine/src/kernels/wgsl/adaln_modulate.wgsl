// adaLN modulation for diffusion transformers:
//
//   y[t, c] = x[t, c] * (1 + scale[c]) + shift[c]
//
// x is [seq, hidden] flat; `scale` and `shift` are length-`hidden` vectors
// (produced per-image from the timestep embedding) broadcast across all tokens.
// This is the modulation applied to a (no-affine) LayerNorm/RMSNorm output
// inside each DiT block — the "+1" makes scale==0 the identity. The companion
// gate is applied separately on the residual branch.
//
// Reference oracle: reference::imagegen::adaln_modulate.

struct Params {
    hidden: u32,
    total:  u32,   // seq * hidden
    _p0:    u32,
    _p1:    u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       scale:  array<f32>;
@group(0) @binding(3) var<storage, read>       shift:  array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.total) { return; }
    let c = i % params.hidden;
    y[i] = x[i] * (1.0 + scale[c]) + shift[c];
}
