// ISTFTNet conv_post split, channel-major. post[2*nbins, T] → spec[nbins,T]=exp(post[:nbins]),
// phase[nbins,T]=sin(post[nbins:]). Keeps the generator's spectral split on-GPU so the
// forward can buffer-chain straight into the iSTFT kernel (no readback).

struct Params { nbins: u32, t: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       post:   array<f32>;
@group(0) @binding(2) var<storage, read_write> spec:   array<f32>;
@group(0) @binding(3) var<storage, read_write> phase:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.nbins * params.t) { return; }
    let b = idx / params.t;
    let t = idx % params.t;
    spec[idx] = exp(post[b * params.t + t]);
    phase[idx] = sin(post[(b + params.nbins) * params.t + t]);
}
