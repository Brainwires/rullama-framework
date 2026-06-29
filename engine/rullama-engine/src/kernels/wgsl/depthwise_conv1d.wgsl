// Depthwise 1D convolution along the time axis (Conformer LightConv).
//
// Input  x:      f32 [T, C]                     (channel-last; same order as glu_split output)
// Weight w:      f32 [C, K]                     (per-channel kernel taps)
// Output y:      f32 [T, C]
//
// y[t, c] = Σ_k x[t - (K-1) + k, c] * w[c, k]   (left-padded with zeros so the
//                                                output time length matches input,
//                                                matching Ollama's manual implementation
//                                                in model_audio.go::forwardLightConv).
//
// One thread per output element. Total threads = T * C.

struct Params {
    seq:      u32,    // T
    channels: u32,    // C
    kernel:   u32,    // K
    _p:       u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       w:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total: u32 = params.seq * params.channels;
    let idx: u32 = gid.x;
    if (idx >= total) { return; }

    let t: u32 = idx / params.channels;
    let c: u32 = idx - t * params.channels;
    let k_max: u32 = params.kernel;

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < k_max; k = k + 1u) {
        let shift: u32 = k_max - 1u - k;
        // Source time index, with left-padding by `shift`.
        if (t < shift) { continue; }
        let src_t: u32 = t - shift;
        let xv: f32 = x[src_t * params.channels + c];
        let wv: f32 = w[c * k_max + k];
        acc = acc + xv * wv;
    }
    y[idx] = acc;
}
