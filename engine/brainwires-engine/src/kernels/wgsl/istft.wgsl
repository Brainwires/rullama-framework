// Exact iSTFT (onesided, center, COLA-normalized) — matches torch.istft and the
// kokoro oracle. Deterministic GATHER form: one thread per OUTPUT sample, summing the
// windowed IFFT contributions of every overlapping frame (no scatter / atomics, so
// it's bit-stable for parity). spec/phase are [nbins, frames] channel-major (mag, angle).
// Output length = (frames-1)*hop + n_fft - 2*pad, pad = n_fft/2.

struct Params {
    nbins:   u32,
    frames:  u32,
    nfft:    u32,
    hop:     u32,
    pad:     u32,
    out_len: u32,
    _p0:     u32,
    _p1:     u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       spec:   array<f32>;
@group(0) @binding(2) var<storage, read>       phase:  array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

const PI: f32 = 3.14159265358979;

// IFFT sample n of frame f, reconstructing the full spectrum from the onesided
// (nbins) bins via conjugate symmetry. Returns the real time-domain value.
fn ifft_sample(f: u32, n: u32) -> f32 {
    let nfft = params.nfft;
    let nbins = params.nbins;
    var s: f32 = 0.0;
    for (var k: u32 = 0u; k < nfft; k = k + 1u) {
        var kk = k;
        var conj = false;
        if (k >= nbins) { kk = nfft - k; conj = true; }
        let m = spec[kk * params.frames + f];
        let ph = phase[kk * params.frames + f];
        var re = m * cos(ph);
        var im = m * sin(ph);
        if (conj) { im = -im; }
        let ang = 2.0 * PI * f32(k * n) / f32(nfft);
        s = s + re * cos(ang) - im * sin(ang);
    }
    return s / f32(nfft);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.out_len) { return; }
    let o = s + params.pad; // position in the OLA (pre-trim) buffer

    // frames f with 0 <= o - f*hop < nfft contribute
    var f_lo: u32 = 0u;
    if (o + 1u > params.nfft) { f_lo = (o + 1u - params.nfft + params.hop - 1u) / params.hop; }
    let f_hi = o / params.hop;

    var acc: f32 = 0.0;
    var env: f32 = 0.0;
    var f: u32 = f_lo;
    loop {
        if (f > f_hi || f >= params.frames) { break; }
        let fh = f * params.hop;
        if (o >= fh) {
            let n = o - fh;
            if (n < params.nfft) {
                let w = 0.5 - 0.5 * cos(2.0 * PI * f32(n) / f32(params.nfft));
                acc = acc + ifft_sample(f, n) * w;
                env = env + w * w;
            }
        }
        f = f + 1u;
    }
    if (env > 1e-11) { acc = acc / env; }
    y[s] = acc;
}
