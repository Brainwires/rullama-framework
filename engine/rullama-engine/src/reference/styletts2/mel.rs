//! StyleTTS2 `compute_style` mel frontend — a 1:1 match of torchaudio's
//! `MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)` followed by
//! `log(1e-5 + mel)` and `(x - (-4)) / 4`.
//!
//! Parity-critical details (all verified against the reference dump):
//!   * torchaudio's **default `sample_rate=16000`** is used even though audio is 24 kHz,
//!     so the Slaney mel filterbank (loaded here as `fb[1025,80]`) is a 16 kHz bank.
//!   * `center=True`, `pad_mode='reflect'` → reflect-pad the signal by `n_fft/2`.
//!   * the 1200-sample Hann window is zero-padded, centered, into the 2048 FFT frame.
//!   * power spectrogram (`power=2`, `normalized=False`), onesided 1025 bins.

const N_FFT: usize = 2048;
const HOP: usize = 300;
const WIN: usize = 1200;
const N_FREQ: usize = N_FFT / 2 + 1; // 1025
pub const N_MELS: usize = 80;

pub struct MelFrontend {
    window: Vec<f32>, // Hann(1200) centered in a 2048 frame (zeros elsewhere)
    fb: Vec<f32>,     // Slaney filterbank [N_FREQ, N_MELS] row-major: fb[k*N_MELS + m]
}

impl MelFrontend {
    /// `hann1200` = torchaudio's `spectrogram.window` (Hann, win_length=1200, periodic);
    /// `fb` = `mel_scale.fb` of shape `[1025, 80]`. Both come from the fixture dump (or,
    /// in production, are baked into the GGUF).
    pub fn new(hann1200: &[f32], fb: &[f32]) -> Self {
        assert_eq!(hann1200.len(), WIN);
        assert_eq!(fb.len(), N_FREQ * N_MELS);
        let pad = (N_FFT - WIN) / 2; // 424
        let mut window = vec![0f32; N_FFT];
        window[pad..pad + WIN].copy_from_slice(hann1200);
        Self {
            window,
            fb: fb.to_vec(),
        }
    }

    /// 24 kHz mono audio → normalized log-mel `[N_MELS, n_frames]` row-major (`m*T + t`),
    /// i.e. the encoder input mel (before the `unsqueeze(1)` channel dim).
    pub fn compute(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let p = N_FFT / 2;
        let padded = reflect_pad(audio, p);
        let n_frames = 1 + audio.len() / HOP; // torch center=True framing
        let mut mel = vec![0f32; N_MELS * n_frames];
        let mut re = vec![0f32; N_FFT];
        let mut im = vec![0f32; N_FFT];
        for t in 0..n_frames {
            let start = t * HOP;
            for i in 0..N_FFT {
                re[i] = padded[start + i] * self.window[i];
                im[i] = 0.0;
            }
            fft_in_place(&mut re, &mut im);
            for m in 0..N_MELS {
                let mut acc = 0f32;
                for k in 0..N_FREQ {
                    let power = re[k] * re[k] + im[k] * im[k];
                    acc += power * self.fb[k * N_MELS + m];
                }
                mel[m * n_frames + t] = ((1e-5 + acc).ln() + 4.0) / 4.0;
            }
        }
        (mel, n_frames)
    }
}

/// Reflect pad by `p` on both sides (torch `F.pad(mode='reflect')`: edge sample not
/// repeated). Requires `audio.len() > p`.
fn reflect_pad(audio: &[f32], p: usize) -> Vec<f32> {
    let n = audio.len();
    let mut out = vec![0f32; n + 2 * p];
    for i in 0..p {
        out[i] = audio[p - i]; // out[0]=audio[p] … out[p-1]=audio[1]
    }
    out[p..p + n].copy_from_slice(audio);
    for j in 0..p {
        out[p + n + j] = audio[n - 2 - j]; // mirror without repeating last sample
    }
    out
}

/// In-place iterative radix-2 Cooley–Tukey FFT (len must be a power of two).
fn fft_in_place(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1f32, 0f32);
            for k in 0..len / 2 {
                let a = i + k;
                let b = i + k + len / 2;
                let tr = cr * re[b] - ci * im[b];
                let ti = cr * im[b] + ci * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}
