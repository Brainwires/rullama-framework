//! Decoder front (istftnet.Decoder.forward up to the generator): F0/N downsample
//! convs, AdaIN encode, and the 4-block AdainResBlk1d decode stack (last upsamples).
//! `s` here is the TIMBRE half of the voice (ref_s[:128]).
#![allow(dead_code)]

use super::convblocks::conv1d;
use super::KokoroModel;

impl KokoroModel {
    /// Returns (`dec_encode [1024, F]`, `x_after_decode [512, 2F]`, `F0_down [F]`, `N_down [F]`).
    /// `t_en [512, T]` channel-major; `f0_curve`/`n_curve` are `[2F]` (= 156).
    pub fn decoder_features(
        &self, t_en: &[f32], f0_curve: &[f32], n_curve: &[f32], dur: &[usize], style: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = self.cfg.hidden_dim; // 512
        let t = dur.len();
        // asr = expand t_en by durations → [512, F]. (transpose to row-major then expand)
        let mut t_en_rm = vec![0.0f32; t * c];
        for ch in 0..c {
            for ti in 0..t {
                t_en_rm[ti * c + ch] = t_en[ch * t + ti];
            }
        }
        let (asr, f) = self.expand_by_dur_cm(&t_en_rm, t, c, dur);

        // F0/N downsample convs: Conv1d(1,1,k3,stride2,pad1): [2F] → [F]
        let f0w = self.t("k.decoder.F0_conv.weight");
        let f0b = self.t("k.decoder.F0_conv.bias");
        let (f0d, _) = conv1d(f0_curve, 1, f0_curve.len(), &f0w, Some(&f0b), 1, 3, 2, 1, 1, 1);
        let nw = self.t("k.decoder.N_conv.weight");
        let nb = self.t("k.decoder.N_conv.bias");
        let (nd, _) = conv1d(n_curve, 1, n_curve.len(), &nw, Some(&nb), 1, 3, 2, 1, 1, 1);

        // x = cat([asr(512), F0(1), N(1)]) → [514, F]; encode → [1024, F]
        let cat0 = self.cat_channels(&[(&asr, c), (&f0d, 1), (&nd, 1)], f);
        let (dec_encode, _) = self.adain_resblk1d("k.decoder.encode", &cat0, c + 2, f, 1024, false, style);

        // asr_res = Conv1d(512,64,k1) → [64, F]
        let arw = self.t("k.decoder.asr_res.0.weight");
        let arb = self.t("k.decoder.asr_res.0.bias");
        let (asr_res, _) = conv1d(&asr, c, f, &arw, Some(&arb), 64, 1, 1, 0, 1, 1);

        // decode stack: 4× AdainResBlk1d, cat([x, asr_res, F0, N]) before each, last upsamples ×2
        let mut x = dec_encode.clone();
        let mut tcur = f;
        for i in 0..4 {
            let xin = self.cat_channels(&[(&x, x.len() / tcur), (&asr_res, 64), (&f0d, 1), (&nd, 1)], tcur);
            let dim_in = x.len() / tcur + 64 + 2; // 1090
            let upsample = i == 3;
            let dim_out = if i < 3 { 1024 } else { 512 };
            let (nx, nt) = self.adain_resblk1d(&format!("k.decoder.decode.{i}"), &xin, dim_in, tcur, dim_out, upsample, style);
            x = nx;
            tcur = nt;
        }
        (dec_encode, x, f0d, nd)
    }

    /// Concatenate channel-major `[C_i, T]` tensors along the channel axis → `[sum C_i, T]`.
    fn cat_channels(&self, parts: &[(&[f32], usize)], t: usize) -> Vec<f32> {
        let ctot: usize = parts.iter().map(|(_, c)| *c).sum();
        let mut out = vec![0.0f32; ctot * t];
        let mut base = 0;
        for (data, c) in parts {
            out[base * t..(base + c) * t].copy_from_slice(&data[..c * t]);
            base += c;
        }
        out
    }
}
