//! TextEncoder (modules.TextEncoder): phoneme embedding → 3×(Conv1d k5 +
//! channel-axis LayerNorm + LeakyReLU) → BiLSTM. Output `[hidden, T]` channel-major.
#![allow(dead_code)]

use super::KokoroModel;
use super::convblocks::conv1d;
use super::ops::leaky_relu;

impl KokoroModel {
    /// Returns `t_en [hidden, T]` channel-major (matches the PyTorch text_encoder output).
    pub fn text_encoder(&self, input_ids: &[i64]) -> Vec<f32> {
        let t = input_ids.len();
        let c = self.cfg.hidden_dim; // 512
        let k = self.cfg.text_encoder_kernel_size; // 5
        let pad = (k - 1) / 2; // 2

        // embedding → channel-major [C, T]
        let emb = self.t("k.text_encoder.embedding.weight"); // [n_token, C]
        let mut x = vec![0.0f32; c * t];
        for (ti, &id) in input_ids.iter().enumerate() {
            let row = &emb[id as usize * c..(id as usize + 1) * c];
            for ch in 0..c {
                x[ch * t + ti] = row[ch];
            }
        }

        // 3× (Conv1d k5 pad2 → channel-axis LayerNorm(512) → LeakyReLU(0.2))
        for i in 0..self.cfg.n_layer {
            let cw = self.t(&format!("k.text_encoder.cnn.{i}.0.weight"));
            let cb = self.t(&format!("k.text_encoder.cnn.{i}.0.bias"));
            let (conv, _) = conv1d(&x, c, t, &cw, Some(&cb), c, k, 1, pad, 1, 1);
            // channel-axis LayerNorm: per time t, normalize over the C channels (eps 1e-5)
            let gamma = self.t(&format!("k.text_encoder.cnn.{i}.1.gamma"));
            let beta = self.t(&format!("k.text_encoder.cnn.{i}.1.beta"));
            let mut ln = vec![0.0f32; c * t];
            for ti in 0..t {
                let mean = (0..c).map(|ch| conv[ch * t + ti]).sum::<f32>() / c as f32;
                let var = (0..c)
                    .map(|ch| (conv[ch * t + ti] - mean).powi(2))
                    .sum::<f32>()
                    / c as f32;
                let inv = 1.0 / (var + 1e-5).sqrt();
                for ch in 0..c {
                    ln[ch * t + ti] = (conv[ch * t + ti] - mean) * inv * gamma[ch] + beta[ch];
                }
            }
            leaky_relu(&mut ln, 0.2);
            x = ln;
        }

        // BiLSTM(512 → 256 bidir): needs row-major [T, C]
        let mut x_rm = vec![0.0f32; t * c];
        for ch in 0..c {
            for ti in 0..t {
                x_rm[ti * c + ch] = x[ch * t + ti];
            }
        }
        let lw = self.load_bilstm("k.text_encoder.lstm");
        let lstm = self.run_bilstm(&lw, &x_rm, t, c, c / 2); // [T, 512]
        // back to channel-major [C, T]
        let mut out = vec![0.0f32; c * t];
        for ti in 0..t {
            for ch in 0..c {
                out[ch * t + ti] = lstm[ti * c + ch];
            }
        }
        out
    }
}
