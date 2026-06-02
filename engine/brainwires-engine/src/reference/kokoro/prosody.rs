//! ProsodyPredictor front half: bert_encoder projection, DurationEncoder
//! (BiLSTM + AdaLayerNorm stack), and duration prediction. Mirrors modules.py.
#![allow(dead_code)]

use super::ops::{bilstm, layer_norm_plain, linear, sigmoid};
use super::KokoroModel;

/// 8 BiLSTM tensors for a `{prefix}` (PyTorch l0 + l0_reverse).
type Bilstm = [Vec<f32>; 8];

impl KokoroModel {
    fn load_bilstm(&self, prefix: &str) -> Bilstm {
        [
            self.t(&format!("{prefix}.weight_ih_l0")),
            self.t(&format!("{prefix}.weight_hh_l0")),
            self.t(&format!("{prefix}.bias_ih_l0")),
            self.t(&format!("{prefix}.bias_hh_l0")),
            self.t(&format!("{prefix}.weight_ih_l0_reverse")),
            self.t(&format!("{prefix}.weight_hh_l0_reverse")),
            self.t(&format!("{prefix}.bias_ih_l0_reverse")),
            self.t(&format!("{prefix}.bias_hh_l0_reverse")),
        ]
    }

    fn run_bilstm(&self, w: &Bilstm, x: &[f32], t: usize, in_dim: usize, hidden: usize) -> Vec<f32> {
        bilstm(x, t, in_dim, hidden, &w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7])
    }

    /// Linear 768 -> 512. Returns `[T, 512]` (the un-transposed bert_encoder output).
    pub fn bert_encoder(&self, bert: &[f32], t: usize) -> Vec<f32> {
        let h = self.cfg.plbert_hidden; // 768
        let d = self.cfg.hidden_dim; // 512
        let w = self.t("k.bert_encoder.weight");
        let b = self.t("k.bert_encoder.bias");
        linear(bert, t, h, &w, Some(&b), d)
    }

    /// DurationEncoder: input bert_encoder `[T,512]` + prosodic style `[128]`,
    /// output `d [T, 640]` (512 + concatenated style). 3× (BiLSTM → AdaLayerNorm).
    pub fn duration_encode(&self, be: &[f32], t: usize, style: &[f32]) -> Vec<f32> {
        let d = self.cfg.hidden_dim; // 512
        let sd = self.cfg.style_dim; // 128
        let cat = d + sd; // 640
        // x[t] = concat(be[t], style)
        let mut x = vec![0.0f32; t * cat];
        for ti in 0..t {
            x[ti * cat..ti * cat + d].copy_from_slice(&be[ti * d..(ti + 1) * d]);
            x[ti * cat + d..(ti + 1) * cat].copy_from_slice(style);
        }
        for layer in 0..self.cfg.n_layer {
            // BiLSTM block (lstms.{0,2,4}): [T,640] -> [T,512]
            let lw = self.load_bilstm(&format!("k.predictor.text_encoder.lstms.{}", 2 * layer));
            let lstm_out = self.run_bilstm(&lw, &x, t, cat, d / 2); // hidden 256 -> out 512

            // AdaLayerNorm block (lstms.{1,3,5}): per-t LN over 512 + (1+gamma)*+beta
            let fc_w = self.t(&format!("k.predictor.text_encoder.lstms.{}.fc.weight", 2 * layer + 1));
            let fc_b = self.t(&format!("k.predictor.text_encoder.lstms.{}.fc.bias", 2 * layer + 1));
            let gb = linear(style, 1, sd, &fc_w, Some(&fc_b), 2 * d); // [1024]
            let (gamma, beta) = gb.split_at(d);
            let ln = layer_norm_plain(&lstm_out, t, d, 1e-5);
            // out = concat((1+gamma)*ln + beta, style) -> [T,640]
            for ti in 0..t {
                for c in 0..d {
                    x[ti * cat + c] = (1.0 + gamma[c]) * ln[ti * d + c] + beta[c];
                }
                x[ti * cat + d..(ti + 1) * cat].copy_from_slice(style);
            }
        }
        x
    }

    /// duration_proj path: predictor.lstm (BiLSTM 640->512) then Linear 512->max_dur.
    /// Returns (duration_logits `[T, max_dur]`, pred_dur `[T]`).
    pub fn predict_duration(&self, d: &[f32], t: usize) -> (Vec<f32>, Vec<usize>) {
        let cat = self.cfg.hidden_dim + self.cfg.style_dim; // 640
        let hid = self.cfg.hidden_dim; // 512
        let lw = self.load_bilstm("k.predictor.lstm");
        let x = self.run_bilstm(&lw, d, t, cat, hid / 2); // [T,512]
        let w = self.t("k.predictor.duration_proj.linear_layer.weight");
        let b = self.t("k.predictor.duration_proj.linear_layer.bias");
        let logits = linear(&x, t, hid, &w, Some(&b), self.cfg.max_dur); // [T, max_dur]
        let mut pred_dur = vec![0usize; t];
        for ti in 0..t {
            let s: f32 = logits[ti * self.cfg.max_dur..(ti + 1) * self.cfg.max_dur]
                .iter()
                .map(|&v| sigmoid(v))
                .sum();
            pred_dur[ti] = s.round().max(1.0) as usize;
        }
        (logits, pred_dur)
    }
}
