//! StyleTTS2 acoustic graph — CPU f32 oracle. text_encoder + bert(PLBERT) +
//! bert_encoder + predictor(duration, F0/N) + length regulator, then the hifigan
//! decoder. Architecturally identical to the Kokoro reference modules
//! (text_encoder.rs/bert.rs/prosody.rs) — same logic, StyleTTS2 weight names (no `k.`
//! prefix) and the hifigan 1-frame asr shift. Validated against the full-synthesis
//! fixtures (scripts/styletts2_dump_synth_fixtures.py).
#![allow(dead_code)]

use std::collections::HashMap;

use super::decoder::StyleTtsDecoder;
use crate::reference::kokoro::convblocks::conv1d;
use crate::reference::kokoro::ops::{bilstm, gelu_new, layer_norm, layer_norm_plain, leaky_relu, linear, sigmoid, softmax};

const HIDDEN: usize = 512;
const N_LAYER: usize = 3;
const TE_K: usize = 5; // text_encoder conv kernel
const STYLE_DIM: usize = 128;
const MAX_DUR: usize = 50;
const PLBERT_HID: usize = 768;
const PLBERT_HEADS: usize = 12;
const PLBERT_LAYERS: usize = 12;
const PLBERT_INTER: usize = 2048;
const EMB: usize = 128; // ALBERT embedding size
const EPS_BERT: f32 = 1e-12;

pub struct StyleTtsAcoustic<'a> {
    w: &'a HashMap<String, Vec<f32>>,
}

impl<'a> StyleTtsAcoustic<'a> {
    pub fn new(w: &'a HashMap<String, Vec<f32>>) -> Self {
        Self { w }
    }
    fn t(&self, n: &str) -> &[f32] {
        self.w.get(n).unwrap_or_else(|| panic!("missing acoustic weight: {n}"))
    }
    fn bilstm_run(&self, prefix: &str, x: &[f32], t: usize, in_dim: usize, hidden: usize) -> Vec<f32> {
        let g = |s: &str| self.t(&format!("{prefix}.{s}"));
        bilstm(x, t, in_dim, hidden,
            g("weight_ih_l0"), g("weight_hh_l0"), g("bias_ih_l0"), g("bias_hh_l0"),
            g("weight_ih_l0_reverse"), g("weight_hh_l0_reverse"), g("bias_ih_l0_reverse"), g("bias_hh_l0_reverse"))
    }

    /// TextEncoder: embedding → 3×(Conv1d k5 + channel-LayerNorm + LeakyReLU) → BiLSTM.
    /// Returns `t_en [512, T]` channel-major.
    pub fn text_encoder(&self, ids: &[i64]) -> Vec<f32> {
        let (t, c) = (ids.len(), HIDDEN);
        let emb = self.t("text_encoder.embedding.weight"); // [178, 512]
        let mut x = vec![0f32; c * t];
        for (ti, &id) in ids.iter().enumerate() {
            for ch in 0..c {
                x[ch * t + ti] = emb[id as usize * c + ch];
            }
        }
        for i in 0..N_LAYER {
            let cw = self.t(&format!("text_encoder.cnn.{i}.0.weight"));
            let cb = self.t(&format!("text_encoder.cnn.{i}.0.bias"));
            let (conv, _) = conv1d(&x, c, t, cw, Some(cb), c, TE_K, 1, (TE_K - 1) / 2, 1, 1);
            let gamma = self.t(&format!("text_encoder.cnn.{i}.1.gamma"));
            let beta = self.t(&format!("text_encoder.cnn.{i}.1.beta"));
            let mut ln = vec![0f32; c * t];
            for ti in 0..t {
                let mean = (0..c).map(|ch| conv[ch * t + ti]).sum::<f32>() / c as f32;
                let var = (0..c).map(|ch| (conv[ch * t + ti] - mean).powi(2)).sum::<f32>() / c as f32;
                let inv = 1.0 / (var + 1e-5).sqrt();
                for ch in 0..c {
                    ln[ch * t + ti] = (conv[ch * t + ti] - mean) * inv * gamma[ch] + beta[ch];
                }
            }
            leaky_relu(&mut ln, 0.2);
            x = ln;
        }
        // BiLSTM(512→256 bidir) on row-major [T,512]
        let mut x_rm = vec![0f32; t * c];
        for ch in 0..c {
            for ti in 0..t {
                x_rm[ti * c + ch] = x[ch * t + ti];
            }
        }
        let lstm = self.bilstm_run("text_encoder.lstm", &x_rm, t, c, c / 2); // [T,512]
        let mut out = vec![0f32; c * t];
        for ti in 0..t {
            for ch in 0..c {
                out[ch * t + ti] = lstm[ti * c + ch];
            }
        }
        out
    }

    /// PL-BERT (ALBERT, shared 12 layers, gelu_new, eps 1e-12). Returns `[T, 768]` row-major.
    pub fn bert(&self, ids: &[i64], progress: Option<&dyn Fn(f32, &str)>) -> Vec<f32> {
        let t = ids.len();
        let (h, heads) = (PLBERT_HID, PLBERT_HEADS);
        let hd = h / heads;
        let word = self.t("bert.embeddings.word_embeddings.weight");
        let pos = self.t("bert.embeddings.position_embeddings.weight");
        let tok = self.t("bert.embeddings.token_type_embeddings.weight");
        let mut emb = vec![0f32; t * EMB];
        for (p, &id) in ids.iter().enumerate() {
            for d in 0..EMB {
                emb[p * EMB + d] = word[id as usize * EMB + d] + pos[p * EMB + d] + tok[d];
            }
        }
        let emb = layer_norm(&emb, t, EMB, self.t("bert.embeddings.LayerNorm.weight"), self.t("bert.embeddings.LayerNorm.bias"), EPS_BERT);
        let mut hidden = linear(&emb, t, EMB, self.t("bert.encoder.embedding_hidden_mapping_in.weight"), Some(self.t("bert.encoder.embedding_hidden_mapping_in.bias")), h);

        let p = "bert.encoder.albert_layer_groups.0.albert_layers.0.";
        let (qw, qb) = (self.t(&format!("{p}attention.query.weight")), self.t(&format!("{p}attention.query.bias")));
        let (kw, kb) = (self.t(&format!("{p}attention.key.weight")), self.t(&format!("{p}attention.key.bias")));
        let (vw, vb) = (self.t(&format!("{p}attention.value.weight")), self.t(&format!("{p}attention.value.bias")));
        let (dw, db) = (self.t(&format!("{p}attention.dense.weight")), self.t(&format!("{p}attention.dense.bias")));
        let (aw, ab) = (self.t(&format!("{p}attention.LayerNorm.weight")), self.t(&format!("{p}attention.LayerNorm.bias")));
        let (fw, fb) = (self.t(&format!("{p}ffn.weight")), self.t(&format!("{p}ffn.bias")));
        let (fow, fob) = (self.t(&format!("{p}ffn_output.weight")), self.t(&format!("{p}ffn_output.bias")));
        let (flw, flb) = (self.t(&format!("{p}full_layer_layer_norm.weight")), self.t(&format!("{p}full_layer_layer_norm.bias")));
        let scale = 1.0 / (hd as f32).sqrt();

        for layer in 0..PLBERT_LAYERS {
            if let Some(p) = progress {
                p(0.05 + 0.13 * layer as f32 / PLBERT_LAYERS as f32, "analyzing text");
            }
            let q = linear(&hidden, t, h, qw, Some(qb), h);
            let k = linear(&hidden, t, h, kw, Some(kb), h);
            let v = linear(&hidden, t, h, vw, Some(vb), h);
            let mut ctx = vec![0f32; t * h];
            let mut scores = vec![0f32; t];
            for head in 0..heads {
                let off = head * hd;
                for i in 0..t {
                    for j in 0..t {
                        let mut acc = 0.0;
                        for d in 0..hd {
                            acc += q[i * h + off + d] * k[j * h + off + d];
                        }
                        scores[j] = acc * scale;
                    }
                    softmax(&mut scores);
                    for d in 0..hd {
                        let mut acc = 0.0;
                        for j in 0..t {
                            acc += scores[j] * v[j * h + off + d];
                        }
                        ctx[i * h + off + d] = acc;
                    }
                }
            }
            let proj = linear(&ctx, t, h, dw, Some(db), h);
            let attn_in: Vec<f32> = proj.iter().zip(&hidden).map(|(a, b)| a + b).collect();
            let attn_out = layer_norm(&attn_in, t, h, aw, ab, EPS_BERT);
            let mut ff = linear(&attn_out, t, h, fw, Some(fb), PLBERT_INTER);
            gelu_new(&mut ff);
            let ffo = linear(&ff, t, PLBERT_INTER, fow, Some(fob), h);
            let ffo_res: Vec<f32> = ffo.iter().zip(&attn_out).map(|(a, b)| a + b).collect();
            hidden = layer_norm(&ffo_res, t, h, flw, flb, EPS_BERT);
        }
        hidden
    }

    /// bert_encoder Linear 768→512. Returns `[T, 512]` row-major.
    pub fn bert_encoder(&self, bert: &[f32], t: usize) -> Vec<f32> {
        linear(bert, t, PLBERT_HID, self.t("bert_encoder.weight"), Some(self.t("bert_encoder.bias")), HIDDEN)
    }

    /// DurationEncoder: bert_encoder `[T,512]` + prosodic style → `d [T,640]`.
    pub fn duration_encode(&self, be: &[f32], t: usize, style: &[f32]) -> Vec<f32> {
        let cat = HIDDEN + STYLE_DIM; // 640
        let mut x = vec![0f32; t * cat];
        for ti in 0..t {
            x[ti * cat..ti * cat + HIDDEN].copy_from_slice(&be[ti * HIDDEN..(ti + 1) * HIDDEN]);
            x[ti * cat + HIDDEN..(ti + 1) * cat].copy_from_slice(style);
        }
        for layer in 0..N_LAYER {
            let lstm_out = self.bilstm_run(&format!("predictor.text_encoder.lstms.{}", 2 * layer), &x, t, cat, HIDDEN / 2);
            let fc_w = self.t(&format!("predictor.text_encoder.lstms.{}.fc.weight", 2 * layer + 1));
            let fc_b = self.t(&format!("predictor.text_encoder.lstms.{}.fc.bias", 2 * layer + 1));
            let gb = linear(style, 1, STYLE_DIM, fc_w, Some(fc_b), 2 * HIDDEN);
            let (gamma, beta) = gb.split_at(HIDDEN);
            let ln = layer_norm_plain(&lstm_out, t, HIDDEN, 1e-5);
            for ti in 0..t {
                for c in 0..HIDDEN {
                    x[ti * cat + c] = (1.0 + gamma[c]) * ln[ti * HIDDEN + c] + beta[c];
                }
                x[ti * cat + HIDDEN..(ti + 1) * cat].copy_from_slice(style);
            }
        }
        x
    }

    /// predictor.lstm (BiLSTM 640→512) → duration_proj (Linear 512→50) → sigmoid·sum → round.
    pub fn predict_duration(&self, d: &[f32], t: usize) -> Vec<usize> {
        let cat = HIDDEN + STYLE_DIM;
        let x = self.bilstm_run("predictor.lstm", d, t, cat, HIDDEN / 2);
        let logits = linear(&x, t, HIDDEN, self.t("predictor.duration_proj.linear_layer.weight"), Some(self.t("predictor.duration_proj.linear_layer.bias")), MAX_DUR);
        (0..t)
            .map(|ti| {
                let s: f32 = logits[ti * MAX_DUR..(ti + 1) * MAX_DUR].iter().map(|&v| sigmoid(v)).sum();
                s.round().max(1.0) as usize
            })
            .collect()
    }

    /// Length regulator: expand row-major `feat [T,C]` to channel-major `[C,F]`, F=Σdur.
    pub fn expand_by_dur_cm(feat: &[f32], t: usize, c: usize, dur: &[usize]) -> (Vec<f32>, usize) {
        let f: usize = dur.iter().sum();
        let mut out = vec![0f32; c * f];
        let mut fi = 0;
        for ti in 0..t {
            for _ in 0..dur[ti] {
                for cc in 0..c {
                    out[cc * f + fi] = feat[ti * c + cc];
                }
                fi += 1;
            }
        }
        (out, f)
    }

    /// F0Ntrain: shared BiLSTM then F0/N AdainResBlk1d stacks. `en [640,F]` → (F0,N) each `[2F]`.
    pub fn f0_n(&self, en: &[f32], f: usize, style: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let cat = HIDDEN + STYLE_DIM;
        let mut x_rm = vec![0f32; f * cat];
        for ff in 0..f {
            for c in 0..cat {
                x_rm[ff * cat + c] = en[c * f + ff];
            }
        }
        let xs = self.bilstm_run("predictor.shared", &x_rm, f, cat, HIDDEN / 2); // [F,512]
        let mut x_cm = vec![0f32; HIDDEN * f];
        for ff in 0..f {
            for c in 0..HIDDEN {
                x_cm[c * f + ff] = xs[ff * HIDDEN + c];
            }
        }
        let half = HIDDEN / 2;
        let dec = StyleTtsDecoder::new(self.w);
        let run = |which: &str| -> Vec<f32> {
            let (h, t1) = dec.adain_resblk1d(&format!("predictor.{which}.0"), &x_cm, HIDDEN, f, HIDDEN, false, style);
            let (h, t2) = dec.adain_resblk1d(&format!("predictor.{which}.1"), &h, HIDDEN, t1, half, true, style);
            let (h, t3) = dec.adain_resblk1d(&format!("predictor.{which}.2"), &h, half, t2, half, false, style);
            conv1d(&h, half, t3, self.t(&format!("predictor.{which}_proj.weight")), Some(self.t(&format!("predictor.{which}_proj.bias"))), 1, 1, 1, 0, 1, 1).0
        };
        (run("F0"), run("N"))
    }

    /// Full zero-shot synthesis: token ids + reference style `ref_s [256]` → 24 kHz audio.
    /// `progress(fraction, stage)` is invoked at stage boundaries (the worker forwards it to the UI).
    pub fn synthesize(&self, ids: &[i64], ref_s: &[f32], progress: Option<&dyn Fn(f32, &str)>) -> Vec<f32> {
        let t = ids.len();
        let s = &ref_s[STYLE_DIM..]; // prosodic
        let r = &ref_s[..STYLE_DIM]; // acoustic
        let t_en = self.text_encoder(ids); // [512,T] cm
        let bert_out = self.bert(ids, progress);
        if let Some(p) = progress {
            p(0.20, "predicting rhythm");
        }
        let be = self.bert_encoder(&bert_out, t); // [T,512]
        let d = self.duration_encode(&be, t, s); // [T,640]
        let dur = self.predict_duration(&d, t);
        let (en, f) = Self::expand_by_dur_cm(&d, t, HIDDEN + STYLE_DIM, &dur); // [640,F]
        if let Some(p) = progress {
            p(0.28, "predicting pitch");
        }
        let (f0, n) = self.f0_n(&en, f, s); // each [2F]
        // asr = expand t_en by dur → [512,F]; convert t_en cm→rm first
        let mut ten_rm = vec![0f32; t * HIDDEN];
        for c in 0..HIDDEN {
            for ti in 0..t {
                ten_rm[ti * HIDDEN + c] = t_en[c * t + ti];
            }
        }
        let (asr, _) = Self::expand_by_dur_cm(&ten_rm, t, HIDDEN, &dur); // [512,F]
        // hifigan 1-frame shift along time
        let mut asr_s = vec![0f32; HIDDEN * f];
        for c in 0..HIDDEN {
            asr_s[c * f] = asr[c * f];
            for fi in 1..f {
                asr_s[c * f + fi] = asr[c * f + fi - 1];
            }
        }
        StyleTtsDecoder::new(self.w).forward(&asr_s, HIDDEN, f, &f0, &n, r, progress)
    }
}
