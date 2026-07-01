//! PL-BERT = a 12-layer ALBERT (shared weights) producing prosody features.
//! Mirrors HF `AlbertModel` (gelu_new, layer_norm_eps=1e-12, absolute positions).
#![allow(dead_code)]

use super::KokoroModel;
use super::ops::{gelu_new, layer_norm, linear, softmax};

const EPS: f32 = 1e-12;
const EMB: usize = 128; // ALBERT embedding_size

impl KokoroModel {
    /// ALBERT last_hidden_state for `input_ids`. Returns `[T, hidden]` row-major.
    /// Batch=1, no attention mask (full-length sequence).
    pub fn bert(&self, input_ids: &[i64]) -> Vec<f32> {
        let t = input_ids.len();
        let h = self.cfg.plbert_hidden; // 768
        let heads = self.cfg.plbert_heads; // 12
        let hd = h / heads; // 64

        // ---- embeddings: word + position + token_type, then LN(128) ----
        let word = self.t("k.bert.embeddings.word_embeddings.weight"); // [n_token, 128]
        let pos = self.t("k.bert.embeddings.position_embeddings.weight"); // [512, 128]
        let tok = self.t("k.bert.embeddings.token_type_embeddings.weight"); // [2, 128]
        let mut emb = vec![0.0f32; t * EMB];
        for (p, &id) in input_ids.iter().enumerate() {
            let wrow = &word[id as usize * EMB..(id as usize + 1) * EMB];
            let prow = &pos[p * EMB..(p + 1) * EMB];
            for d in 0..EMB {
                emb[p * EMB + d] = wrow[d] + prow[d] + tok[d]; // token_type 0
            }
        }
        let ln_w = self.t("k.bert.embeddings.LayerNorm.weight");
        let ln_b = self.t("k.bert.embeddings.LayerNorm.bias");
        let emb = layer_norm(&emb, t, EMB, &ln_w, &ln_b, EPS);

        // ---- project 128 -> 768 ----
        let map_w = self.t("k.bert.encoder.embedding_hidden_mapping_in.weight");
        let map_b = self.t("k.bert.encoder.embedding_hidden_mapping_in.bias");
        let mut hidden = linear(&emb, t, EMB, &map_w, Some(&map_b), h);

        // ---- shared ALBERT layer weights (reused 12x) ----
        let p = "k.bert.encoder.albert_layer_groups.0.albert_layers.0.";
        let qw = self.t(&format!("{p}attention.query.weight"));
        let qb = self.t(&format!("{p}attention.query.bias"));
        let kw = self.t(&format!("{p}attention.key.weight"));
        let kb = self.t(&format!("{p}attention.key.bias"));
        let vw = self.t(&format!("{p}attention.value.weight"));
        let vb = self.t(&format!("{p}attention.value.bias"));
        let dw = self.t(&format!("{p}attention.dense.weight"));
        let db = self.t(&format!("{p}attention.dense.bias"));
        let aln_w = self.t(&format!("{p}attention.LayerNorm.weight"));
        let aln_b = self.t(&format!("{p}attention.LayerNorm.bias"));
        let fw = self.t(&format!("{p}ffn.weight"));
        let fb = self.t(&format!("{p}ffn.bias"));
        let fow = self.t(&format!("{p}ffn_output.weight"));
        let fob = self.t(&format!("{p}ffn_output.bias"));
        let flw = self.t(&format!("{p}full_layer_layer_norm.weight"));
        let flb = self.t(&format!("{p}full_layer_layer_norm.bias"));
        let inter = self.cfg.plbert_inter; // 2048
        let scale = 1.0 / (hd as f32).sqrt();

        for _layer in 0..self.cfg.plbert_layers {
            let q = linear(&hidden, t, h, &qw, Some(&qb), h);
            let k = linear(&hidden, t, h, &kw, Some(&kb), h);
            let v = linear(&hidden, t, h, &vw, Some(&vb), h);

            // multi-head self-attention → context [T, h]
            let mut ctx = vec![0.0f32; t * h];
            let mut scores = vec![0.0f32; t];
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

            // output projection + residual + LN
            let proj = linear(&ctx, t, h, &dw, Some(&db), h);
            let mut attn_in = vec![0.0f32; t * h];
            for idx in 0..t * h {
                attn_in[idx] = proj[idx] + hidden[idx];
            }
            let attn_out = layer_norm(&attn_in, t, h, &aln_w, &aln_b, EPS);

            // FFN: 768 -> 2048 (gelu_new) -> 768, residual + LN
            let mut ff = linear(&attn_out, t, h, &fw, Some(&fb), inter);
            gelu_new(&mut ff);
            let ffo = linear(&ff, t, inter, &fow, Some(&fob), h);
            let mut ffo_res = vec![0.0f32; t * h];
            for idx in 0..t * h {
                ffo_res[idx] = ffo[idx] + attn_out[idx];
            }
            hidden = layer_norm(&ffo_res, t, h, &flw, &flb, EPS);
        }

        hidden
    }
}
