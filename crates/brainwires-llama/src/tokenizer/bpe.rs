//! Byte-Pair-Encoding tokenizer for Gemma 4 GGUFs.
//!
//! Mirrors the active path of `tokenizer/bytepairencoding.go` in the Ollama reference
//! impl (with `spaceToSpmSep=true` and no regex pretokenizer — that's how Gemma 4 is
//! constructed in `model/models/gemma4/model.go:76`).
//!
//! Algorithm:
//!   1. Split the input around special tokens (CONTROL or USER_DEFINED). Specials
//!      emit their id directly and don't go through BPE.
//!   2. For each non-special fragment, replace ASCII space with U+2581 (`▁`),
//!      SentencePiece-style.
//!   3. Short-circuit: if the whole fragment is already a vocab entry, emit its id.
//!   4. Otherwise, run BPE merges (lowest-rank adjacent pair wins, repeat until
//!      no further merges possible).
//!   5. For any leftover string not in the vocab, fall back to per-byte
//!      `<0xHH>` tokens.
//!
//! Performance is O(n²·V) for an input of length n where V is the merge map. Fine for
//! a CPU reference path; if we ever need fast tokenization in the browser we'll
//! upgrade to the heap-based approach Ollama uses.

use std::collections::HashMap;

use crate::error::{Result, RullamaError};
use crate::gguf::GgufReader;

/// Token-type codes from `tokenizer.ggml.token_type`. Match `GGUF_TOKEN_TYPE_*` in llama.cpp.
pub const TOKEN_TYPE_NORMAL: u32 = 1;
pub const TOKEN_TYPE_UNKNOWN: u32 = 2;
pub const TOKEN_TYPE_CONTROL: u32 = 3;
pub const TOKEN_TYPE_USER_DEFINED: u32 = 4;
pub const TOKEN_TYPE_UNUSED: u32 = 5;
pub const TOKEN_TYPE_BYTE: u32 = 6;

/// SentencePiece whitespace replacement.
pub const SPM_SPACE: char = '▁'; // U+2581

pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    rev_vocab: Vec<String>,
    /// (left, right) → rank. Lower rank is preferred (earlier in the merges list).
    merges: HashMap<(String, String), u32>,
    /// Specials, sorted by length descending so longest-match wins when scanning.
    specials: Vec<(String, u32)>,
    /// Byte-fallback table: byte → token id of `<0xHH>`. Empty if absent in vocab.
    byte_fallback: [Option<u32>; 256],
}

impl BpeTokenizer {
    /// Build a tokenizer from the metadata embedded in a GGUF file. Reads
    /// `tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`, `tokenizer.ggml.merges`.
    pub fn from_gguf(r: &GgufReader) -> Result<Self> {
        let tokens = r.get("tokenizer.ggml.tokens")?.as_string_array()?.to_vec();
        let types = r.get("tokenizer.ggml.token_type")?.as_u32_array()?;
        if types.len() != tokens.len() {
            return Err(RullamaError::Tokenizer(format!(
                "token_type len {} != tokens len {}", types.len(), tokens.len()
            )));
        }

        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            vocab.insert(t.clone(), i as u32);
        }

        // Collect specials (CONTROL + USER_DEFINED), sort by length desc for longest-match.
        let mut specials: Vec<(String, u32)> = tokens.iter().enumerate()
            .filter(|(i, _)| types[*i] == TOKEN_TYPE_CONTROL || types[*i] == TOKEN_TYPE_USER_DEFINED)
            .map(|(i, s)| (s.clone(), i as u32))
            .filter(|(s, _)| !s.is_empty())
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));

        // Merges: each entry is "left right" (space-separated). Rank = index.
        let merge_strs = r.get("tokenizer.ggml.merges")?.as_string_array()?;
        let mut merges: HashMap<(String, String), u32> = HashMap::with_capacity(merge_strs.len());
        for (rank, m) in merge_strs.iter().enumerate() {
            // The merge string is "left right"; split on the first ASCII space.
            // Some merge entries contain ▁ characters (the SP space marker) but not
            // ASCII space, so a split on ' ' is unambiguous.
            if let Some(sp) = m.find(' ') {
                let left = m[..sp].to_string();
                let right = m[sp + 1..].to_string();
                merges.insert((left, right), rank as u32);
            }
        }

        // Byte-fallback table: lookup `<0xHH>` for each possible byte.
        let mut byte_fallback = [None; 256];
        for b in 0u32..256 {
            let key = format!("<0x{:02X}>", b);
            if let Some(&id) = vocab.get(&key) {
                byte_fallback[b as usize] = Some(id);
            }
        }

        Ok(Self {
            vocab,
            rev_vocab: tokens,
            merges,
            specials,
            byte_fallback,
        })
    }

    pub fn vocab_size(&self) -> u32 { self.rev_vocab.len() as u32 }

    pub fn id_to_str(&self, id: u32) -> Option<&str> {
        self.rev_vocab.get(id as usize).map(|s| s.as_str())
    }

    /// Encode a UTF-8 string into token ids.
    pub fn encode(&self, s: &str) -> Vec<u32> {
        // ----- 1) split around special tokens -----
        let mut frags: Vec<Frag> = vec![Frag::Text(s.to_string())];
        for (special, sid) in &self.specials {
            let mut next: Vec<Frag> = Vec::new();
            for f in frags.into_iter() {
                match f {
                    Frag::Special(_) => next.push(f),
                    Frag::Text(t) => split_around(&t, special, *sid, &mut next),
                }
            }
            frags = next;
        }

        // ----- 2-5) encode each fragment -----
        let mut out = Vec::new();
        for f in frags {
            match f {
                Frag::Special(id) => out.push(id),
                Frag::Text(t) => self.encode_text(&t, &mut out),
            }
        }
        out
    }

    fn encode_text(&self, raw: &str, out: &mut Vec<u32>) {
        if raw.is_empty() { return; }

        // SP normalize: ' ' → '▁'
        let normalized: String = raw.chars().map(|c| if c == ' ' { SPM_SPACE } else { c }).collect();

        // short-circuit on full match
        if let Some(&id) = self.vocab.get(&normalized) {
            out.push(id);
            return;
        }

        // Initial token list: one entry per char (each as a String).
        let mut toks: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();

        // Repeatedly find the lowest-rank adjacent pair and merge.
        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx: i32 = -1;
            for i in 0..toks.len().saturating_sub(1) {
                if let Some(&rank) = self.merges.get(&(toks[i].clone(), toks[i + 1].clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i as i32;
                    }
                }
            }
            if best_idx < 0 { break; }
            let i = best_idx as usize;
            let merged = format!("{}{}", toks[i], toks[i + 1]);
            // Sanity: the merged result should be in vocab; if not, the merge entry
            // is stale/unusable, skip it. Mark as "tried" by removing the rank lookup
            // implicitly — we'll just not pick the same pair again because after the
            // merge, its (left, right) won't recur. Actually it could — for safety,
            // bail if not in vocab.
            if !self.vocab.contains_key(&merged) {
                // Drop this merge from consideration by removing it from a temporary
                // veto set; but the simplest fix is to break and accept the partial
                // tokenization. In practice every merges-list entry is also a vocab
                // entry, so this branch is defensive only.
                break;
            }
            toks[i] = merged;
            toks.remove(i + 1);
        }

        // Map final tokens to ids; fall back to per-byte for any leftover.
        for tok in toks {
            if let Some(&id) = self.vocab.get(&tok) {
                out.push(id);
            } else {
                // byte fallback
                for b in tok.as_bytes() {
                    if let Some(id) = self.byte_fallback[*b as usize] {
                        out.push(id);
                    } else {
                        // unknown byte and no fallback — best-effort: emit unk if present
                        log::debug!("unknown byte token: 0x{:02X}", b);
                    }
                }
            }
        }
    }
}

/// Internal: a fragment of the input being processed.
enum Frag {
    Text(String),
    Special(u32),
}

/// Find every occurrence of `special` in `text`, splitting `text` into Text/Special
/// pieces and pushing them onto `out`.
fn split_around(text: &str, special: &str, sid: u32, out: &mut Vec<Frag>) {
    if text.is_empty() { return; }
    if special.is_empty() {
        out.push(Frag::Text(text.to_string()));
        return;
    }
    let mut start = 0usize;
    while let Some(pos) = text[start..].find(special) {
        let abs = start + pos;
        if abs > start {
            out.push(Frag::Text(text[start..abs].to_string()));
        }
        out.push(Frag::Special(sid));
        start = abs + special.len();
    }
    if start < text.len() {
        out.push(Frag::Text(text[start..].to_string()));
    }
}
