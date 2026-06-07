//! SentencePiece (SPM) tokenizer for unigram-scored GGUFs — EmbeddingGemma.
//!
//! EmbeddingGemma's GGUF ships `tokenizer.ggml.scores` (real unigram log-probs)
//! and NO `tokenizer.ggml.merges`, unlike Ollama's gemma4 GGUF which is BPE.
//! Same 262144-token Gemma vocabulary, different segmentation algorithm.
//!
//! Mirrors llama.cpp's `llm_tokenizer_spm`: greedy merge of the highest-scoring
//! adjacent symbol pair (priority queue), then per-symbol resegmentation with
//! byte fallback. This is what produces Ollama's `/api/embed` token stream, so
//! matching it is required for embedding parity.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;

use crate::error::{Result, RullamaError};
use crate::gguf::GgufReader;

use super::bpe::{SPM_SPACE, TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED};

pub struct SpmTokenizer {
    vocab: HashMap<String, u32>,
    rev_vocab: Vec<String>,
    scores: Vec<f32>,
    /// Specials (CONTROL + USER_DEFINED), longest-first.
    specials: Vec<(String, u32)>,
    byte_fallback: [Option<u32>; 256],
    unk_id: Option<u32>,
    /// SPM dummy-prefix: prepend a `▁` to the whole input. `false` for
    /// EmbeddingGemma (`tokenizer.ggml.add_space_prefix = false`).
    add_space_prefix: bool,
}

impl SpmTokenizer {
    pub fn from_gguf(r: &GgufReader) -> Result<Self> {
        let tokens = r.get("tokenizer.ggml.tokens")?.as_string_array()?.to_vec();
        let types = r.get("tokenizer.ggml.token_type")?.as_u32_array()?;
        let scores_raw = r.get("tokenizer.ggml.scores").map_err(|_| {
            RullamaError::Tokenizer("SPM tokenizer requires tokenizer.ggml.scores".into())
        })?;
        // scores is a float array the length of the vocab.
        let scores: Vec<f32> = scores_raw
            .as_f32_array()
            .map_err(|e| RullamaError::Tokenizer(format!("scores array: {e}")))?;
        if scores.len() != tokens.len() {
            return Err(RullamaError::Tokenizer(format!(
                "scores len {} != tokens len {}",
                scores.len(),
                tokens.len()
            )));
        }

        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            vocab.insert(t.clone(), i as u32);
        }

        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                types[*i] == TOKEN_TYPE_CONTROL || types[*i] == TOKEN_TYPE_USER_DEFINED
            })
            .map(|(i, s)| (s.clone(), i as u32))
            .filter(|(s, _)| !s.is_empty())
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));

        let mut byte_fallback = [None; 256];
        for b in 0u32..256 {
            let key = format!("<0x{:02X}>", b);
            if let Some(&id) = vocab.get(&key) {
                byte_fallback[b as usize] = Some(id);
            }
        }

        let unk_id = r
            .get("tokenizer.ggml.unknown_token_id")
            .ok()
            .and_then(|v| v.as_u32().ok());
        let add_space_prefix = r
            .get("tokenizer.ggml.add_space_prefix")
            .ok()
            .and_then(|v| v.as_bool().ok())
            .unwrap_or(true);

        Ok(Self {
            vocab,
            rev_vocab: tokens,
            scores,
            specials,
            byte_fallback,
            unk_id,
            add_space_prefix,
        })
    }

    pub fn vocab_size(&self) -> u32 {
        self.rev_vocab.len() as u32
    }

    pub fn id_to_str(&self, id: u32) -> Option<&str> {
        self.rev_vocab.get(id as usize).map(|s| s.as_str())
    }

    /// Encode a UTF-8 string into token ids. Does NOT add BOS/EOS — the caller
    /// wraps per `tokenizer.ggml.add_bos_token` / `add_eos_token`.
    pub fn encode(&self, s: &str) -> Vec<u32> {
        // Split around specials first; specials emit their id directly.
        let mut frags: Vec<Frag> = vec![Frag::Text(s.to_string())];
        for (special, sid) in &self.specials {
            let mut next = Vec::new();
            for f in frags.into_iter() {
                match f {
                    Frag::Special(_) => next.push(f),
                    Frag::Text(t) => split_around(&t, special, *sid, &mut next),
                }
            }
            frags = next;
        }

        let mut out = Vec::new();
        for f in frags {
            match f {
                Frag::Special(id) => out.push(id),
                Frag::Text(t) => self.encode_text(&t, &mut out),
            }
        }
        out
    }

    /// SPM tokenize a plain text fragment (no specials inside).
    fn encode_text(&self, raw: &str, out: &mut Vec<u32>) {
        if raw.is_empty() {
            return;
        }
        // Normalize: ' ' → '▁'. Optionally prepend a dummy '▁'.
        let mut norm = String::with_capacity(raw.len() + 3);
        if self.add_space_prefix {
            norm.push(SPM_SPACE);
        }
        for c in raw.chars() {
            norm.push(if c == ' ' { SPM_SPACE } else { c });
        }

        // Build the symbol list: one UTF-8 char per symbol, as a doubly-linked
        // list over byte ranges into `norm`.
        let bytes = norm.as_bytes();
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut idx = 0usize;
        while idx < bytes.len() {
            let ch_len = utf8_len(bytes[idx]);
            let n = ch_len.min(bytes.len() - idx);
            let prev = symbols.len() as i64 - 1;
            symbols.push(Symbol {
                start: idx,
                len: n,
                prev,
                next: -1, // patched below
            });
            idx += n;
        }
        let count = symbols.len();
        for (i, sym) in symbols.iter_mut().enumerate() {
            sym.next = if i + 1 < count { i as i64 + 1 } else { -1 };
        }

        // Seed the priority queue with all adjacent bigrams that exist in vocab.
        let mut pq: BinaryHeap<Bigram> = BinaryHeap::new();
        for i in 1..symbols.len() {
            self.try_add_bigram(&symbols, i as i64 - 1, i as i64, bytes, &mut pq);
        }

        // Greedy merge by score.
        while let Some(bg) = pq.pop() {
            let (li, ri) = (bg.left as usize, bg.right as usize);
            let left_len = symbols[li].len;
            let right_len = symbols[ri].len;
            // Stale entry: a symbol involved was already merged away.
            if left_len == 0 || right_len == 0 || left_len + right_len != bg.size {
                continue;
            }
            // Merge right into left.
            symbols[li].len += symbols[ri].len;
            symbols[ri].len = 0;
            symbols[li].next = symbols[ri].next;
            let rnext = symbols[ri].next;
            if rnext >= 0 {
                symbols[rnext as usize].prev = bg.left;
            }
            // New candidate bigrams around the merged symbol.
            let lprev = symbols[li].prev;
            self.try_add_bigram(&symbols, lprev, bg.left, bytes, &mut pq);
            self.try_add_bigram(&symbols, bg.left, symbols[li].next, bytes, &mut pq);
        }

        // Walk the linked list and emit ids, with byte fallback.
        let mut i: i64 = 0;
        // find head (prev == -1)
        while i >= 0 && symbols[i as usize].prev != -1 {
            i = symbols[i as usize].prev;
        }
        // safety: head is index 0 by construction unless all merged; start at 0
        i = 0;
        while i >= 0 {
            let sym = &symbols[i as usize];
            if sym.len > 0 {
                let text = &norm[sym.start..sym.start + sym.len];
                self.resegment(text, out);
            }
            i = sym.next;
        }
    }

    fn try_add_bigram(
        &self,
        symbols: &[Symbol],
        left: i64,
        right: i64,
        bytes: &[u8],
        pq: &mut BinaryHeap<Bigram>,
    ) {
        if left < 0 || right < 0 {
            return;
        }
        let l = &symbols[left as usize];
        let r = &symbols[right as usize];
        if l.len == 0 || r.len == 0 {
            return;
        }
        let start = l.start;
        let end = r.start + r.len;
        let text = std::str::from_utf8(&bytes[start..end]).unwrap_or("");
        if let Some(&id) = self.vocab.get(text) {
            pq.push(Bigram {
                left,
                right,
                score: self.scores[id as usize],
                size: end - start,
            });
        }
    }

    /// Look up a final symbol's text; emit its id or byte-fallback to `<0xHH>`.
    fn resegment(&self, text: &str, out: &mut Vec<u32>) {
        if let Some(&id) = self.vocab.get(text) {
            out.push(id);
            return;
        }
        // Byte fallback: emit each UTF-8 byte as its `<0xHH>` token.
        for &b in text.as_bytes() {
            if let Some(id) = self.byte_fallback[b as usize] {
                out.push(id);
            } else if let Some(unk) = self.unk_id {
                out.push(unk);
            }
        }
    }
}

struct Symbol {
    start: usize,
    len: usize,
    prev: i64,
    next: i64,
}

/// A merge candidate. The `BinaryHeap` is a max-heap, so `Ord` ranks higher
/// score first; on a tie, the left-most (smaller `left`) wins — matching
/// llama.cpp's bigram comparator.
struct Bigram {
    left: i64,
    right: i64,
    score: f32,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score && self.left == o.left
    }
}
impl Eq for Bigram {}
impl Ord for Bigram {
    fn cmp(&self, o: &Self) -> Ordering {
        // Higher score = greater. Tie → smaller `left` = greater (pops first).
        match self.score.partial_cmp(&o.score).unwrap_or(Ordering::Equal) {
            Ordering::Equal => o.left.cmp(&self.left),
            ord => ord,
        }
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

enum Frag {
    Text(String),
    Special(u32),
}

fn split_around(text: &str, special: &str, sid: u32, out: &mut Vec<Frag>) {
    if text.is_empty() {
        return;
    }
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

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}
