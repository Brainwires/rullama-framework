//! Minimal English G2P (text → IPA phonemes) for Kokoro, lexicon-first.
//!
//! v1: dictionary lookup against misaki's `us_gold.json` (+ `us_silver.json` fallback;
//! DEFAULT for POS-conditioned entries), plus a small suffix-morphology fallback
//! (stem + -s/-es/-ed/-ing), the/to vowel-conditioning, and punctuation passthrough.
//! Remaining misaki behaviour (full POS, sentence-stress, number expansion, espeak-ng
//! fallback) is DEFERRED — this matches misaki on common in-dictionary text and
//! diverges on those cases. OOV words are flagged. Output maps to ids via `phonemes_to_ids`.
#![allow(dead_code)]

use std::collections::HashMap;

/// word (lowercased) → DEFAULT phoneme string. gold takes priority over silver.
pub struct Lexicon(HashMap<String, String>);

fn extract_default(v: serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Object(o) => o.get("DEFAULT").and_then(|d| d.as_str().map(String::from)),
        _ => None,
    }
}

impl Lexicon {
    pub fn from_json(gold: &[u8]) -> Self {
        Self::load(gold, &[])
    }

    /// Merge gold (primary) + silver (fallback) dictionaries.
    pub fn load(gold: &[u8], silver: &[u8]) -> Self {
        let mut m = HashMap::new();
        for bytes in [silver, gold] {
            // silver first so gold overwrites it
            let raw: HashMap<String, serde_json::Value> =
                serde_json::from_slice(bytes).unwrap_or_default();
            for (k, v) in raw {
                if let Some(ph) = extract_default(v) {
                    m.insert(k.to_lowercase(), ph);
                }
            }
        }
        Lexicon(m)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn raw(&self, w: &str) -> Option<&str> {
        self.0.get(w).map(String::as_str)
    }

    /// Lexicon lookup with a suffix-morphology fallback for inflected forms.
    fn lookup(&self, word: &str) -> Option<String> {
        let w = word.to_lowercase();
        if let Some(p) = self.raw(&w) {
            return Some(p.to_string());
        }
        // morphology: stem + suffix phoneme (try stem, and stem+"e" for drop-e forms)
        let try_stem = |stem: &str| -> Option<&str> { self.raw(stem) };
        let voiceless = |p: &str| matches!(p.chars().last(), Some('p' | 't' | 'k' | 'f' | 'θ'));
        let sibilant =
            |p: &str| matches!(p.chars().last(), Some('s' | 'z' | 'ʃ' | 'ʒ' | 'ʧ' | 'ʤ'));
        if let Some(stem) = w.strip_suffix("ing").filter(|s| s.len() >= 2)
            && let Some(p) = try_stem(stem).or_else(|| try_stem(&format!("{stem}e")))
        {
            return Some(format!("{p}ɪŋ"));
        }
        if let Some(stem) = w.strip_suffix("ed").filter(|s| s.len() >= 2)
            && let Some(p) = try_stem(stem).or_else(|| try_stem(&format!("{stem}e")))
        {
            let suf = if matches!(p.chars().last(), Some('t' | 'd')) {
                "ɪd"
            } else if voiceless(p) {
                "t"
            } else {
                "d"
            };
            return Some(format!("{p}{suf}"));
        }
        if let Some(stem) = w.strip_suffix("es").filter(|s| s.len() >= 2)
            && let Some(p) = try_stem(stem)
        {
            return Some(format!("{p}ɪz"));
        }
        if let Some(stem) = w.strip_suffix('s').filter(|s| s.len() >= 2)
            && let Some(p) = try_stem(stem)
        {
            let suf = if sibilant(p) {
                "ɪz"
            } else if voiceless(p) {
                "s"
            } else {
                "z"
            };
            return Some(format!("{p}{suf}"));
        }
        None
    }
}

fn is_kept_punct(c: char) -> bool {
    matches!(
        c,
        ',' | '.' | '!' | '?' | ';' | ':' | '—' | '…' | '"' | '(' | ')' | '\''
    )
}

/// First phonetic segment is a vowel (skip leading stress marks).
fn starts_with_vowel(ph: &str) -> bool {
    let c = ph.chars().find(|c| !matches!(c, 'ˈ' | 'ˌ'));
    matches!(
        c,
        Some(
            'a' | 'e'
                | 'i'
                | 'o'
                | 'u'
                | 'æ'
                | 'ɑ'
                | 'ɐ'
                | 'ɒ'
                | 'ɔ'
                | 'ə'
                | 'ɛ'
                | 'ɜ'
                | 'ɪ'
                | 'ʊ'
                | 'ʌ'
                | 'ɚ'
                | 'ᵊ'
                | 'A'
                | 'E'
                | 'I'
                | 'O'
                | 'U'
                | 'W'
                | 'Y'
        )
    )
}

enum Tok {
    Word(String),
    Punct(char),
}

fn tokenize(text: &str) -> Vec<Tok> {
    let mut toks = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_alphabetic() || (c == '\'' && !word.is_empty()) {
            word.push(c);
        } else {
            if !word.is_empty() {
                toks.push(Tok::Word(std::mem::take(&mut word)));
            }
            if is_kept_punct(c) {
                toks.push(Tok::Punct(c));
            }
        }
    }
    if !word.is_empty() {
        toks.push(Tok::Word(word));
    }
    toks
}

/// Text → (phoneme string, OOV words). Words space-separated; punctuation attaches to
/// the preceding token. Applies the/to vowel-conditioning using the following word.
pub fn g2p(text: &str, lex: &Lexicon) -> (String, Vec<String>) {
    let toks = tokenize(text);
    // resolve each word token's phoneme (or None=OOV) up front for look-ahead.
    let resolved: Vec<Option<(String, Option<String>)>> = toks
        .iter()
        .map(|t| match t {
            Tok::Word(w) => Some((w.clone(), lex.lookup(w))),
            Tok::Punct(_) => None,
        })
        .collect();

    let mut out = String::new();
    let mut oov = Vec::new();
    let mut first = true;
    for (i, tok) in toks.iter().enumerate() {
        match tok {
            Tok::Word(w) => {
                let next_vowel = resolved[i + 1..]
                    .iter()
                    .flatten()
                    .find_map(|(_, p)| p.as_deref())
                    .map(starts_with_vowel)
                    .unwrap_or(false);
                let wl = w.to_lowercase();
                let ph = match wl.as_str() {
                    "the" => Some(if next_vowel {
                        "ði".to_string()
                    } else {
                        "ðə".to_string()
                    }),
                    "to" => Some(if next_vowel {
                        "tu".to_string()
                    } else {
                        "tə".to_string()
                    }),
                    _ => resolved[i].as_ref().and_then(|(_, p)| p.clone()),
                };
                match ph {
                    Some(ph) => {
                        if !first {
                            out.push(' ');
                        }
                        out.push_str(&ph);
                        first = false;
                    }
                    None => oov.push(w.clone()),
                }
            }
            Tok::Punct(c) => {
                out.push(*c);
                first = false;
            }
        }
    }
    (out, oov)
}
