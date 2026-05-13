//! Tokenizer adapter for Gemma 4 GGUFs.
//!
//! M0 spike result (against `gemma4:e2b` blob `4e30e2665218…`):
//!   tokenizer.ggml.model = "llama"
//!   tokenizer.ggml.pre   = "gemma4"
//!   tokenizer.ggml.merges present  → BPE
//!   tokenizer.ggml.scores present  → legacy dual-store (we ignore on the BPE path)
//!
//! Decision: BPE via the `tokenizers` crate (HuggingFace), constructed from
//! `tokens` + `merges` + `token_type`, with a Gemma-4-specific pretokenizer regex
//! mirrored from llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GEMMA*` strategy. shimmytok
//! considered but doesn't ship the gemma4 pre-rules — same custom work either way,
//! and `tokenizers` is more battle-tested.

pub mod bpe;
pub mod special;

pub use bpe::BpeTokenizer;
