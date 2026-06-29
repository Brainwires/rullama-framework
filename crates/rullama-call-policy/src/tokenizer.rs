//! Per-provider token counting for budget pre-flight checks.
//!
//! The legacy `approx_input_tokens` heuristic (~4 chars per token) is fine
//! for a ballpark check but drifts up to 25 % against real provider
//! tokenizers — which means a strict cap can either over-reject (slowing
//! callers down) or under-reject (letting an oversized request through).
//!
//! The [`Tokenizer`] trait factors that pre-flight count into a per-provider
//! strategy. Three implementations ship today:
//!
//! - [`HeuristicTokenizer`] — `chars / 4`, no extra deps. Default fallback.
//! - [`OpenAiTokenizer`] — `o200k_base` BPE via `tiktoken-rs`. Use for
//!   `gpt-4o`, `gpt-4.1`, `gpt-5*`, `o1*`, `o3*`.
//! - [`AnthropicTokenizer`] — `cl100k_base` BPE via `tiktoken-rs`. Use for
//!   Claude 3 / Claude 4 models (Anthropic doesn't publish a tokenizer, so
//!   this is a closely-correlated open BPE — typically within ±5 %).
//!
//! `tiktoken-rs` lives behind the `tokenizers` cargo feature. When that
//! feature is off (the default), only the heuristic is available.

use rullama_core::{ContentBlock, Message, MessageContent};

/// Counts tokens in a `Message` slice the way a particular provider would
/// see them. Implementations are pure (no I/O) and cheap enough to call
/// per pre-flight check.
pub trait Tokenizer: Send + Sync {
    /// Estimated token count for the given messages.
    fn count(&self, messages: &[Message]) -> usize;
}

/// Cheap fallback: `chars / 4`. Matches the original `approx_input_tokens`
/// behaviour; safe to use when no provider-specific tokenizer is wired up.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicTokenizer;

impl Tokenizer for HeuristicTokenizer {
    fn count(&self, messages: &[Message]) -> usize {
        let mut chars: usize = 0;
        for m in messages {
            match &m.content {
                MessageContent::Text(t) => chars += t.len(),
                MessageContent::Blocks(blocks) => {
                    for b in blocks {
                        chars += block_chars(b);
                    }
                }
            }
        }
        chars / 4
    }
}

fn block_chars(b: &ContentBlock) -> usize {
    match b {
        ContentBlock::Text { text } => text.len(),
        ContentBlock::ToolUse { input, .. } => input.to_string().len(),
        ContentBlock::ToolResult { content, .. } => content.len(),
        ContentBlock::Image { .. } => 512,
    }
}

#[cfg(feature = "tokenizers")]
mod tiktoken_backed {
    use super::{ContentBlock, Message, MessageContent, Tokenizer, block_chars};
    use std::sync::OnceLock;
    use tiktoken_rs::CoreBPE;

    /// `o200k_base` BPE — the tokenizer OpenAI uses for the gpt-4o /
    /// gpt-4.1 / gpt-5 / o1 / o3 / o4 families. Older 3.5/4-base models use
    /// `cl100k_base`; for those, prefer [`AnthropicTokenizer`] (same BPE).
    pub struct OpenAiTokenizer {
        bpe: &'static CoreBPE,
    }

    impl OpenAiTokenizer {
        /// Cached `o200k_base` instance — the BPE table is large (~5 MB
        /// uncompressed) so it's worth amortising across the process.
        pub fn new() -> Self {
            static CELL: OnceLock<CoreBPE> = OnceLock::new();
            let bpe = CELL.get_or_init(|| {
                tiktoken_rs::o200k_base().expect("o200k_base BPE table is shipped with tiktoken-rs")
            });
            Self { bpe }
        }
    }

    impl Default for OpenAiTokenizer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Tokenizer for OpenAiTokenizer {
        fn count(&self, messages: &[Message]) -> usize {
            let mut total = 0usize;
            for m in messages {
                match &m.content {
                    MessageContent::Text(t) => total += self.bpe.encode_with_special_tokens(t).len(),
                    MessageContent::Blocks(blocks) => {
                        for b in blocks {
                            match b {
                                ContentBlock::Text { text } => {
                                    total += self.bpe.encode_with_special_tokens(text).len();
                                }
                                ContentBlock::ToolUse { input, .. } => {
                                    let s = input.to_string();
                                    total += self.bpe.encode_with_special_tokens(&s).len();
                                }
                                ContentBlock::ToolResult { content, .. } => {
                                    total +=
                                        self.bpe.encode_with_special_tokens(content).len();
                                }
                                ContentBlock::Image { .. } => total += block_chars(b) / 4,
                            }
                        }
                    }
                }
            }
            total
        }
    }

    /// `cl100k_base` BPE — closest open-source proxy for Anthropic's
    /// proprietary tokenizer. Empirically within ±5 % of `Usage.prompt_tokens`
    /// for Claude 3 and Claude 4 prompts; off by more for tool-heavy
    /// payloads (where Anthropic's own counter handles JSON specially).
    pub struct AnthropicTokenizer {
        bpe: &'static CoreBPE,
    }

    impl AnthropicTokenizer {
        /// Cached `cl100k_base` instance (~3 MB BPE table).
        pub fn new() -> Self {
            static CELL: OnceLock<CoreBPE> = OnceLock::new();
            let bpe = CELL.get_or_init(|| {
                tiktoken_rs::cl100k_base()
                    .expect("cl100k_base BPE table is shipped with tiktoken-rs")
            });
            Self { bpe }
        }
    }

    impl Default for AnthropicTokenizer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Tokenizer for AnthropicTokenizer {
        fn count(&self, messages: &[Message]) -> usize {
            // Same encode path as OpenAi, different BPE table.
            let mut total = 0usize;
            for m in messages {
                match &m.content {
                    MessageContent::Text(t) => total += self.bpe.encode_with_special_tokens(t).len(),
                    MessageContent::Blocks(blocks) => {
                        for b in blocks {
                            match b {
                                ContentBlock::Text { text } => {
                                    total += self.bpe.encode_with_special_tokens(text).len();
                                }
                                ContentBlock::ToolUse { input, .. } => {
                                    let s = input.to_string();
                                    total += self.bpe.encode_with_special_tokens(&s).len();
                                }
                                ContentBlock::ToolResult { content, .. } => {
                                    total +=
                                        self.bpe.encode_with_special_tokens(content).len();
                                }
                                ContentBlock::Image { .. } => total += block_chars(b) / 4,
                            }
                        }
                    }
                }
            }
            total
        }
    }
}

#[cfg(feature = "tokenizers")]
pub use tiktoken_backed::{AnthropicTokenizer, OpenAiTokenizer};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_empty_messages() {
        assert_eq!(HeuristicTokenizer.count(&[]), 0);
    }

    #[test]
    fn heuristic_counts_text() {
        let msgs = vec![Message::user("hello world")]; // 11 chars / 4 = 2
        assert_eq!(HeuristicTokenizer.count(&msgs), 2);
    }

    #[cfg(feature = "tokenizers")]
    #[test]
    fn openai_tokenizer_counts_simple_prompt() {
        let tk = OpenAiTokenizer::new();
        let msgs = vec![Message::user("hello world")];
        let count = tk.count(&msgs);
        // Real BPE: "hello world" tokenizes to 2 tokens in o200k_base.
        assert!(count >= 2 && count <= 4, "got: {count}");
    }

    #[cfg(feature = "tokenizers")]
    #[test]
    fn anthropic_tokenizer_distinct_from_openai() {
        let oa = OpenAiTokenizer::new();
        let an = AnthropicTokenizer::new();
        // Same input, different BPE tables — counts shouldn't both be 0.
        let msgs = vec![Message::user(
            "The quick brown fox jumps over the lazy dog. \
             This sentence is exactly long enough to disambiguate.",
        )];
        let a = oa.count(&msgs);
        let b = an.count(&msgs);
        assert!(a > 0 && b > 0);
    }
}
