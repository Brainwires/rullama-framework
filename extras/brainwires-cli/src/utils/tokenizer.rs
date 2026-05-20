//! Token counting utilities
//!
//! Provides token count estimation for different model families.
//! Uses character-based heuristics calibrated to match actual tokenizer behavior.
//!
//! # Token Estimation Accuracy
//!
//! The heuristics are tuned based on empirical testing:
//! - GPT-4/Claude: ~4.0 chars per token for English prose
//! - Code: ~3.5 chars per token (more symbols, shorter tokens)
//! - CJK text: ~1.5-2.0 chars per token (each character often = 1-2 tokens)
//!
//! For production accuracy, consider integrating tiktoken-rs (OpenAI models)
//! or the Anthropic tokenizer when available.

use crate::types::message::{ContentBlock, Message, MessageContent};

/// Token count estimate with metadata
#[derive(Debug, Clone)]
pub struct TokenEstimate {
    /// Estimated token count
    pub tokens: usize,
    /// Confidence level (0.0-1.0)
    pub confidence: f32,
    /// Model family the estimate is calibrated for
    pub model_family: ModelFamily,
}

/// Model family for token counting calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelFamily {
    /// OpenAI models (GPT-3.5, GPT-4, etc.) - uses cl100k_base tokenizer
    #[default]
    OpenAI,
    /// Anthropic models (Claude) - similar to OpenAI
    Anthropic,
    /// Google models (Gemini) - slightly different tokenization
    Google,
    /// Local LLM models - varies widely
    Local,
}

/// Configuration for token counting
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Characters per token for prose (default: 4.0)
    pub chars_per_token_prose: f32,
    /// Characters per token for code (default: 3.5)
    pub chars_per_token_code: f32,
    /// Base overhead per message (role, separators)
    pub message_overhead: usize,
    /// Token estimate for an image (varies by resolution)
    pub image_base_tokens: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            chars_per_token_prose: 4.0,
            chars_per_token_code: 3.5,
            message_overhead: 4,
            image_base_tokens: 85, // OpenAI's base for low-res images
        }
    }
}

impl TokenizerConfig {
    /// Create config calibrated for OpenAI models
    pub fn openai() -> Self {
        Self::default()
    }

    /// Create config calibrated for Anthropic models
    pub fn anthropic() -> Self {
        Self {
            chars_per_token_prose: 3.8, // Claude slightly more efficient
            chars_per_token_code: 3.3,
            message_overhead: 3,
            image_base_tokens: 68, // Anthropic's base
        }
    }

    /// Create config calibrated for Google models
    pub fn google() -> Self {
        Self {
            chars_per_token_prose: 4.2,
            chars_per_token_code: 3.8,
            message_overhead: 4,
            image_base_tokens: 258, // Gemini's base
        }
    }

    /// Create config for a model family
    pub fn for_family(family: ModelFamily) -> Self {
        match family {
            ModelFamily::OpenAI => Self::openai(),
            ModelFamily::Anthropic => Self::anthropic(),
            ModelFamily::Google => Self::google(),
            ModelFamily::Local => Self::default(),
        }
    }
}

/// Estimate token count for a text string
pub fn estimate_tokens(text: &str) -> usize {
    estimate_tokens_with_config(text, &TokenizerConfig::default())
}

/// Estimate token count with custom configuration
pub fn estimate_tokens_with_config(text: &str, config: &TokenizerConfig) -> usize {
    if text.is_empty() {
        return 0;
    }

    // Detect content type
    let is_code = is_likely_code(text);
    let chars_per_token = if is_code {
        config.chars_per_token_code
    } else {
        config.chars_per_token_prose
    };

    // Count CJK characters (they typically use more tokens per character)
    let cjk_chars = count_cjk_chars(text);
    let non_cjk_chars = text.chars().count() - cjk_chars;

    // CJK characters average ~1.5 tokens each
    let cjk_tokens = (cjk_chars as f32 * 1.5) as usize;
    let non_cjk_tokens = (non_cjk_chars as f32 / chars_per_token) as usize;

    cjk_tokens + non_cjk_tokens + 1 // +1 to avoid returning 0 for tiny strings
}

/// Estimate token count for a message
pub fn estimate_message_tokens(message: &Message) -> usize {
    estimate_message_tokens_with_config(message, &TokenizerConfig::default())
}

/// Estimate token count for a message with custom configuration
pub fn estimate_message_tokens_with_config(message: &Message, config: &TokenizerConfig) -> usize {
    let content_tokens = match &message.content {
        MessageContent::Text(text) => estimate_tokens_with_config(text, config),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .map(|block| estimate_block_tokens_with_config(block, config))
            .sum(),
    };

    content_tokens + config.message_overhead
}

/// Estimate tokens for a content block
fn estimate_block_tokens_with_config(block: &ContentBlock, config: &TokenizerConfig) -> usize {
    match block {
        ContentBlock::Text { text } => estimate_tokens_with_config(text, config),
        ContentBlock::Image { .. } => config.image_base_tokens,
        ContentBlock::ToolUse { name, input, .. } => {
            let name_tokens = estimate_tokens_with_config(name, config);
            let input_tokens = estimate_tokens_with_config(
                &serde_json::to_string(input).unwrap_or_default(),
                config,
            );
            name_tokens + input_tokens + 10 // Overhead for tool structure
        }
        ContentBlock::ToolResult { content, .. } => {
            estimate_tokens_with_config(content, config) + 5 // Overhead for result structure
        }
    }
}

/// Check if text is likely code (simple heuristic)
fn is_likely_code(text: &str) -> bool {
    // Code indicators
    let code_patterns = [
        "fn ",
        "def ",
        "function ",
        "class ",
        "impl ",
        "pub ",
        "async ",
        "const ",
        "let ",
        "var ",
        "import ",
        "export ",
        "return ",
        "if (",
        "if(",
        "for (",
        "for(",
        "while ",
        "match ",
        "=>",
        "->",
        "::",
        "{}",
    ];

    let code_chars = ['{', '}', '(', ')', '[', ']', ';', ':'];

    // Check for common code patterns
    let has_pattern = code_patterns.iter().any(|p| text.contains(p));

    // Count special characters
    let special_char_ratio =
        text.chars().filter(|c| code_chars.contains(c)).count() as f32 / text.len().max(1) as f32;

    has_pattern || special_char_ratio > 0.05
}

/// Count CJK (Chinese, Japanese, Korean) characters
fn count_cjk_chars(text: &str) -> usize {
    text.chars()
        .filter(|c| {
            let c = *c;
            // CJK Unified Ideographs and common ranges
            ('\u{4E00}'..='\u{9FFF}').contains(&c)  // CJK Unified Ideographs
                || ('\u{3040}'..='\u{309F}').contains(&c)  // Hiragana
                || ('\u{30A0}'..='\u{30FF}').contains(&c)  // Katakana
                || ('\u{AC00}'..='\u{D7A3}').contains(&c)  // Hangul Syllables
                || ('\u{3400}'..='\u{4DBF}').contains(&c) // CJK Extension A
        })
        .count()
}

/// Estimate total tokens for a conversation
pub fn estimate_conversation_tokens(messages: &[Message]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Determine model family from model ID
pub fn model_family_from_id(model_id: &str) -> ModelFamily {
    let model_lower = model_id.to_lowercase();

    if model_lower.contains("gpt") || model_lower.contains("o1") || model_lower.contains("davinci")
    {
        ModelFamily::OpenAI
    } else if model_lower.contains("claude") || model_lower.contains("anthropic") {
        ModelFamily::Anthropic
    } else if model_lower.contains("gemini") || model_lower.contains("palm") {
        ModelFamily::Google
    } else if model_lower.contains("llama")
        || model_lower.contains("mistral")
        || model_lower.contains("qwen")
    {
        ModelFamily::Local
    } else {
        ModelFamily::OpenAI // Default to OpenAI-style
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::Role;

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short() {
        let tokens = estimate_tokens("Hello, world!");
        assert!((2..=5).contains(&tokens));
    }

    #[test]
    fn test_estimate_tokens_prose() {
        // ~100 characters of prose
        let text =
            "The quick brown fox jumps over the lazy dog. This is a sample sentence for testing.";
        let tokens = estimate_tokens(text);
        // Should be roughly 20-25 tokens
        assert!((15..=30).contains(&tokens));
    }

    #[test]
    fn test_estimate_tokens_code() {
        let code = "fn main() { println!(\"Hello, world!\"); }";
        let tokens = estimate_tokens(code);
        // Code should have slightly higher token density
        assert!((8..=20).contains(&tokens));
    }

    #[test]
    fn test_estimate_tokens_cjk() {
        let cjk = "你好世界"; // "Hello world" in Chinese
        let tokens = estimate_tokens(cjk);
        // 4 CJK chars should be ~6 tokens
        assert!((4..=10).contains(&tokens));
    }

    #[test]
    fn test_estimate_message_tokens() {
        let message = Message {
            role: Role::User,
            content: MessageContent::Text("Hello, how are you?".to_string()),
            name: None,
            metadata: None,
        };
        let tokens = estimate_message_tokens(&message);
        // ~5 content tokens + 4 overhead
        assert!((6..=15).contains(&tokens));
    }

    #[test]
    fn test_model_family_detection() {
        assert_eq!(model_family_from_id("gpt-4-turbo"), ModelFamily::OpenAI);
        assert_eq!(
            model_family_from_id("claude-3-opus"),
            ModelFamily::Anthropic
        );
        assert_eq!(model_family_from_id("gemini-pro"), ModelFamily::Google);
        assert_eq!(model_family_from_id("llama-3-70b"), ModelFamily::Local);
        assert_eq!(model_family_from_id("unknown-model"), ModelFamily::OpenAI);
    }

    #[test]
    fn test_is_likely_code() {
        assert!(is_likely_code("fn main() {}"));
        assert!(is_likely_code("def hello(): pass"));
        assert!(is_likely_code("const x = { a: 1 };"));
        assert!(!is_likely_code("Hello, this is regular text."));
    }

    #[test]
    fn test_config_families() {
        let openai = TokenizerConfig::openai();
        let anthropic = TokenizerConfig::anthropic();
        let google = TokenizerConfig::google();

        // Anthropic is slightly more efficient
        assert!(anthropic.chars_per_token_prose < openai.chars_per_token_prose);

        // Google uses more tokens for images
        assert!(google.image_base_tokens > openai.image_base_tokens);
    }
}
