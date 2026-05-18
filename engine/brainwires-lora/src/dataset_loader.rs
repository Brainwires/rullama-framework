//! Dataset loading for local training.
//!
//! Parses JSONL training files into tokenized batches. Supports
//! instruction-tuning formats:
//!
//! - `{"prompt": "...", "completion": "..."}`
//! - `{"instruction": "...", "input": "...", "output": "..."}` (Alpaca)
//! - `{"messages": [{"role": "user", "content": "..."}, ...]}` (chat)
//!
//! Preference-pair (DPO / ORPO) datasets are out of scope after the
//! teardown — alignment is deferred until a real loss function exists.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use tracing::info;

use crate::shared::error::TrainingError;

/// A single training example (prompt + completion text).
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input text (prompt/instruction).
    pub prompt: String,
    /// Target text (completion/response).
    pub completion: String,
}

/// Parsed dataset ready for batching.
#[derive(Debug)]
pub struct TrainingDataset {
    /// All training examples.
    pub examples: Vec<TrainingExample>,
}

impl TrainingDataset {
    /// Load a JSONL dataset from disk. Native-only convenience wrapper
    /// around [`Self::load_jsonl_from_bytes`].
    ///
    /// Supports the same formats:
    /// 1. `{"prompt": "...", "completion": "..."}`
    /// 2. `{"messages": [{"role": "user", "content": "..."}, ...]}`
    /// 3. `{"instruction": "...", "output": "..."}` (Alpaca)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_jsonl(path: &Path) -> Result<Self, TrainingError> {
        let bytes = std::fs::read(path).map_err(|e| {
            TrainingError::Config(format!("Failed to open dataset: {}: {}", path.display(), e))
        })?;
        let ds = Self::load_jsonl_from_bytes(&bytes)?;
        info!(
            "Loaded {} training examples from {:?}",
            ds.examples.len(),
            path
        );
        Ok(ds)
    }

    /// Parse a JSONL byte buffer into a `TrainingDataset`. Cross-platform:
    /// the browser worker passes the file contents in as bytes, the
    /// native loader reads them off disk via [`Self::load_jsonl`].
    pub fn load_jsonl_from_bytes(bytes: &[u8]) -> Result<Self, TrainingError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| TrainingError::Config(format!("Dataset is not valid UTF-8: {e}")))?;
        let mut examples = Vec::new();
        for (line_num, raw) in text.lines().enumerate() {
            // Strip a leading UTF-8 BOM (only legal on the first line, but harmless
            // to attempt every line) before trim/JSON parsing.
            let line = raw.trim_start_matches('\u{feff}').trim();
            if line.is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
                TrainingError::Config(format!("Invalid JSON on line {}: {}", line_num + 1, e))
            })?;

            let example = if value.get("messages").is_some() {
                parse_chat_format(&value, line_num + 1)?
            } else if value.get("prompt").is_some() && value.get("completion").is_some() {
                parse_prompt_completion(&value, line_num + 1)?
            } else if value.get("instruction").is_some() {
                parse_alpaca_format(&value, line_num + 1)?
            } else {
                return Err(TrainingError::Config(format!(
                    "Line {}: expected 'prompt'+'completion', 'messages', or 'instruction'+'output' field",
                    line_num + 1,
                )));
            };

            examples.push(example);
        }

        if examples.is_empty() {
            return Err(TrainingError::Config(
                "Dataset is empty (no valid examples found)".to_string(),
            ));
        }
        Ok(Self { examples })
    }

    /// Number of examples in the dataset.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Calculate steps per epoch given a batch size.
    pub fn steps_per_epoch(&self, batch_size: usize) -> u64 {
        (self.examples.len() / batch_size.max(1)).max(1) as u64
    }

    /// Get a batch of examples by index range.
    pub fn get_batch(&self, start: usize, batch_size: usize) -> &[TrainingExample] {
        let end = (start + batch_size).min(self.examples.len());
        &self.examples[start..end]
    }
}

/// Parse `{"prompt": "...", "completion": "..."}` format.
fn parse_prompt_completion(
    value: &serde_json::Value,
    line_num: usize,
) -> Result<TrainingExample, TrainingError> {
    let prompt = value
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TrainingError::Config(format!("Line {}: 'prompt' must be a string", line_num))
        })?
        .to_string();

    let completion = value
        .get("completion")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TrainingError::Config(format!("Line {}: 'completion' must be a string", line_num))
        })?
        .to_string();

    Ok(TrainingExample { prompt, completion })
}

/// Parse `{"messages": [{"role": "...", "content": "..."}]}` chat format.
fn parse_chat_format(
    value: &serde_json::Value,
    line_num: usize,
) -> Result<TrainingExample, TrainingError> {
    let messages = value
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            TrainingError::Config(format!("Line {}: 'messages' must be an array", line_num))
        })?;

    let mut prompt_parts = Vec::new();
    let mut completion_parts = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");

        match role {
            "system" | "user" => prompt_parts.push(content.to_string()),
            // Multi-turn assistant traces: concatenate (previously overwrote,
            // silently dropping earlier turns).
            "assistant" => completion_parts.push(content.to_string()),
            _ => {}
        }
    }
    let completion = completion_parts.join("\n");

    if prompt_parts.is_empty() {
        return Err(TrainingError::Config(format!(
            "Line {}: no user/system messages found",
            line_num
        )));
    }
    if completion.is_empty() {
        return Err(TrainingError::Config(format!(
            "Line {}: no assistant message found",
            line_num
        )));
    }

    Ok(TrainingExample {
        prompt: prompt_parts.join("\n"),
        completion,
    })
}

/// Trait for tokenizers used in training.
///
/// Both `SimpleTokenizer` (byte-level fallback) and `ModelTokenizer` (BPE via
/// HuggingFace `tokenizers` crate) implement this trait.
pub trait Tokenizer {
    /// Encode text into token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Encode a training example into (input_ids, target_ids).
    ///
    /// Concatenates prompt + completion, with prompt tokens masked in targets
    /// (set to `u32::MAX`). Truncates to max sequence length.
    fn encode_example(&self, example: &TrainingExample) -> (Vec<u32>, Vec<u32>);

    /// Vocabulary size of this tokenizer.
    fn vocab_size(&self) -> usize;
}

/// Simple character-level tokenizer for training.
///
/// In production, this would be a BPE/SentencePiece tokenizer loaded from the model.
/// This basic implementation enables the training loop to work end-to-end.
pub struct SimpleTokenizer {
    max_seq_len: usize,
}

impl SimpleTokenizer {
    /// Create a tokenizer with the given maximum sequence length.
    pub fn new(max_seq_len: usize) -> Self {
        Self { max_seq_len }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes()
            .take(self.max_seq_len)
            .map(|b| b as u32)
            .collect()
    }

    fn encode_example(&self, example: &TrainingExample) -> (Vec<u32>, Vec<u32>) {
        let prompt_tokens = self.encode(&example.prompt);
        let completion_tokens = self.encode(&example.completion);
        let prompt_len = prompt_tokens.len();

        let mut input_ids = prompt_tokens;
        input_ids.extend_from_slice(&completion_tokens);
        input_ids.truncate(self.max_seq_len);

        next_token_targets(&input_ids, prompt_len)
    }

    fn vocab_size(&self) -> usize {
        256
    }
}

/// BPE tokenizer wrapping HuggingFace `tokenizers` crate. Native-only
/// because the `tokenizers` crate's transitive `getrandom@0.3` doesn't
/// build for `wasm32-unknown-unknown` without backend cfg. Browser
/// callers should pass pre-tokenised input IDs through the
/// wasm-bindgen surface — the rullama `Model.encodeTokens` already
/// covers the gemma4 BPE path on both targets.
///
/// Provides correct vocab-size alignment with real models (e.g., LLaMA, Mistral).
/// Load from a `tokenizer.json` file or a pretrained HuggingFace model ID.
#[cfg(not(target_arch = "wasm32"))]
pub struct ModelTokenizer {
    tokenizer: tokenizers::Tokenizer,
    max_seq_len: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl ModelTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file on disk.
    pub fn from_file(path: &Path) -> Result<Self, TrainingError> {
        let tokenizer = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            TrainingError::Config(format!(
                "Failed to load tokenizer from {}: {}",
                path.display(),
                e
            ))
        })?;
        Ok(Self {
            tokenizer,
            max_seq_len: 2048,
        })
    }

    /// Load a tokenizer from raw JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainingError> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| {
            TrainingError::Config(format!("Failed to load tokenizer from bytes: {}", e))
        })?;
        Ok(Self {
            tokenizer,
            max_seq_len: 2048,
        })
    }

    /// Set the maximum sequence length for encoding.
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Tokenizer for ModelTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => encoding
                .get_ids()
                .iter()
                .take(self.max_seq_len)
                .copied()
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    fn encode_example(&self, example: &TrainingExample) -> (Vec<u32>, Vec<u32>) {
        let prompt_tokens = self.encode(&example.prompt);
        let completion_tokens = self.encode(&example.completion);
        let prompt_len = prompt_tokens.len();

        let mut input_ids = prompt_tokens;
        input_ids.extend_from_slice(&completion_tokens);
        input_ids.truncate(self.max_seq_len);

        next_token_targets(&input_ids, prompt_len)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

/// Build next-token-prediction targets from a token stream.
///
/// At position `i`, the trainer expects the model to predict `input_ids[i+1]`.
/// Positions inside the prompt (up to but not including `prompt_len - 1`) are
/// masked with `u32::MAX` so the loss only fires on the completion. The last
/// position has no next token and is also masked.
fn next_token_targets(input_ids: &[u32], prompt_len: usize) -> (Vec<u32>, Vec<u32>) {
    let n = input_ids.len();
    let mut targets = vec![u32::MAX; n];
    if n < 2 {
        return (input_ids.to_vec(), targets);
    }
    // First trained position predicts the first completion token, i.e. the
    // model sees prompt_tokens[..prompt_len] and must emit
    // input_ids[prompt_len] = completion_tokens[0]. That prediction
    // happens at logits position `prompt_len - 1` (1-indexed: the last prompt
    // token's logits). For prompts shorter than 1 token, start at 0.
    let start = prompt_len.saturating_sub(1);
    targets[start..n - 1].copy_from_slice(&input_ids[start + 1..n]);
    (input_ids.to_vec(), targets)
}

/// Parse `{"instruction": "...", "input": "...", "output": "..."}` Alpaca format.
fn parse_alpaca_format(
    value: &serde_json::Value,
    line_num: usize,
) -> Result<TrainingExample, TrainingError> {
    let instruction = value
        .get("instruction")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TrainingError::Config(format!("Line {}: 'instruction' must be a string", line_num))
        })?;

    let input = value.get("input").and_then(|v| v.as_str()).unwrap_or("");

    let output = value
        .get("output")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TrainingError::Config(format!("Line {}: 'output' must be a string", line_num))
        })?;

    let prompt = if input.is_empty() {
        instruction.to_string()
    } else {
        format!("{}\n{}", instruction, input)
    };

    Ok(TrainingExample {
        prompt,
        completion: output.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_prompt_completion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"prompt": "Hello", "completion": "World"}}"#).unwrap();
        writeln!(f, r#"{{"prompt": "Foo", "completion": "Bar"}}"#).unwrap();

        let dataset = TrainingDataset::load_jsonl(&path).unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.examples[0].prompt, "Hello");
        assert_eq!(dataset.examples[0].completion, "World");
    }

    #[test]
    fn test_load_chat_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"messages": [{{"role": "user", "content": "Hi"}}, {{"role": "assistant", "content": "Hello!"}}]}}"#
        )
        .unwrap();

        let dataset = TrainingDataset::load_jsonl(&path).unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.examples[0].prompt, "Hi");
        assert_eq!(dataset.examples[0].completion, "Hello!");
    }

    #[test]
    fn test_load_alpaca_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}}"#
        )
        .unwrap();
        writeln!(f, r#"{{"instruction": "What is 2+2?", "output": "4"}}"#).unwrap();

        let dataset = TrainingDataset::load_jsonl(&path).unwrap();
        assert_eq!(dataset.len(), 2);
        assert!(dataset.examples[0].prompt.contains("Translate to French"));
        assert!(dataset.examples[0].prompt.contains("Hello"));
        assert_eq!(dataset.examples[0].completion, "Bonjour");
        assert_eq!(dataset.examples[1].prompt, "What is 2+2?");
    }

    #[test]
    fn test_empty_dataset_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::File::create(&path).unwrap();

        let result = TrainingDataset::load_jsonl(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_steps_per_epoch() {
        let dataset = TrainingDataset {
            examples: vec![
                TrainingExample {
                    prompt: "a".into(),
                    completion: "b".into(),
                };
                100
            ],
        };
        assert_eq!(dataset.steps_per_epoch(4), 25);
        assert_eq!(dataset.steps_per_epoch(10), 10);
    }

    #[test]
    fn test_simple_tokenizer() {
        let tok = SimpleTokenizer::new(512);
        let tokens = tok.encode("Hello");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], b'H' as u32);
    }

    #[test]
    fn test_encode_example() {
        let tok = SimpleTokenizer::new(512);
        let example = TrainingExample {
            prompt: "Hi".to_string(),
            completion: "Ok".to_string(),
        };
        let (input, target) = tok.encode_example(&example);
        assert_eq!(input.len(), 4); // "Hi" + "Ok"
        // Position 0 ('H') predicts 'i' — but that's still inside the prompt,
        // so it stays masked.
        assert_eq!(target[0], u32::MAX);
        // Position 1 ('i') predicts the first completion token 'O'.
        assert_eq!(target[1], b'O' as u32);
        // Position 2 ('O') predicts 'k'.
        assert_eq!(target[2], b'k' as u32);
        // Position 3 ('k') has no next token, masked.
        assert_eq!(target[3], u32::MAX);
    }

    #[test]
    fn test_tokenizer_trait_simple() {
        let tok: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(512));
        assert_eq!(tok.vocab_size(), 256);
        let tokens = tok.encode("Hello");
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_model_tokenizer_from_file() {
        // Create a minimal tokenizer.json for testing
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        // Minimal BPE tokenizer JSON (3-token vocab: a, b, c)
        let tokenizer_json = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "c": 2},
                "merges": []
            }
        }"#;
        std::fs::write(&path, tokenizer_json).unwrap();

        let tok = ModelTokenizer::from_file(&path).unwrap();
        assert!(tok.vocab_size() >= 3);
        let tokens = tok.encode("abc");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_model_tokenizer_encode_example() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        let tokenizer_json = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {"H": 0, "e": 1, "l": 2, "o": 3, "W": 4, "r": 5, "d": 6},
                "merges": []
            }
        }"#;
        std::fs::write(&path, tokenizer_json).unwrap();

        let tok = ModelTokenizer::from_file(&path).unwrap();
        let example = TrainingExample {
            prompt: "Hello".to_string(),
            completion: "World".to_string(),
        };
        let (input, target) = tok.encode_example(&example);
        assert!(!input.is_empty());
        // Prompt portion (everything before the last prompt token) stays masked.
        let prompt_len = tok.encode("Hello").len();
        for (i, tok_id) in target.iter().take(prompt_len.saturating_sub(1)).enumerate() {
            assert_eq!(*tok_id, u32::MAX, "Prompt token {i} should be masked");
        }
        // Last position has no next token; masked.
        assert_eq!(*target.last().unwrap(), u32::MAX);
    }

    #[test]
    fn test_bom_stripped_on_first_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // Write a UTF-8 BOM followed by valid JSON.
        f.write_all(b"\xef\xbb\xbf").unwrap();
        writeln!(f, r#"{{"prompt": "Hello", "completion": "World"}}"#).unwrap();
        let dataset = TrainingDataset::load_jsonl(&path).unwrap();
        assert_eq!(dataset.examples[0].prompt, "Hello");
    }

    #[test]
    fn test_chat_format_multi_turn_appends() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"messages": [{{"role": "user", "content": "Hi"}}, {{"role": "assistant", "content": "Hello!"}}, {{"role": "user", "content": "More"}}, {{"role": "assistant", "content": "Sure"}}]}}"#
        )
        .unwrap();
        let dataset = TrainingDataset::load_jsonl(&path).unwrap();
        // Earlier assistant turn must not be silently dropped.
        assert!(dataset.examples[0].completion.contains("Hello!"));
        assert!(dataset.examples[0].completion.contains("Sure"));
    }
}
