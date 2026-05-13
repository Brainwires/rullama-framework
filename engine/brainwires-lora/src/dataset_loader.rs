//! Dataset loading for local training.
//!
//! Parses JSONL training files into tokenized batches for the Burn training loop.
//! Supports instruction-tuning formats: `{"prompt": ..., "completion": ...}` and
//! `{"messages": [...]}` (chat format).
//!
//! Also supports preference pair datasets for DPO/ORPO alignment:
//! `{"prompt": "...", "chosen": "...", "rejected": "..."}`.

use std::io::BufRead;
use std::path::Path;

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
    /// Load a JSONL dataset from disk.
    ///
    /// Supports two formats:
    /// 1. `{"prompt": "...", "completion": "..."}`
    /// 2. `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
    pub fn load_jsonl(path: &Path) -> Result<Self, TrainingError> {
        let file = std::fs::File::open(path).map_err(|e| {
            TrainingError::Config(format!("Failed to open dataset: {}: {}", path.display(), e))
        })?;
        let reader = std::io::BufReader::new(file);
        let mut examples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                TrainingError::Config(format!("Failed to read line {}: {}", line_num + 1, e))
            })?;
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
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

        info!(
            "Loaded {} training examples from {:?}",
            examples.len(),
            path
        );
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
    let mut completion = String::new();

    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");

        match role {
            "system" | "user" => prompt_parts.push(content.to_string()),
            "assistant" => completion = content.to_string(),
            _ => {}
        }
    }

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

        // Targets: shifted input_ids, with prompt portion masked
        let mut target_ids = vec![u32::MAX; input_ids.len()];
        target_ids[prompt_len..input_ids.len()].copy_from_slice(&input_ids[prompt_len..]);

        (input_ids, target_ids)
    }

    fn vocab_size(&self) -> usize {
        257
    }
}

/// BPE tokenizer wrapping HuggingFace `tokenizers` crate.
///
/// Provides correct vocab-size alignment with real models (e.g., LLaMA, Mistral).
/// Load from a `tokenizer.json` file or a pretrained HuggingFace model ID.
pub struct ModelTokenizer {
    tokenizer: tokenizers::Tokenizer,
    max_seq_len: usize,
}

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

        let mut target_ids = vec![u32::MAX; input_ids.len()];
        target_ids[prompt_len..input_ids.len()].copy_from_slice(&input_ids[prompt_len..]);

        (input_ids, target_ids)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
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

/// A single preference pair example for DPO/ORPO alignment training.
#[derive(Debug, Clone)]
pub struct PreferenceExample {
    /// Input prompt text.
    pub prompt: String,
    /// Preferred (chosen) completion.
    pub chosen: String,
    /// Dispreferred (rejected) completion.
    pub rejected: String,
}

/// Preference pair dataset for alignment training (DPO/ORPO).
#[derive(Debug)]
pub struct PreferenceDataset {
    /// All preference examples.
    pub examples: Vec<PreferenceExample>,
}

impl PreferenceDataset {
    /// Load preference pairs from JSONL.
    ///
    /// Each line: `{"prompt": "...", "chosen": "...", "rejected": "..."}`
    pub fn load_jsonl(path: &Path) -> Result<Self, TrainingError> {
        let file = std::fs::File::open(path).map_err(|e| {
            TrainingError::Config(format!(
                "Failed to open preference dataset: {}: {}",
                path.display(),
                e
            ))
        })?;
        let reader = std::io::BufReader::new(file);
        let mut examples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                TrainingError::Config(format!("Failed to read line {}: {}", line_num + 1, e))
            })?;
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
                TrainingError::Config(format!("Invalid JSON on line {}: {}", line_num + 1, e))
            })?;

            let prompt = value
                .get("prompt")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    TrainingError::Config(format!(
                        "Line {}: 'prompt' must be a string",
                        line_num + 1
                    ))
                })?
                .to_string();

            let chosen = value
                .get("chosen")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    TrainingError::Config(format!(
                        "Line {}: 'chosen' must be a string",
                        line_num + 1
                    ))
                })?
                .to_string();

            let rejected = value
                .get("rejected")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    TrainingError::Config(format!(
                        "Line {}: 'rejected' must be a string",
                        line_num + 1
                    ))
                })?
                .to_string();

            examples.push(PreferenceExample {
                prompt,
                chosen,
                rejected,
            });
        }

        if examples.is_empty() {
            return Err(TrainingError::Config(
                "Preference dataset is empty (no valid examples found)".to_string(),
            ));
        }

        info!(
            "Loaded {} preference examples from {:?}",
            examples.len(),
            path
        );
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
    pub fn get_batch(&self, start: usize, batch_size: usize) -> &[PreferenceExample] {
        let end = (start + batch_size).min(self.examples.len());
        &self.examples[start..end]
    }
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
        // First 2 tokens (prompt) should be masked
        assert_eq!(target[0], u32::MAX);
        assert_eq!(target[1], u32::MAX);
        // Completion tokens should have actual values
        assert_eq!(target[2], b'O' as u32);
        assert_eq!(target[3], b'k' as u32);
    }

    #[test]
    fn test_tokenizer_trait_simple() {
        let tok: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(512));
        assert_eq!(tok.vocab_size(), 257);
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
        // Prompt portion should be masked
        let prompt_len = tok.encode("Hello").len();
        for (i, tok_id) in target.iter().take(prompt_len).enumerate() {
            assert_eq!(*tok_id, u32::MAX, "Prompt token {i} should be masked");
        }
    }

    #[test]
    fn test_preference_dataset_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prefs.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"prompt": "Capital of France?", "chosen": "Paris", "rejected": "London"}}"#
        )
        .unwrap();

        let dataset = PreferenceDataset::load_jsonl(&path).unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.examples[0].prompt, "What is 2+2?");
        assert_eq!(dataset.examples[0].chosen, "4");
        assert_eq!(dataset.examples[0].rejected, "5");
    }

    #[test]
    fn test_preference_dataset_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::File::create(&path).unwrap();

        let result = PreferenceDataset::load_jsonl(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_preference_dataset_batching() {
        let dataset = PreferenceDataset {
            examples: vec![
                PreferenceExample {
                    prompt: "a".into(),
                    chosen: "b".into(),
                    rejected: "c".into(),
                };
                10
            ],
        };
        assert_eq!(dataset.steps_per_epoch(3), 3);
        let batch = dataset.get_batch(0, 3);
        assert_eq!(batch.len(), 3);
    }
}
