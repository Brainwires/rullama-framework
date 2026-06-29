use super::super::error::{DatasetError, DatasetResult};
use super::Tokenizer;

/// OpenAI tiktoken tokenizer wrapper.
pub struct TiktokenTokenizer {
    bpe: tiktoken_rs::CoreBPE,
    vocab_size: usize,
}

impl TiktokenTokenizer {
    /// Create a tiktoken tokenizer for a specific model (e.g., "gpt-4", "gpt-3.5-turbo").
    pub fn for_model(model: &str) -> DatasetResult<Self> {
        let bpe = tiktoken_rs::get_bpe_from_model(model).map_err(|e| DatasetError::Tokenizer {
            message: format!("Failed to load tiktoken for model '{}': {}", model, e),
        })?;
        // tiktoken-rs doesn't expose vocab_size directly; use known values
        let vocab_size = match model {
            m if m.starts_with("gpt-4") => 100277,
            m if m.starts_with("gpt-3.5") => 100277,
            _ => 100277, // cl100k_base default
        };
        Ok(Self { bpe, vocab_size })
    }

    /// Create with cl100k_base encoding (GPT-4 / GPT-3.5-turbo).
    pub fn cl100k_base() -> DatasetResult<Self> {
        let bpe = tiktoken_rs::cl100k_base().map_err(|e| DatasetError::Tokenizer {
            message: format!("Failed to load cl100k_base: {}", e),
        })?;
        Ok(Self {
            bpe,
            vocab_size: 100277,
        })
    }

    /// Create with o200k_base encoding (GPT-4o).
    pub fn o200k_base() -> DatasetResult<Self> {
        let bpe = tiktoken_rs::o200k_base().map_err(|e| DatasetError::Tokenizer {
            message: format!("Failed to load o200k_base: {}", e),
        })?;
        Ok(Self {
            bpe,
            vocab_size: 200019,
        })
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str) -> DatasetResult<Vec<u32>> {
        let tokens = self.bpe.encode_with_special_tokens(text);
        Ok(tokens.into_iter().collect())
    }

    fn decode(&self, ids: &[u32]) -> DatasetResult<String> {
        self.bpe
            .decode(ids.to_vec())
            .map_err(|e| DatasetError::Tokenizer {
                message: format!("Decoding error: {}", e),
            })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> Vec<(String, u32)> {
        // tiktoken-rs doesn't expose special tokens directly, provide known ones
        let known = vec![
            ("<|endoftext|>", 100257u32),
            ("<|fim_prefix|>", 100258),
            ("<|fim_middle|>", 100259),
            ("<|fim_suffix|>", 100260),
        ];
        known
            .into_iter()
            .filter(|&(_, id)| (id as usize) < self.vocab_size)
            .map(|(name, id)| (name.to_string(), id))
            .collect()
    }
}
