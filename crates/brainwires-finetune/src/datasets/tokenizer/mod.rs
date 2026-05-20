use super::error::DatasetResult;

/// Abstraction over tokenizers for token counting and encoding.
pub trait Tokenizer: Send + Sync {
    /// Encode text into a sequence of token IDs.
    fn encode(&self, text: &str) -> DatasetResult<Vec<u32>>;
    /// Decode a sequence of token IDs back into text.
    fn decode(&self, ids: &[u32]) -> DatasetResult<String>;
    /// Return the vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Count tokens in a text string.
    fn count_tokens(&self, text: &str) -> DatasetResult<usize> {
        Ok(self.encode(text)?.len())
    }

    /// Encode a batch of texts into token ID sequences.
    fn encode_batch(&self, texts: &[&str]) -> DatasetResult<Vec<Vec<u32>>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Decode a batch of token ID sequences back into text.
    fn decode_batch(&self, ids_batch: &[&[u32]]) -> DatasetResult<Vec<String>> {
        ids_batch.iter().map(|ids| self.decode(ids)).collect()
    }

    /// Return special tokens and their IDs (if known).
    fn special_tokens(&self) -> Vec<(String, u32)> {
        Vec::new()
    }
}

/// HuggingFace tokenizer integration.
#[cfg(feature = "datasets-hf-tokenizer")]
pub mod hf;

/// Tiktoken tokenizer integration.
#[cfg(feature = "datasets-tiktoken")]
pub mod tiktoken;

#[cfg(feature = "datasets-hf-tokenizer")]
pub use hf::HfTokenizer;

#[cfg(feature = "datasets-tiktoken")]
pub use tiktoken::TiktokenTokenizer;
