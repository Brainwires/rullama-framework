use serde::{Deserialize, Serialize};

/// Configuration for a Transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (for GQA; set equal to num_heads for MHA).
    pub num_kv_heads: usize,
    /// Intermediate dimension in FFN.
    pub intermediate_size: usize,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// RoPE theta for positional encoding.
    pub rope_theta: f64,
    /// Layer norm epsilon.
    pub layer_norm_eps: f64,
    /// Whether to use SwiGLU activation.
    pub use_swiglu: bool,
    /// Tie input/output embeddings.
    pub tie_word_embeddings: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_layers: 22,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 5632,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            use_swiglu: true,
            tie_word_embeddings: true,
        }
    }
}

impl TransformerConfig {
    /// Estimated parameter count.
    pub fn estimated_params(&self) -> u64 {
        let embed_params = self.vocab_size * self.hidden_size;
        let attention_params = self.num_layers
            * (self.hidden_size * self.hidden_size * 3 // Q, K, V projections
               + self.hidden_size * self.hidden_size); // output projection
        let ffn_params = self.num_layers
            * (self.hidden_size * self.intermediate_size * 2 // gate + up proj
               + self.intermediate_size * self.hidden_size); // down proj
        let norm_params = self.num_layers * self.hidden_size * 2; // pre/post layer norms

        let total = embed_params + attention_params + ffn_params + norm_params;
        if self.tie_word_embeddings {
            total as u64
        } else {
            (total + embed_params) as u64 // separate output head
        }
    }

    /// Human-readable parameter count.
    pub fn params_human(&self) -> String {
        let params = self.estimated_params();
        if params >= 1_000_000_000 {
            format!("{:.1}B", params as f64 / 1e9)
        } else if params >= 1_000_000 {
            format!("{:.1}M", params as f64 / 1e6)
        } else {
            format!("{:.1}K", params as f64 / 1e3)
        }
    }
}

/// Pre-configured small language model configs for from-scratch training.
pub struct SmallLmConfig;

impl SmallLmConfig {
    /// ~125M parameter model.
    pub fn tiny() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 32000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            intermediate_size: 2048,
            max_position_embeddings: 2048,
            ..Default::default()
        }
    }

    /// ~350M parameter model.
    pub fn small() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 32000,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 8,
            intermediate_size: 2816,
            max_position_embeddings: 4096,
            ..Default::default()
        }
    }

    /// ~1B parameter model.
    pub fn medium() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 32000,
            hidden_size: 2048,
            num_layers: 22,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 5632,
            max_position_embeddings: 4096,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_config() {
        let config = SmallLmConfig::tiny();
        let params = config.estimated_params();
        assert!(params > 100_000_000 && params < 200_000_000);
        assert!(config.params_human().contains('M'));
    }

    #[test]
    fn test_medium_config() {
        let config = SmallLmConfig::medium();
        let params = config.estimated_params();
        assert!(params > 500_000_000);
    }
}
