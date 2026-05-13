use super::config::TransformerConfig;

/// A single Transformer block (structural definition).
///
/// Each block contains:
/// 1. Pre-norm (RMSNorm)
/// 2. Multi-head attention (with RoPE positional encoding)
/// 3. Residual connection
/// 4. Post-norm (RMSNorm)
/// 5. Feed-forward network (SwiGLU)
/// 6. Residual connection
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Index of this layer within the model.
    pub layer_index: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (for grouped-query attention).
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Whether to use SwiGLU activation in the FFN.
    pub use_swiglu: bool,
}

impl TransformerBlock {
    /// Create a new transformer block from a model config and layer index.
    pub fn new(config: &TransformerConfig, layer_index: usize) -> Self {
        Self {
            layer_index,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.hidden_size / config.num_heads,
            intermediate_size: config.intermediate_size,
            use_swiglu: config.use_swiglu,
        }
    }

    /// Parameters in the attention sub-layer.
    pub fn attention_params(&self) -> usize {
        let q_proj = self.hidden_size * self.hidden_size;
        let k_proj = self.hidden_size * (self.num_kv_heads * self.head_dim);
        let v_proj = self.hidden_size * (self.num_kv_heads * self.head_dim);
        let o_proj = self.hidden_size * self.hidden_size;
        q_proj + k_proj + v_proj + o_proj
    }

    /// Parameters in the feed-forward sub-layer.
    pub fn ffn_params(&self) -> usize {
        if self.use_swiglu {
            // gate_proj + up_proj + down_proj
            self.hidden_size * self.intermediate_size * 2
                + self.intermediate_size * self.hidden_size
        } else {
            // up_proj + down_proj
            self.hidden_size * self.intermediate_size + self.intermediate_size * self.hidden_size
        }
    }

    /// Total parameters in this block.
    pub fn total_params(&self) -> usize {
        self.attention_params() + self.ffn_params() + self.hidden_size * 2 // norms
    }
}

/// Create all transformer blocks for a model.
pub fn create_blocks(config: &TransformerConfig) -> Vec<TransformerBlock> {
    (0..config.num_layers)
        .map(|i| TransformerBlock::new(config, i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::SmallLmConfig;

    #[test]
    fn test_create_blocks() {
        let config = SmallLmConfig::tiny();
        let blocks = create_blocks(&config);
        assert_eq!(blocks.len(), 12);
        assert_eq!(blocks[0].layer_index, 0);
        assert_eq!(blocks[11].layer_index, 11);
    }

    #[test]
    fn test_block_params() {
        let config = SmallLmConfig::tiny();
        let block = TransformerBlock::new(&config, 0);
        assert!(block.total_params() > 0);
        assert!(block.attention_params() > 0);
        assert!(block.ffn_params() > 0);
    }
}
