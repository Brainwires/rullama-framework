/// LoRA (Low-Rank Adaptation) layer.
///
/// Decomposes weight update ΔW = BA where:
/// - B is (d × r) — projects down to rank r
/// - A is (r × d) — projects back up
/// - Scaling factor: α / r
///
/// This is a structural definition. The actual Burn tensor implementation
/// will be wired when the full training loop is built.
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// Input dimension.
    pub in_features: usize,
    /// Output dimension.
    pub out_features: usize,
    /// LoRA rank.
    pub rank: usize,
    /// Alpha scaling factor.
    pub alpha: f32,
    /// Dropout rate.
    pub dropout: f32,
    /// Whether this layer is active (can be frozen).
    pub active: bool,
}

impl LoraLayer {
    /// Create a new LoRA layer with the given dimensions, rank, and alpha.
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        Self {
            in_features,
            out_features,
            rank,
            alpha,
            dropout: 0.0,
            active: true,
        }
    }

    /// Set the dropout rate for the LoRA adapter.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Scaling factor applied to the low-rank update.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Number of trainable parameters in this LoRA layer.
    pub fn trainable_params(&self) -> usize {
        // A: (rank × in_features) + B: (out_features × rank)
        self.rank * self.in_features + self.out_features * self.rank
    }

    /// Frozen base parameter count.
    pub fn frozen_params(&self) -> usize {
        self.in_features * self.out_features
    }

    /// Compression ratio (trainable / total).
    pub fn compression_ratio(&self) -> f64 {
        self.trainable_params() as f64 / (self.frozen_params() + self.trainable_params()) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_layer() {
        let layer = LoraLayer::new(4096, 4096, 16, 32.0);
        assert_eq!(layer.scaling(), 2.0);
        assert_eq!(layer.trainable_params(), 16 * 4096 + 4096 * 16);
        assert!(layer.compression_ratio() < 0.01); // LoRA is highly efficient
    }
}
