/// DoRA (Weight-Decomposed Low-Rank Adaptation) layer.
///
/// Decomposes the weight matrix into direction and magnitude:
///   W = m * (W₀ + BA) / ‖W₀ + BA‖
///
/// Where:
/// - m is a learnable magnitude vector (per output neuron)
/// - W₀ is the frozen base weight
/// - BA is the standard LoRA update
///
/// DoRA consistently outperforms LoRA by learning magnitude separately
/// from direction, closer to how full fine-tuning works.
#[derive(Debug, Clone)]
pub struct DoraLayer {
    /// Input dimension.
    pub in_features: usize,
    /// Output dimension.
    pub out_features: usize,
    /// LoRA rank for the directional component.
    pub rank: usize,
    /// Alpha scaling factor.
    pub alpha: f32,
    /// Dropout rate.
    pub dropout: f32,
}

impl DoraLayer {
    /// Create a new DoRA layer with the given dimensions, rank, and alpha.
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        Self {
            in_features,
            out_features,
            rank,
            alpha,
            dropout: 0.0,
        }
    }

    /// Scaling factor for the directional LoRA component.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Trainable parameters: LoRA A + LoRA B + magnitude vector.
    pub fn trainable_params(&self) -> usize {
        let lora_params = self.rank * self.in_features + self.out_features * self.rank;
        let magnitude_params = self.out_features; // one scalar per output neuron
        lora_params + magnitude_params
    }

    /// Frozen base parameter count.
    pub fn frozen_params(&self) -> usize {
        self.in_features * self.out_features
    }

    /// Estimated bytes for adapter weights (FP16).
    pub fn adapter_bytes(&self) -> usize {
        self.trainable_params() * 2 // FP16
    }

    /// Estimated bytes for frozen base weights (FP16).
    pub fn frozen_base_bytes(&self) -> usize {
        self.frozen_params() * 2 // FP16
    }

    /// Total estimated VRAM bytes (frozen base + adapter).
    pub fn total_vram_bytes(&self) -> usize {
        self.frozen_base_bytes() + self.adapter_bytes()
    }

    /// VRAM savings compared to full fine-tuning.
    pub fn vram_savings_ratio(&self) -> f64 {
        let full_trainable = self.frozen_params() * 2; // All params in FP16
        let adapter_only = self.adapter_bytes();
        1.0 - (adapter_only as f64 / full_trainable as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dora_layer() {
        let layer = DoraLayer::new(4096, 4096, 16, 32.0);
        // LoRA params + magnitude vector
        let expected = 16 * 4096 + 4096 * 16 + 4096;
        assert_eq!(layer.trainable_params(), expected);
        assert_eq!(layer.scaling(), 2.0);
    }

    #[test]
    fn test_dora_vram_estimation() {
        let layer = DoraLayer::new(4096, 4096, 16, 32.0);
        assert!(layer.adapter_bytes() > 0);
        assert!(layer.frozen_base_bytes() > 0);
        assert!(layer.total_vram_bytes() > layer.adapter_bytes());
        assert!(layer.vram_savings_ratio() > 0.0);
        assert!(layer.vram_savings_ratio() < 1.0);
    }
}
