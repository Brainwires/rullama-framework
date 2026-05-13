/// QLoRA (Quantized Low-Rank Adaptation) layer.
///
/// Base weights are quantized to INT4/INT8, while LoRA adapter weights
/// remain in FP16 for full gradient precision. This drastically reduces
/// VRAM usage while maintaining training quality.
#[derive(Debug, Clone)]
pub struct QLoraLayer {
    /// Input dimension.
    pub in_features: usize,
    /// Output dimension.
    pub out_features: usize,
    /// LoRA rank.
    pub rank: usize,
    /// Alpha scaling factor.
    pub alpha: f32,
    /// Quantization bits for base weights (4 or 8).
    pub quant_bits: u8,
    /// Block size for quantization grouping.
    pub group_size: usize,
    /// Whether double quantization is enabled (quantize the quantization constants).
    pub double_quant: bool,
}

impl QLoraLayer {
    /// Create a new QLoRA layer with the given dimensions, rank, alpha, and quantization bits.
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32, bits: u8) -> Self {
        Self {
            in_features,
            out_features,
            rank,
            alpha,
            quant_bits: bits,
            group_size: 64,
            double_quant: true,
        }
    }

    /// 4-bit QLoRA (most common).
    pub fn int4(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        Self::new(in_features, out_features, rank, alpha, 4)
    }

    /// 8-bit QLoRA.
    pub fn int8(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        Self::new(in_features, out_features, rank, alpha, 8)
    }

    /// Scaling factor.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Estimated VRAM for quantized base weights (bytes).
    pub fn quantized_base_bytes(&self) -> usize {
        let total_elements = self.in_features * self.out_features;
        // bits per element, plus scale factors per group
        let base_bytes = (total_elements * self.quant_bits as usize) / 8;
        let num_groups = total_elements / self.group_size;
        let scale_bytes = num_groups * 2; // FP16 scale per group
        base_bytes + scale_bytes
    }

    /// Adapter parameters (FP16).
    pub fn adapter_bytes(&self) -> usize {
        let trainable_params = self.rank * self.in_features + self.out_features * self.rank;
        trainable_params * 2 // FP16
    }

    /// VRAM savings compared to full FP16.
    pub fn vram_savings_ratio(&self) -> f64 {
        let full_fp16 = self.in_features * self.out_features * 2; // FP16 base
        let quantized = self.quantized_base_bytes() + self.adapter_bytes();
        1.0 - (quantized as f64 / full_fp16 as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlora_int4() {
        let layer = QLoraLayer::int4(4096, 4096, 16, 32.0);
        assert_eq!(layer.quant_bits, 4);
        assert!(layer.vram_savings_ratio() > 0.5); // Should save >50% VRAM
    }

    #[test]
    fn test_qlora_int8() {
        let layer = QLoraLayer::int8(4096, 4096, 16, 32.0);
        assert_eq!(layer.quant_bits, 8);
    }
}
