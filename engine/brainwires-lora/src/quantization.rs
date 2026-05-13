//! Quantization utilities for QLoRA and model compression.
//!
//! Supports INT4 and INT8 quantization with per-group scaling factors.

/// Quantization configuration.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Number of bits (4 or 8).
    pub bits: u8,
    /// Group size for quantization (elements per scale factor).
    pub group_size: usize,
    /// Enable double quantization (quantize the scale factors).
    pub double_quant: bool,
}

impl QuantConfig {
    /// Create an INT4 quantization configuration (4-bit, group size 64, double quant enabled).
    pub fn int4() -> Self {
        Self {
            bits: 4,
            group_size: 64,
            double_quant: true,
        }
    }

    /// Create an INT8 quantization configuration (8-bit, group size 128).
    pub fn int8() -> Self {
        Self {
            bits: 8,
            group_size: 128,
            double_quant: false,
        }
    }
}

/// Quantize a f32 tensor to INT4/INT8 with per-group scale factors.
///
/// Returns (quantized_data, scales, zero_points).
pub fn quantize_tensor(data: &[f32], config: &QuantConfig) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let num_groups = data.len().div_ceil(config.group_size);
    let mut quantized = Vec::new();
    let mut scales = Vec::with_capacity(num_groups);
    let mut zero_points = Vec::with_capacity(num_groups);

    let max_val = ((1u32 << config.bits) - 1) as f32;

    for group in data.chunks(config.group_size) {
        let min = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max - min;
        let scale = if range > 0.0 { range / max_val } else { 1.0 };
        let zero_point = min;

        scales.push(scale);
        zero_points.push(zero_point);

        for &val in group {
            let quantized_val = ((val - zero_point) / scale).round().clamp(0.0, max_val) as u8;
            quantized.push(quantized_val);
        }
    }

    (quantized, scales, zero_points)
}

/// Dequantize an INT4/INT8 tensor back to f32.
pub fn dequantize_tensor(
    quantized: &[u8],
    scales: &[f32],
    zero_points: &[f32],
    group_size: usize,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(quantized.len());

    for (group_idx, chunk) in quantized.chunks(group_size).enumerate() {
        let scale = scales[group_idx];
        let zero_point = zero_points[group_idx];

        for &q in chunk {
            result.push(q as f32 * scale + zero_point);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_int8() {
        let data: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let config = QuantConfig::int8();

        let (quantized, scales, zero_points) = quantize_tensor(&data, &config);
        let recovered = dequantize_tensor(&quantized, &scales, &zero_points, config.group_size);

        // Check that values are approximately preserved
        for (original, &recovered_val) in data.iter().zip(recovered.iter()) {
            assert!(
                (original - recovered_val).abs() < 0.02,
                "Original: {}, Recovered: {}",
                original,
                recovered_val
            );
        }
    }

    #[test]
    fn test_quantize_int4() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let config = QuantConfig::int4();

        let (quantized, scales, _) = quantize_tensor(&data, &config);
        // INT4 values should fit in 0..15
        for &q in &quantized {
            assert!(q <= 15, "INT4 value out of range: {}", q);
        }
        assert_eq!(scales.len(), 1); // One group
    }
}
