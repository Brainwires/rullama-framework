//! SafeTensors model weight loading.
//!
//! Loads pre-trained base weights from SafeTensors files for LoRA/DoRA/QLoRA
//! fine-tuning, enabling training from real model weights instead of random init.

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;
use tracing::{info, warn};

use crate::shared::error::TrainingError;
use crate::architectures::config::TransformerConfig;
use crate::quantization::{QuantConfig, dequantize_tensor, quantize_tensor};

/// Loader for SafeTensors model weight files.
pub struct SafeTensorsLoader {
    /// Raw file bytes (memory-mapped conceptually, loaded in full for simplicity).
    data: Vec<u8>,
}

impl SafeTensorsLoader {
    /// Open a SafeTensors file from disk.
    pub fn open(path: &Path) -> Result<Self, TrainingError> {
        let data = std::fs::read(path).map_err(|e| {
            TrainingError::Config(format!(
                "Failed to read SafeTensors file {}: {}",
                path.display(),
                e
            ))
        })?;

        // Validate that the file is parseable
        SafeTensors::deserialize(&data).map_err(|e| {
            TrainingError::Config(format!(
                "Invalid SafeTensors file {}: {}",
                path.display(),
                e
            ))
        })?;

        info!(
            "Opened SafeTensors file: {} ({} bytes)",
            path.display(),
            data.len()
        );
        Ok(Self { data })
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<String> {
        match SafeTensors::deserialize(&self.data) {
            Ok(st) => st.names().into_iter().map(|s| s.to_string()).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Load a single tensor as f32 data.
    ///
    /// Handles dtype conversion from f16/bf16/f32 to f32.
    /// Returns the flattened f32 values and the tensor shape.
    pub fn load_tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), TrainingError> {
        let st = SafeTensors::deserialize(&self.data)
            .map_err(|e| TrainingError::Backend(format!("Failed to parse SafeTensors: {}", e)))?;

        let view = st
            .tensor(name)
            .map_err(|e| TrainingError::Backend(format!("Tensor '{}' not found: {}", name, e)))?;

        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();

        let f32_data = match view.dtype() {
            safetensors::Dtype::F32 => data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            safetensors::Dtype::F16 => data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f16_to_f32(bits)
                })
                .collect(),
            safetensors::Dtype::BF16 => data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    bf16_to_f32(bits)
                })
                .collect(),
            other => {
                return Err(TrainingError::Backend(format!(
                    "Unsupported tensor dtype {:?} for '{}'",
                    other, name
                )));
            }
        };

        Ok((f32_data, shape))
    }

    /// Load a tensor, quantize it, then dequantize back to f32.
    ///
    /// This simulates QLoRA's approach: store base weights in quantized format
    /// for memory savings, then dequantize for the forward pass.
    pub fn load_tensor_quantized(
        &self,
        name: &str,
        quant_config: &QuantConfig,
    ) -> Result<(Vec<f32>, Vec<usize>), TrainingError> {
        let (raw_f32, shape) = self.load_tensor_f32(name)?;
        let (quantized, scales, zeros) = quantize_tensor(&raw_f32, quant_config);
        let dequantized = dequantize_tensor(&quantized, &scales, &zeros, quant_config.group_size);
        Ok((dequantized, shape))
    }

    /// Load a tensor that may already be pre-quantized (INT8/U8).
    ///
    /// If the tensor dtype is I8 or U8, returns the raw quantized bytes along with
    /// an associated scale tensor (looked up as `{name}_scale`).
    /// Returns `Ok(None)` if the tensor is not pre-quantized.
    #[allow(clippy::type_complexity)]
    pub fn load_tensor_prequantized(
        &self,
        name: &str,
    ) -> Result<Option<(Vec<u8>, Vec<f32>, Vec<usize>)>, TrainingError> {
        let st = SafeTensors::deserialize(&self.data)
            .map_err(|e| TrainingError::Backend(format!("Failed to parse SafeTensors: {}", e)))?;

        let view = st
            .tensor(name)
            .map_err(|e| TrainingError::Backend(format!("Tensor '{}' not found: {}", name, e)))?;

        match view.dtype() {
            safetensors::Dtype::I8 | safetensors::Dtype::U8 => {
                let shape = view.shape().to_vec();
                let quantized_bytes = view.data().to_vec();

                // Look for associated scale tensor
                let scale_name = format!("{}_scale", name);
                let scales = match st.tensor(&scale_name) {
                    Ok(scale_view) => scale_view
                        .data()
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                    Err(_) => {
                        warn!(
                            "No scale tensor found for pre-quantized tensor '{}', using default scale 1.0",
                            name
                        );
                        vec![1.0f32]
                    }
                };

                Ok(Some((quantized_bytes, scales, shape)))
            }
            _ => Ok(None),
        }
    }

    /// Try to extract model config from SafeTensors metadata.
    ///
    /// Looks for common metadata keys like `vocab_size`, `hidden_size`, etc.
    /// Returns `None` if metadata is missing or incomplete.
    pub fn load_config(&self) -> Option<TransformerConfig> {
        let (_, meta) = SafeTensors::read_metadata(&self.data).ok()?;
        let metadata = meta.metadata().as_ref()?;

        let get_usize = |key: &str| -> Option<usize> { metadata.get(key)?.parse().ok() };
        let get_f64 = |key: &str| -> Option<f64> { metadata.get(key)?.parse().ok() };

        Some(TransformerConfig {
            vocab_size: get_usize("vocab_size").unwrap_or(32000),
            hidden_size: get_usize("hidden_size")?,
            num_layers: get_usize("num_hidden_layers").or_else(|| get_usize("num_layers"))?,
            num_heads: get_usize("num_attention_heads").or_else(|| get_usize("num_heads"))?,
            num_kv_heads: get_usize("num_key_value_heads")
                .unwrap_or_else(|| get_usize("num_heads").unwrap_or(32)),
            intermediate_size: get_usize("intermediate_size").unwrap_or(5632),
            max_position_embeddings: get_usize("max_position_embeddings").unwrap_or(4096),
            rope_theta: get_f64("rope_theta").unwrap_or(10000.0),
            layer_norm_eps: get_f64("rms_norm_eps").unwrap_or(1e-5),
            use_swiglu: true,
            tie_word_embeddings: metadata
                .get("tie_word_embeddings")
                .is_some_and(|v| v == "true"),
        })
    }

    /// Find tensor names matching a pattern for a specific layer and projection.
    ///
    /// Common patterns:
    /// - `model.layers.{layer}.self_attn.{proj}.weight`
    /// - `model.layers.{layer}.mlp.{proj}.weight`
    pub fn find_layer_tensors(
        &self,
        layer_idx: usize,
        proj_names: &[&str],
    ) -> HashMap<String, String> {
        let names = self.tensor_names();
        let mut found = HashMap::new();

        for proj in proj_names {
            // Try common naming conventions
            let patterns = [
                format!("model.layers.{}.self_attn.{}.weight", layer_idx, proj),
                format!("model.layers.{}.mlp.{}.weight", layer_idx, proj),
                format!("layers.{}.attention.{}.weight", layer_idx, proj),
                format!("transformer.h.{}.attn.{}.weight", layer_idx, proj),
            ];

            for pattern in &patterns {
                if names.iter().any(|n| n == pattern) {
                    found.insert(proj.to_string(), pattern.clone());
                    break;
                }
            }
        }

        if found.is_empty() {
            warn!(
                "No matching tensors found for layer {} with projections {:?}",
                layer_idx, proj_names
            );
        }

        found
    }
}

/// Convert IEEE 754 half-precision (f16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut m = mantissa;
            let mut e = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let f32_exp = 127 - 15 - e;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13))
    } else {
        // Normal
        let f32_exp = exponent + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}

/// Convert bfloat16 bits to f32 (simply shift left by 16).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_f16_conversions() {
        // Zero
        assert_eq!(f16_to_f32(0), 0.0);
        // One (exponent=15, mantissa=0 → 2^(15-15) * 1.0 = 1.0)
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // Negative one
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_conversions() {
        // bf16 for 1.0 = 0x3F80
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
        // bf16 for -1.0 = 0xBF80
        assert!((bf16_to_f32(0xBF80) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_safetensors_roundtrip() {
        // Create a small SafeTensors file with known data
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        // Build SafeTensors data: a 2x3 f32 tensor
        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = HashMap::new();
        tensors.insert(
            "test_weight".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![2, 3],
                &tensor_bytes,
            )
            .unwrap(),
        );

        let serialized = safetensors::tensor::serialize(&tensors, None).unwrap();
        std::fs::write(&path, &serialized).unwrap();

        // Load and verify
        let loader = SafeTensorsLoader::open(&path).unwrap();
        let names = loader.tensor_names();
        assert!(names.contains(&"test_weight".to_string()));

        let (data, shape) = loader.load_tensor_f32("test_weight").unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data.len(), 6);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_safetensors_quantized_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        let tensor_data: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![128], &tensor_bytes)
                .unwrap(),
        );

        let serialized = safetensors::tensor::serialize(&tensors, None).unwrap();
        std::fs::write(&path, &serialized).unwrap();

        let loader = SafeTensorsLoader::open(&path).unwrap();
        let quant_config = QuantConfig::int8();
        let (dequantized, shape) = loader
            .load_tensor_quantized("weight", &quant_config)
            .unwrap();
        assert_eq!(shape, vec![128]);
        assert_eq!(dequantized.len(), 128);

        // Values should be approximately preserved
        for (orig, deq) in tensor_data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.02,
                "Quantization roundtrip too lossy: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_tensor_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.safetensors");

        let tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
        let serialized = safetensors::tensor::serialize(&tensors, None).unwrap();
        std::fs::write(&path, &serialized).unwrap();

        let loader = SafeTensorsLoader::open(&path).unwrap();
        assert!(loader.load_tensor_f32("nonexistent").is_err());
    }
}
