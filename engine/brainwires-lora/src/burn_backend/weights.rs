use std::time::Instant;

use burn_core::prelude::*;
use burn_wgpu::WgpuDevice;
use tracing::{info, warn};

use super::types::TrainBackend;
use crate::shared::error::TrainingError;
use crate::dataset_loader::{ModelTokenizer, SimpleTokenizer, Tokenizer};
use crate::quantization::QuantConfig;
use crate::weight_loader::SafeTensorsLoader;
use crate::{LocalTrainingConfig, TrainedModelArtifact};

/// Create the appropriate tokenizer based on config.
pub(super) fn create_tokenizer(
    config: &LocalTrainingConfig,
) -> Result<Box<dyn Tokenizer>, TrainingError> {
    if let Some(ref tok_path) = config.tokenizer_path {
        info!("Loading BPE tokenizer from {:?}", tok_path);
        let tok =
            ModelTokenizer::from_file(tok_path)?.with_max_seq_len(config.hyperparams.max_seq_len);
        info!("Tokenizer vocab size: {}", tok.vocab_size());
        Ok(Box::new(tok))
    } else {
        info!("Using byte-level fallback tokenizer (vocab=257)");
        Ok(Box::new(SimpleTokenizer::new(
            config.hyperparams.max_seq_len,
        )))
    }
}

/// Helper to write export metadata and create TrainedModelArtifact.
pub(super) fn finalize_training(
    config: &LocalTrainingConfig,
    running_loss: f32,
    total_steps: u64,
    start: &Instant,
    a_bytes: &[u8],
    b_bytes: &[u8],
    extra_bytes: Option<&[u8]>,
) -> Result<TrainedModelArtifact, TrainingError> {
    let output_path = config.output_dir.join("adapter_weights.bin");
    info!("Training complete. Saving adapter to {:?}", output_path);

    let mut buf = Vec::new();
    buf.extend_from_slice(&(a_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(a_bytes);
    buf.extend_from_slice(&(b_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(b_bytes);
    if let Some(extra) = extra_bytes {
        buf.extend_from_slice(&(extra.len() as u64).to_le_bytes());
        buf.extend_from_slice(extra);
    }

    std::fs::write(&output_path, &buf)
        .map_err(|e| TrainingError::Backend(format!("Failed to write adapter weights: {}", e)))?;
    info!("Wrote {} bytes of adapter weights", buf.len());

    let metadata = crate::export::ExportMetadata {
        format: "adapter_only".to_string(),
        base_model: config.model_path.to_string_lossy().to_string(),
        adapter_method: Some(format!("{:?}", config.lora.method)),
        training_epochs: config.hyperparams.epochs,
        final_loss: Some(running_loss as f64),
        exported_at: chrono::Utc::now(),
    };
    crate::export::write_export_metadata(&config.output_dir, &metadata)
        .map_err(TrainingError::Io)?;

    Ok(TrainedModelArtifact {
        model_path: output_path,
        format: "adapter_only".to_string(),
        base_model: config.model_path.to_string_lossy().to_string(),
        metrics: crate::shared::types::TrainingMetrics {
            final_train_loss: Some(running_loss as f64),
            final_eval_loss: None,
            total_steps,
            total_epochs: config.hyperparams.epochs,
            total_tokens_trained: Some(
                total_steps
                    * config.hyperparams.batch_size as u64
                    * config.hyperparams.max_seq_len as u64,
            ),
            duration_secs: start.elapsed().as_secs(),
            estimated_cost_usd: None,
        },
        lora_config: Some(config.lora.clone()),
    })
}

/// Try to load base weights from a SafeTensors file.
/// Returns `None` if the model path is not a .safetensors file.
pub(super) fn try_load_safetensors_weights(
    config: &LocalTrainingConfig,
    dim: usize,
    device: &WgpuDevice,
) -> Option<Tensor<TrainBackend, 2>> {
    let path = &config.model_path;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext != "safetensors" {
        return None;
    }

    match SafeTensorsLoader::open(path) {
        Ok(loader) => {
            let names = loader.tensor_names();
            // Try to find a suitable weight tensor matching our dimensions
            let target_names = [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "lm_head.weight",
            ];

            for name in &target_names {
                if names.iter().any(|n| n == *name) {
                    match loader.load_tensor_f32(name) {
                        Ok((data, shape)) => {
                            if shape.len() == 2 && shape[0] == dim && shape[1] == dim {
                                info!(
                                    "Loaded base weights from '{}' [{},{}]",
                                    name, shape[0], shape[1]
                                );
                                let tensor = Tensor::<TrainBackend, 1>::from_floats(
                                    burn_core::tensor::TensorData::new(data, [dim * dim]),
                                    device,
                                )
                                .reshape([dim, dim]);
                                return Some(tensor);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to load tensor '{}': {}", name, e);
                        }
                    }
                }
            }

            warn!(
                "SafeTensors file opened but no tensor with matching dimensions [{}x{}] found, using random init",
                dim, dim
            );
            None
        }
        Err(e) => {
            warn!("Failed to open SafeTensors file: {}, using random init", e);
            None
        }
    }
}

/// Try to load quantized base weights from a SafeTensors file.
pub(super) fn try_load_quantized_weights(
    config: &LocalTrainingConfig,
    dim: usize,
    bits: u8,
    _device: &WgpuDevice,
) -> Option<Vec<f32>> {
    let path = &config.model_path;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext != "safetensors" {
        return None;
    }

    let quant_config = match bits {
        4 => QuantConfig::int4(),
        8 => QuantConfig::int8(),
        _ => {
            warn!("Unsupported quantization bits: {}, using 4-bit", bits);
            QuantConfig::int4()
        }
    };

    match SafeTensorsLoader::open(path) {
        Ok(loader) => {
            let names = loader.tensor_names();
            let target_names = [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ];

            for name in &target_names {
                if names.iter().any(|n| n == *name) {
                    match loader.load_tensor_quantized(name, &quant_config) {
                        Ok((data, shape)) => {
                            if shape.len() == 2 && shape[0] == dim && shape[1] == dim {
                                info!(
                                    "Loaded {}-bit quantized base weights from '{}' [{},{}]",
                                    bits, name, shape[0], shape[1]
                                );
                                return Some(data);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to load quantized tensor '{}': {}", name, e);
                        }
                    }
                }
            }

            warn!("No matching quantized tensor found, using random init");
            None
        }
        Err(e) => {
            warn!("Failed to open SafeTensors file: {}, using random init", e);
            None
        }
    }
}
