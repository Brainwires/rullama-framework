use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::info;

/// Export format for trained models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// GGUF format (for llama.cpp / Ollama inference).
    Gguf,
    /// SafeTensors format (HuggingFace compatible).
    SafeTensors,
    /// Adapter-only weights (LoRA/QLoRA/DoRA).
    AdapterOnly,
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf => write!(f, "gguf"),
            Self::SafeTensors => write!(f, "safetensors"),
            Self::AdapterOnly => write!(f, "adapter_only"),
        }
    }
}

/// Configuration for model export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Output format.
    pub format: ExportFormat,
    /// Output file path.
    pub output_path: PathBuf,
    /// Quantization for GGUF export (e.g., "Q4_K_M", "Q5_K_S").
    pub gguf_quantization: Option<String>,
    /// Include model metadata.
    pub include_metadata: bool,
}

impl ExportConfig {
    /// Create a GGUF export configuration with default Q4_K_M quantization.
    pub fn gguf(output_path: impl Into<PathBuf>) -> Self {
        Self {
            format: ExportFormat::Gguf,
            output_path: output_path.into(),
            gguf_quantization: Some("Q4_K_M".to_string()),
            include_metadata: true,
        }
    }

    /// Create a SafeTensors export configuration.
    pub fn safetensors(output_path: impl Into<PathBuf>) -> Self {
        Self {
            format: ExportFormat::SafeTensors,
            output_path: output_path.into(),
            gguf_quantization: None,
            include_metadata: true,
        }
    }

    /// Create an adapter-only export configuration.
    pub fn adapter_only(output_path: impl Into<PathBuf>) -> Self {
        Self {
            format: ExportFormat::AdapterOnly,
            output_path: output_path.into(),
            gguf_quantization: None,
            include_metadata: true,
        }
    }
}

/// Export metadata written alongside the model.
#[derive(Debug, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Export format name (e.g., "gguf", "safetensors", "adapter_only").
    pub format: String,
    /// Path or identifier of the base model used for training.
    pub base_model: String,
    /// Adapter method used (e.g., "LoRA", "QLoRA", "DoRA"), if applicable.
    pub adapter_method: Option<String>,
    /// Number of training epochs completed.
    pub training_epochs: u32,
    /// Final training loss at export time.
    pub final_loss: Option<f64>,
    /// Timestamp when the model was exported.
    pub exported_at: chrono::DateTime<chrono::Utc>,
}

/// Write export metadata to a JSON file next to the model.
pub fn write_export_metadata(output_dir: &Path, metadata: &ExportMetadata) -> std::io::Result<()> {
    let meta_path = output_dir.join("export_metadata.json");
    let json = serde_json::to_string_pretty(metadata).map_err(std::io::Error::other)?;
    std::fs::write(&meta_path, json)?;
    info!("Export metadata written to {:?}", meta_path);
    Ok(())
}

/// Export a trained model in the specified format.
pub fn export_model(
    config: &ExportConfig,
    weights: &std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    metadata: &ExportMetadata,
) -> std::io::Result<()> {
    std::fs::create_dir_all(&config.output_path)?;

    match config.format {
        ExportFormat::SafeTensors => {
            let tensors: std::collections::HashMap<String, safetensors::tensor::TensorView<'_>> =
                weights
                    .iter()
                    .filter_map(|(name, (data, shape))| {
                        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                        let bytes = Box::leak(bytes.into_boxed_slice());
                        safetensors::tensor::TensorView::new(
                            safetensors::Dtype::F32,
                            shape.clone(),
                            bytes,
                        )
                        .ok()
                        .map(|view| (name.clone(), view))
                    })
                    .collect();

            let serialized = safetensors::tensor::serialize(&tensors, None)
                .map_err(|e| std::io::Error::other(format!("SafeTensors error: {}", e)))?;
            std::fs::write(config.output_path.join("model.safetensors"), serialized)?;
            info!("Exported {} tensors as SafeTensors", weights.len());
        }
        ExportFormat::AdapterOnly => {
            let adapter_weights: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)> =
                weights
                    .iter()
                    .filter(|(name, _)| {
                        name.contains("lora_a")
                            || name.contains("lora_b")
                            || name.contains("magnitude")
                    })
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();

            let tensors: std::collections::HashMap<String, safetensors::tensor::TensorView<'_>> =
                adapter_weights
                    .iter()
                    .filter_map(|(name, (data, shape))| {
                        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                        let bytes = Box::leak(bytes.into_boxed_slice());
                        safetensors::tensor::TensorView::new(
                            safetensors::Dtype::F32,
                            shape.clone(),
                            bytes,
                        )
                        .ok()
                        .map(|view| (name.clone(), view))
                    })
                    .collect();

            let serialized = safetensors::tensor::serialize(&tensors, None)
                .map_err(|e| std::io::Error::other(format!("SafeTensors error: {}", e)))?;
            std::fs::write(
                config.output_path.join("adapter_weights.safetensors"),
                serialized,
            )?;
            info!("Exported {} adapter tensors", adapter_weights.len());
        }
        ExportFormat::Gguf => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "GGUF export not supported. Convert SafeTensors output using llama.cpp tools (convert-safetensors-to-gguf.py).",
            ));
        }
    }

    if config.include_metadata {
        write_export_metadata(&config.output_path, metadata)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_format_display() {
        assert_eq!(ExportFormat::Gguf.to_string(), "gguf");
        assert_eq!(ExportFormat::SafeTensors.to_string(), "safetensors");
        assert_eq!(ExportFormat::AdapterOnly.to_string(), "adapter_only");
    }

    #[test]
    fn test_export_config_builders() {
        let gguf = ExportConfig::gguf("/tmp/model.gguf");
        assert_eq!(gguf.format, ExportFormat::Gguf);
        assert!(gguf.gguf_quantization.is_some());

        let st = ExportConfig::safetensors("/tmp/model.safetensors");
        assert_eq!(st.format, ExportFormat::SafeTensors);
        assert!(st.gguf_quantization.is_none());
    }

    #[test]
    fn test_export_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let config = ExportConfig::safetensors(dir.path());

        let mut weights = std::collections::HashMap::new();
        weights.insert(
            "layer.weight".to_string(),
            (vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]),
        );

        let metadata = ExportMetadata {
            format: "safetensors".to_string(),
            base_model: "test-model".to_string(),
            adapter_method: Some("LoRA".to_string()),
            training_epochs: 3,
            final_loss: Some(0.5),
            exported_at: chrono::Utc::now(),
        };

        export_model(&config, &weights, &metadata).unwrap();
        assert!(dir.path().join("model.safetensors").exists());
        assert!(dir.path().join("export_metadata.json").exists());
    }

    #[test]
    fn test_export_adapter_only() {
        let dir = tempfile::tempdir().unwrap();
        let config = ExportConfig::adapter_only(dir.path());

        let mut weights = std::collections::HashMap::new();
        weights.insert("layer.lora_a".to_string(), (vec![1.0f32, 2.0], vec![1, 2]));
        weights.insert("layer.lora_b".to_string(), (vec![3.0f32, 4.0], vec![2, 1]));
        weights.insert(
            "layer.base_weight".to_string(),
            (vec![5.0f32; 100], vec![10, 10]),
        );

        let metadata = ExportMetadata {
            format: "adapter_only".to_string(),
            base_model: "test-model".to_string(),
            adapter_method: Some("LoRA".to_string()),
            training_epochs: 3,
            final_loss: Some(0.5),
            exported_at: chrono::Utc::now(),
        };

        export_model(&config, &weights, &metadata).unwrap();
        assert!(dir.path().join("adapter_weights.safetensors").exists());
    }

    #[test]
    fn test_export_gguf_error() {
        let dir = tempfile::tempdir().unwrap();
        let config = ExportConfig::gguf(dir.path());
        let weights = std::collections::HashMap::new();
        let metadata = ExportMetadata {
            format: "gguf".to_string(),
            base_model: "test".to_string(),
            adapter_method: None,
            training_epochs: 1,
            final_loss: None,
            exported_at: chrono::Utc::now(),
        };
        assert!(export_model(&config, &weights, &metadata).is_err());
    }
}
