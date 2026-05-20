//! Configuration types for local LLM inference
//!
//! Defines settings for local model loading and inference parameters.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Default batch size for local LLM prompt processing.
const DEFAULT_LOCAL_LLM_BATCH_SIZE: usize = 512;

/// Configuration for a local LLM model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalLlmConfig {
    /// Unique identifier for this model configuration
    pub id: String,

    /// Human-readable name for the model
    pub name: String,

    /// Path to the GGUF model file
    pub model_path: PathBuf,

    /// Context window size (default: 4096)
    #[serde(default = "default_context_size")]
    pub context_size: u32,

    /// Number of CPU threads to use (default: auto-detect)
    #[serde(default)]
    pub num_threads: Option<u32>,

    /// Batch size for prompt processing (default: 512)
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,

    /// GPU layers to offload (0 = CPU only, default: 0)
    #[serde(default)]
    pub gpu_layers: u32,

    /// Enable memory mapping for faster loading (default: true)
    #[serde(default = "default_true")]
    pub use_mmap: bool,

    /// Enable memory locking to prevent swapping (default: false)
    #[serde(default)]
    pub use_mlock: bool,

    /// Maximum tokens to generate per response (default: 2048)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Model family/type for proper prompt formatting
    #[serde(default)]
    pub model_type: LocalModelType,

    /// Optional system prompt template override
    #[serde(default)]
    pub system_template: Option<String>,

    /// Whether this model supports tool/function calling
    #[serde(default)]
    pub supports_tools: bool,

    /// Estimated RAM usage in MB (for display purposes)
    #[serde(default)]
    pub estimated_ram_mb: Option<u32>,
}

fn default_context_size() -> u32 {
    4096
}

fn default_batch_size() -> u32 {
    DEFAULT_LOCAL_LLM_BATCH_SIZE as u32
}

fn default_max_tokens() -> u32 {
    2048
}

fn default_true() -> bool {
    true
}

impl Default for LocalLlmConfig {
    fn default() -> Self {
        Self {
            id: "local-model".to_string(),
            name: "Local Model".to_string(),
            model_path: PathBuf::new(),
            context_size: default_context_size(),
            num_threads: None,
            batch_size: default_batch_size(),
            gpu_layers: 0,
            use_mmap: true,
            use_mlock: false,
            max_tokens: default_max_tokens(),
            model_type: LocalModelType::default(),
            system_template: None,
            supports_tools: false,
            estimated_ram_mb: None,
        }
    }
}

impl LocalLlmConfig {
    /// Create a new configuration for an LFM2 model
    pub fn lfm2_350m(model_path: PathBuf) -> Self {
        Self {
            id: "lfm2-350m".to_string(),
            name: "LFM2 350M".to_string(),
            model_path,
            context_size: 32768,
            batch_size: 512,
            max_tokens: 2048,
            model_type: LocalModelType::Lfm2,
            supports_tools: false,
            estimated_ram_mb: Some(220),
            ..Default::default()
        }
    }

    /// Create a new configuration for LFM2-1.2B model
    pub fn lfm2_1_2b(model_path: PathBuf) -> Self {
        Self {
            id: "lfm2-1.2b".to_string(),
            name: "LFM2 1.2B".to_string(),
            model_path,
            context_size: 32768,
            batch_size: 512,
            max_tokens: 2048,
            model_type: LocalModelType::Lfm2,
            supports_tools: false,
            estimated_ram_mb: Some(700),
            ..Default::default()
        }
    }

    /// Create a new configuration for LFM2-2.6B-Exp (agentic) model
    pub fn lfm2_2_6b_exp(model_path: PathBuf) -> Self {
        Self {
            id: "lfm2-2.6b-exp".to_string(),
            name: "LFM2 2.6B Experimental".to_string(),
            model_path,
            context_size: 32768,
            batch_size: 512,
            max_tokens: 4096,
            model_type: LocalModelType::Lfm2Agentic,
            supports_tools: true,
            estimated_ram_mb: Some(1500),
            ..Default::default()
        }
    }

    /// Create a new configuration for Granite 4.0 Nano 350M
    pub fn granite_nano_350m(model_path: PathBuf) -> Self {
        Self {
            id: "granite-nano-350m".to_string(),
            name: "Granite 4.0 Nano 350M".to_string(),
            model_path,
            context_size: 8192,
            batch_size: 512,
            max_tokens: 2048,
            model_type: LocalModelType::Granite,
            supports_tools: false,
            estimated_ram_mb: Some(250),
            ..Default::default()
        }
    }

    /// Create a new configuration for Granite 4.0 Nano 1.5B
    pub fn granite_nano_1_5b(model_path: PathBuf) -> Self {
        Self {
            id: "granite-nano-1.5b".to_string(),
            name: "Granite 4.0 Nano 1.5B".to_string(),
            model_path,
            context_size: 8192,
            batch_size: 512,
            max_tokens: 2048,
            model_type: LocalModelType::Granite,
            supports_tools: false,
            estimated_ram_mb: Some(900),
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), LocalLlmConfigError> {
        if self.model_path.as_os_str().is_empty() {
            return Err(LocalLlmConfigError::MissingModelPath);
        }
        if !self.model_path.exists() {
            return Err(LocalLlmConfigError::ModelNotFound(self.model_path.clone()));
        }
        if self.context_size == 0 {
            return Err(LocalLlmConfigError::InvalidContextSize);
        }
        if self.batch_size == 0 {
            return Err(LocalLlmConfigError::InvalidBatchSize);
        }
        Ok(())
    }
}

/// Model type/family for proper prompt formatting
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum LocalModelType {
    /// LFM2 (Liquid Foundation Model 2) - hybrid architecture
    #[default]
    Lfm2,
    /// LFM2 Experimental variant optimized for agentic tasks
    Lfm2Agentic,
    /// Granite family models (IBM)
    Granite,
    /// Qwen family models (Alibaba)
    Qwen,
    /// Llama family models (Meta)
    Llama,
    /// Phi family models (Microsoft)
    Phi,
    /// Generic/unknown model type
    Generic,
}

impl LocalModelType {
    /// Get the chat template for this model type
    pub fn chat_template(&self) -> &'static str {
        match self {
            Self::Lfm2 | Self::Lfm2Agentic => {
                "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
            }
            Self::Granite => "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n",
            Self::Qwen => {
                "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            }
            Self::Llama => {
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            }
            Self::Phi => "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n",
            Self::Generic => "### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n",
        }
    }

    /// Get the stop tokens for this model type
    pub fn stop_tokens(&self) -> Vec<&'static str> {
        match self {
            Self::Lfm2 | Self::Lfm2Agentic => vec!["<|end|>", "<|user|>"],
            Self::Granite => vec!["<|user|>", "<|system|>"],
            Self::Qwen => vec!["<|im_end|>", "<|im_start|>"],
            Self::Llama => vec!["<|eot_id|>", "<|start_header_id|>"],
            Self::Phi => vec!["<|end|>", "<|user|>"],
            Self::Generic => vec!["### User:", "### System:"],
        }
    }
}

/// Configuration errors for local LLM
#[derive(Debug, thiserror::Error)]
pub enum LocalLlmConfigError {
    /// Model path was not provided.
    #[error("Model path is required")]
    MissingModelPath,

    /// Model file does not exist at the given path.
    #[error("Model file not found: {0}")]
    ModelNotFound(PathBuf),

    /// Context size was set to zero.
    #[error("Context size must be greater than 0")]
    InvalidContextSize,

    /// Batch size was set to zero.
    #[error("Batch size must be greater than 0")]
    InvalidBatchSize,

    /// Model loading failed.
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    /// Inference failed during generation.
    #[error("Inference error: {0}")]
    InferenceError(String),
}

/// Inference parameters for a single generation request
#[derive(Debug, Clone)]
pub struct LocalInferenceParams {
    /// Temperature for sampling (0.0 = deterministic, 1.0 = random)
    pub temperature: f32,
    /// Top-p (nucleus) sampling parameter
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: u32,
    /// Repetition penalty (1.0 = no penalty)
    pub repeat_penalty: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

impl Default for LocalInferenceParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            max_tokens: 2048,
            stop_sequences: Vec::new(),
        }
    }
}

impl LocalInferenceParams {
    /// Create params optimized for deterministic, factual responses
    pub fn factual() -> Self {
        Self {
            temperature: 0.1,
            top_p: 0.9,
            top_k: 20,
            repeat_penalty: 1.0,
            max_tokens: 1024,
            stop_sequences: Vec::new(),
        }
    }

    /// Create params for creative, varied responses
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 50,
            repeat_penalty: 1.2,
            max_tokens: 2048,
            stop_sequences: Vec::new(),
        }
    }

    /// Create params for routing/classification tasks
    pub fn routing() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            repeat_penalty: 1.0,
            max_tokens: 50,
            stop_sequences: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LocalLlmConfig::default();
        assert_eq!(config.context_size, 4096);
        assert_eq!(config.batch_size, 512);
        assert!(config.use_mmap);
        assert!(!config.use_mlock);
    }

    #[test]
    fn test_lfm2_350m_config() {
        let config = LocalLlmConfig::lfm2_350m(PathBuf::from("/models/lfm2-350m.gguf"));
        assert_eq!(config.id, "lfm2-350m");
        assert_eq!(config.context_size, 32768);
        assert_eq!(config.estimated_ram_mb, Some(220));
        assert!(!config.supports_tools);
    }

    #[test]
    fn test_lfm2_2_6b_exp_config() {
        let config = LocalLlmConfig::lfm2_2_6b_exp(PathBuf::from("/models/lfm2-2.6b-exp.gguf"));
        assert_eq!(config.id, "lfm2-2.6b-exp");
        assert!(config.supports_tools);
        assert_eq!(config.model_type, LocalModelType::Lfm2Agentic);
    }

    #[test]
    fn test_model_type_chat_templates() {
        let lfm2 = LocalModelType::Lfm2;
        assert!(lfm2.chat_template().contains("<|system|>"));
        assert!(lfm2.chat_template().contains("<|user|>"));

        let qwen = LocalModelType::Qwen;
        assert!(qwen.chat_template().contains("<|im_start|>"));
    }

    #[test]
    fn test_model_type_stop_tokens() {
        let lfm2 = LocalModelType::Lfm2;
        assert!(lfm2.stop_tokens().contains(&"<|end|>"));

        let llama = LocalModelType::Llama;
        assert!(llama.stop_tokens().contains(&"<|eot_id|>"));
    }

    #[test]
    fn test_inference_params_presets() {
        let factual = LocalInferenceParams::factual();
        assert!(factual.temperature < 0.5);

        let creative = LocalInferenceParams::creative();
        assert!(creative.temperature > 0.7);

        let routing = LocalInferenceParams::routing();
        assert_eq!(routing.temperature, 0.0);
        assert_eq!(routing.top_k, 1);
    }

    #[test]
    fn test_config_validation_missing_path() {
        let config = LocalLlmConfig::default();
        let result = config.validate();
        assert!(matches!(result, Err(LocalLlmConfigError::MissingModelPath)));
    }
}
