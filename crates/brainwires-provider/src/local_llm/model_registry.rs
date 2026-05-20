//! Local Model Registry
//!
//! Manages registered local LLM models, their paths, and metadata.

use super::config::{LocalLlmConfig, LocalModelType};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Registry of available local LLM models
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LocalModelRegistry {
    /// Registered models by ID
    pub models: HashMap<String, LocalLlmConfig>,
    /// Default model ID to use
    pub default_model: Option<String>,
    /// Base directory for model storage
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,
}

fn default_models_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("brainwires")
        .join("models")
}

impl LocalModelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with default models directory
    pub fn with_default_dir() -> Self {
        Self {
            models: HashMap::new(),
            default_model: None,
            models_dir: default_models_dir(),
        }
    }

    /// Get the models directory path
    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
    }

    /// Set the models directory
    pub fn set_models_dir(&mut self, dir: PathBuf) {
        self.models_dir = dir;
    }

    /// Register a new model
    pub fn register(&mut self, config: LocalLlmConfig) {
        let id = config.id.clone();
        self.models.insert(id, config);
    }

    /// Get a model by ID
    pub fn get(&self, id: &str) -> Option<&LocalLlmConfig> {
        self.models.get(id)
    }

    /// Get the default model
    pub fn get_default(&self) -> Option<&LocalLlmConfig> {
        self.default_model
            .as_ref()
            .and_then(|id| self.models.get(id))
    }

    /// Set the default model
    pub fn set_default(&mut self, id: &str) -> bool {
        if self.models.contains_key(id) {
            self.default_model = Some(id.to_string());
            true
        } else {
            false
        }
    }

    /// Remove a model from the registry
    pub fn remove(&mut self, id: &str) -> Option<LocalLlmConfig> {
        if self.default_model.as_deref() == Some(id) {
            self.default_model = None;
        }
        self.models.remove(id)
    }

    /// List all registered models
    pub fn list(&self) -> Vec<&LocalLlmConfig> {
        self.models.values().collect()
    }

    /// Scan the models directory for GGUF files and auto-register them
    pub fn scan_models_dir(&mut self) -> Result<Vec<String>> {
        let mut discovered = Vec::new();

        if !self.models_dir.exists() {
            std::fs::create_dir_all(&self.models_dir)
                .context("Failed to create models directory")?;
            return Ok(discovered);
        }

        for entry in std::fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                let filename = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");

                // Skip if already registered
                if self.models.contains_key(filename) {
                    continue;
                }

                // Auto-detect model type from filename
                let model_type = detect_model_type(filename);
                let config = create_config_from_filename(filename, path.clone(), model_type);

                self.register(config);
                discovered.push(filename.to_string());
            }
        }

        Ok(discovered)
    }

    /// Load registry from the config file
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;
        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)
                .context("Failed to read local models config")?;
            let registry: LocalModelRegistry =
                serde_json::from_str(&contents).context("Failed to parse local models config")?;
            Ok(registry)
        } else {
            Ok(Self::with_default_dir())
        }
    }

    /// Save registry to the config file
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;
        Ok(())
    }

    /// Get the path to the config file
    fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?;
        Ok(config_dir.join("brainwires").join("local_models.json"))
    }
}

/// Detect model type from filename
fn detect_model_type(filename: &str) -> LocalModelType {
    let lower = filename.to_lowercase();

    if lower.contains("lfm2") || lower.contains("lfm-2") || lower.contains("liquid") {
        if lower.contains("exp") || lower.contains("agent") {
            LocalModelType::Lfm2Agentic
        } else {
            LocalModelType::Lfm2
        }
    } else if lower.contains("granite") {
        LocalModelType::Granite
    } else if lower.contains("qwen") {
        LocalModelType::Qwen
    } else if lower.contains("llama") {
        LocalModelType::Llama
    } else if lower.contains("phi") {
        LocalModelType::Phi
    } else {
        LocalModelType::Generic
    }
}

/// Create a config from a filename with detected model type
fn create_config_from_filename(
    filename: &str,
    path: PathBuf,
    model_type: LocalModelType,
) -> LocalLlmConfig {
    let lower = filename.to_lowercase();

    // Try to detect model size from filename
    let (context_size, estimated_ram) = if lower.contains("350m") || lower.contains("0.3b") {
        (8192, Some(220))
    } else if lower.contains("1.2b") || lower.contains("1b") {
        (16384, Some(700))
    } else if lower.contains("1.5b") {
        (16384, Some(900))
    } else if lower.contains("2.6b") || lower.contains("3b") || lower.contains("2b") {
        (32768, Some(1500))
    } else if lower.contains("7b") {
        (32768, Some(4000))
    } else {
        (4096, None)
    };

    LocalLlmConfig {
        id: filename.to_string(),
        name: format_model_name(filename),
        model_path: path,
        context_size,
        model_type,
        estimated_ram_mb: estimated_ram,
        supports_tools: model_type == LocalModelType::Lfm2Agentic,
        ..Default::default()
    }
}

/// Format a filename into a human-readable model name
fn format_model_name(filename: &str) -> String {
    filename
        .replace(['-', '_'], " ")
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Pre-configured model definitions for easy downloading
#[derive(Debug, Clone)]
pub struct KnownModel {
    /// Unique model identifier.
    pub id: &'static str,
    /// Human-readable model name.
    pub name: &'static str,
    /// HuggingFace repository path.
    pub huggingface_repo: &'static str,
    /// Model filename within the repository.
    pub filename: &'static str,
    /// Model architecture type.
    pub model_type: LocalModelType,
    /// Default context window size in tokens.
    pub context_size: u32,
    /// Estimated RAM usage in megabytes.
    pub estimated_ram_mb: u32,
    /// Whether the model supports tool/function calling.
    pub supports_tools: bool,
    /// Short description of the model.
    pub description: &'static str,
    /// Hugging Face revision (branch, tag, or commit SHA) to pin against.
    ///
    /// Defaults to `"main"` for legacy GGUF entries.
    pub huggingface_revision: &'static str,
    /// Tokenizer asset filename within the repository, when the model uses a
    /// `tokenizer.json` companion (Candle / safetensors path). Empty for GGUF.
    pub tokenizer_filename: &'static str,
    /// SHA-256 of the weights file, hex encoded. `None` until weights are
    /// published and pinned.
    pub weights_sha256: Option<&'static str>,
    /// SHA-256 of the tokenizer file, hex encoded. `None` until weights are
    /// published and pinned.
    pub tokenizer_sha256: Option<&'static str>,
}

/// Get list of known/recommended models
pub fn known_models() -> Vec<KnownModel> {
    vec![
        KnownModel {
            id: "lfm2-350m",
            name: "LFM2 350M",
            huggingface_repo: "LiquidAI/LFM2-350M-GGUF",
            filename: "lfm2-350m-q8_0.gguf",
            model_type: LocalModelType::Lfm2,
            context_size: 32768,
            estimated_ram_mb: 220,
            supports_tools: false,
            description: "Fastest. For routing, binary decisions. ~220MB RAM.",
            huggingface_revision: "main",
            tokenizer_filename: "",
            weights_sha256: None,
            tokenizer_sha256: None,
        },
        KnownModel {
            id: "lfm2-1.2b",
            name: "LFM2 1.2B",
            huggingface_repo: "LiquidAI/LFM2-1.2B-GGUF",
            filename: "lfm2-1.2b-q8_0.gguf",
            model_type: LocalModelType::Lfm2,
            context_size: 32768,
            estimated_ram_mb: 700,
            supports_tools: false,
            description: "Sweet spot for agentic logic. Competitive with larger models.",
            huggingface_revision: "main",
            tokenizer_filename: "",
            weights_sha256: None,
            tokenizer_sha256: None,
        },
        KnownModel {
            id: "lfm2-2.6b-exp",
            name: "LFM2 2.6B Experimental",
            huggingface_repo: "LiquidAI/LFM2-2.6B-Exp-GGUF",
            filename: "lfm2-2.6b-exp-q8_0.gguf",
            model_type: LocalModelType::Lfm2Agentic,
            context_size: 32768,
            estimated_ram_mb: 1500,
            supports_tools: true,
            description: "Complex reasoning, tool-calling. Best for agents.",
            huggingface_revision: "main",
            tokenizer_filename: "",
            weights_sha256: None,
            tokenizer_sha256: None,
        },
        KnownModel {
            id: "granite-nano-350m",
            name: "Granite 4.0 Nano 350M",
            huggingface_repo: "ibm-granite/granite-4.0-nano-350m-gguf",
            filename: "granite-4.0-nano-350m-q8_0.gguf",
            model_type: LocalModelType::Granite,
            context_size: 8192,
            estimated_ram_mb: 250,
            supports_tools: false,
            description: "Sub-second CPU responses. Classification, summarization.",
            huggingface_revision: "main",
            tokenizer_filename: "",
            weights_sha256: None,
            tokenizer_sha256: None,
        },
        KnownModel {
            id: "granite-nano-1.5b",
            name: "Granite 4.0 Nano 1.5B",
            huggingface_repo: "ibm-granite/granite-4.0-nano-1.5b-gguf",
            filename: "granite-4.0-nano-1.5b-q8_0.gguf",
            model_type: LocalModelType::Granite,
            context_size: 8192,
            estimated_ram_mb: 900,
            supports_tools: false,
            description: "Balanced performance. Good for business tasks.",
            huggingface_revision: "main",
            tokenizer_filename: "",
            weights_sha256: None,
            tokenizer_sha256: None,
        },
        // Candle / safetensors path — used by `CandleLlmProvider` (WASM-friendly).
        // Hashes pulled from HF API: huggingface.co/api/models/google/gemma-4-E2B/tree/main
        KnownModel {
            id: "gemma-4-e2b",
            name: "Gemma 4 E2B",
            huggingface_repo: "google/gemma-4-e2b",
            filename: "model.safetensors",
            model_type: LocalModelType::Generic,
            context_size: 8192,
            estimated_ram_mb: 10000,
            supports_tools: false,
            description: "Gemma 4 E2B (5.1B params, multimodal) — Candle/safetensors.",
            huggingface_revision: "main",
            tokenizer_filename: "tokenizer.json",
            weights_sha256: Some(
                "76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57",
            ),
            tokenizer_sha256: Some(
                "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6",
            ),
        },
    ]
}

/// Get a known model by ID
pub fn get_known_model(id: &str) -> Option<KnownModel> {
    known_models().into_iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_new() {
        let registry = LocalModelRegistry::new();
        assert!(registry.models.is_empty());
        assert!(registry.default_model.is_none());
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = LocalModelRegistry::new();
        let config = LocalLlmConfig::lfm2_350m(PathBuf::from("/models/test.gguf"));

        registry.register(config.clone());
        assert!(registry.get("lfm2-350m").is_some());
        assert_eq!(registry.get("lfm2-350m").unwrap().name, "LFM2 350M");
    }

    #[test]
    fn test_set_default() {
        let mut registry = LocalModelRegistry::new();
        let config = LocalLlmConfig::lfm2_350m(PathBuf::from("/models/test.gguf"));

        registry.register(config);
        assert!(registry.set_default("lfm2-350m"));
        assert!(registry.get_default().is_some());

        assert!(!registry.set_default("nonexistent"));
    }

    #[test]
    fn test_detect_model_type() {
        assert_eq!(detect_model_type("lfm2-350m"), LocalModelType::Lfm2);
        assert_eq!(
            detect_model_type("lfm2-2.6b-exp"),
            LocalModelType::Lfm2Agentic
        );
        assert_eq!(detect_model_type("granite-nano"), LocalModelType::Granite);
        assert_eq!(detect_model_type("qwen3-1.7b"), LocalModelType::Qwen);
        assert_eq!(detect_model_type("unknown-model"), LocalModelType::Generic);
    }

    #[test]
    fn test_format_model_name() {
        assert_eq!(format_model_name("lfm2-350m"), "Lfm2 350m");
        assert_eq!(format_model_name("granite_nano_1.5b"), "Granite Nano 1.5b");
    }

    #[test]
    fn test_known_models() {
        let models = known_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.id == "lfm2-350m"));
        assert!(models.iter().any(|m| m.id == "granite-nano-350m"));
    }

    #[test]
    fn test_get_known_model() {
        let model = get_known_model("lfm2-1.2b");
        assert!(model.is_some());
        assert_eq!(model.unwrap().estimated_ram_mb, 700);

        assert!(get_known_model("nonexistent").is_none());
    }
}
