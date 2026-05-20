use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use brainwires_memory::dream::policy::DemotionPolicy;

/// Top-level configuration for claude-brain.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClaudeBrainConfig {
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub policy: PolicyConfig,
    #[serde(default)]
    pub session_start: SessionStartConfig,
    #[serde(default)]
    pub capture: CaptureConfig,
}

/// Storage paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    pub brain_path: String,
    pub pks_path: String,
    pub bks_path: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        let base = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".brainwires");
        Self {
            brain_path: base.join("claude-brain").to_string_lossy().into_owned(),
            pks_path: base.join("pks.db").to_string_lossy().into_owned(),
            bks_path: base.join("bks.db").to_string_lossy().into_owned(),
        }
    }
}

/// Demotion policy tunables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Hours before hot-tier messages become consolidation candidates.
    #[serde(default = "default_hot_max_age_hours")]
    pub hot_max_age_hours: u64,
    /// Days before warm-tier summaries become fact-extraction candidates.
    #[serde(default = "default_warm_max_age_days")]
    pub warm_max_age_days: u64,
    /// Token budget for the hot tier.
    #[serde(default = "default_hot_token_budget")]
    pub hot_token_budget: usize,
    /// Number of recent messages always kept in hot tier.
    #[serde(default = "default_keep_recent")]
    pub keep_recent: usize,
    /// Minimum importance score for hot-tier retention.
    #[serde(default = "default_min_importance")]
    pub min_importance: f32,
}

fn default_hot_max_age_hours() -> u64 {
    24
}
fn default_warm_max_age_days() -> u64 {
    7
}
fn default_hot_token_budget() -> usize {
    50_000
}
fn default_keep_recent() -> usize {
    4
}
fn default_min_importance() -> f32 {
    0.3
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            hot_max_age_hours: default_hot_max_age_hours(),
            warm_max_age_days: default_warm_max_age_days(),
            hot_token_budget: default_hot_token_budget(),
            keep_recent: default_keep_recent(),
            min_importance: default_min_importance(),
        }
    }
}

impl PolicyConfig {
    /// Convert to the framework's DemotionPolicy.
    pub fn to_demotion_policy(&self) -> DemotionPolicy {
        DemotionPolicy {
            hot_max_age: Duration::from_secs(self.hot_max_age_hours * 3600),
            warm_max_age: Duration::from_secs(self.warm_max_age_days * 24 * 3600),
            hot_token_budget: self.hot_token_budget,
            keep_recent: self.keep_recent,
            min_importance: self.min_importance,
        }
    }
}

/// Tuning for session-start context loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStartConfig {
    /// Maximum number of cold-tier facts to load.
    #[serde(default = "default_max_facts")]
    pub max_facts: usize,
    /// Maximum number of warm-tier summaries to load.
    #[serde(default = "default_max_summaries")]
    pub max_summaries: usize,
    /// Maximum token budget for loaded context.
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,
}

fn default_max_facts() -> usize {
    20
}
fn default_max_summaries() -> usize {
    5
}
fn default_max_context_tokens() -> usize {
    4000
}

impl Default for SessionStartConfig {
    fn default() -> Self {
        Self {
            max_facts: default_max_facts(),
            max_summaries: default_max_summaries(),
            max_context_tokens: default_max_context_tokens(),
        }
    }
}

/// Capture behaviour for the stop hook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    /// Extract facts from captured turns.
    #[serde(default = "default_true")]
    pub extract_facts: bool,
    /// Turn count threshold before triggering consolidation.
    #[serde(default = "default_consolidation_threshold")]
    pub consolidation_threshold: usize,
}

fn default_true() -> bool {
    true
}
fn default_consolidation_threshold() -> usize {
    20
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            extract_facts: true,
            consolidation_threshold: default_consolidation_threshold(),
        }
    }
}

impl ClaudeBrainConfig {
    /// Load from `~/.brainwires/claude-brain.toml`, falling back to defaults.
    pub fn load() -> Result<Self> {
        let config_path = dirs::home_dir()
            .context("Cannot determine home directory")?
            .join(".brainwires")
            .join("claude-brain.toml");

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read {}", config_path.display()))?;
            let config: Self = toml::from_str(&content)
                .with_context(|| format!("Failed to parse {}", config_path.display()))?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }
}
