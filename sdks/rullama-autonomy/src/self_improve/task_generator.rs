//! Task generation from improvement strategies.

use anyhow::Result;

use super::strategies::{self, ImprovementStrategy, ImprovementTask};
use crate::config::{SelfImprovementConfig, StrategyConfig};

/// Generates improvement tasks by running enabled strategies.
pub struct TaskGenerator {
    strategies: Vec<Box<dyn ImprovementStrategy>>,
}

impl TaskGenerator {
    /// Create a new task generator with the given strategies.
    pub fn new(strategies: Vec<Box<dyn ImprovementStrategy>>) -> Self {
        Self { strategies }
    }

    /// Create from config, filtering to only enabled strategies.
    pub fn from_config(config: &SelfImprovementConfig) -> Self {
        let all = strategies::all_strategies();
        let filtered: Vec<Box<dyn ImprovementStrategy>> = if config.strategies.is_empty() {
            all
        } else {
            all.into_iter()
                .filter(|s| config.is_strategy_enabled(s.name()))
                .collect()
        };
        Self::new(filtered)
    }

    /// Generate all tasks from all enabled strategies, sorted by priority (highest first).
    pub async fn generate_all(
        &self,
        config: &SelfImprovementConfig,
    ) -> Result<Vec<ImprovementTask>> {
        let strategy_config = StrategyConfig {
            repo_path: std::env::current_dir()?.to_string_lossy().to_string(),
            max_tasks_per_strategy: 5,
        };

        let mut all_tasks = Vec::new();

        for strategy in &self.strategies {
            if !config.is_strategy_enabled(strategy.name()) {
                continue;
            }

            tracing::info!("Running strategy: {}", strategy.name());

            match strategy
                .generate_tasks(&strategy_config.repo_path, &strategy_config)
                .await
            {
                Ok(tasks) => {
                    tracing::info!(
                        "Strategy '{}' generated {} task(s)",
                        strategy.name(),
                        tasks.len()
                    );
                    all_tasks.extend(tasks);
                }
                Err(e) => {
                    tracing::warn!("Strategy '{}' failed: {}", strategy.name(), e);
                }
            }
        }

        all_tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(all_tasks)
    }

    /// Get the names of all registered strategies.
    pub fn strategy_names(&self) -> Vec<&str> {
        self.strategies.iter().map(|s| s.name()).collect()
    }
}
