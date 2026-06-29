//! Dead code detection strategy.

use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;
use std::collections::HashMap;

use super::{ImprovementCategory, ImprovementStrategy, ImprovementTask};
use crate::config::StrategyConfig;

/// Strategy that detects dead and unreachable code by parsing `cargo build` warnings.
///
/// Identifies unused imports, variables, functions, and other items, then generates
/// tasks to remove or properly annotate the dead code.
pub struct DeadCodeStrategy;

#[async_trait]
impl ImprovementStrategy for DeadCodeStrategy {
    fn name(&self) -> &str {
        "dead_code"
    }

    fn category(&self) -> ImprovementCategory {
        ImprovementCategory::DeadCode
    }

    async fn generate_tasks(
        &self,
        repo_path: &str,
        config: &StrategyConfig,
    ) -> Result<Vec<ImprovementTask>> {
        let output = tokio::process::Command::new("cargo")
            .args(["build", "--message-format=json"])
            .current_dir(repo_path)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let unused_pattern = Regex::new(
            r"warning: unused (import|variable|function|method|field|variant|struct|enum|const|type|trait).*`([^`]+)`",
        )?;

        let mut unused_by_file: HashMap<String, Vec<String>> = HashMap::new();

        for line in stdout.lines() {
            if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
                if msg.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
                    continue;
                }
                if let Some(message) = msg.get("message") {
                    let level = message.get("level").and_then(|l| l.as_str()).unwrap_or("");
                    if level != "warning" {
                        continue;
                    }
                    let msg_text = message
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("");

                    let is_unused = msg_text.contains("unused")
                        || msg_text.contains("never used")
                        || msg_text.contains("never read")
                        || msg_text.contains("never constructed");

                    if !is_unused {
                        continue;
                    }

                    if let Some(spans) = message.get("spans").and_then(|s| s.as_array())
                        && let Some(span) = spans.first()
                    {
                        let file = span.get("file_name").and_then(|f| f.as_str()).unwrap_or("");
                        let line_start =
                            span.get("line_start").and_then(|l| l.as_u64()).unwrap_or(0);

                        if file.starts_with("src/") {
                            unused_by_file
                                .entry(file.to_string())
                                .or_default()
                                .push(format!("Line {line_start}: {msg_text}"));
                        }
                    }
                }
            }
        }

        // Parse plain stderr as fallback
        for line in stderr.lines() {
            if let Some(captures) = unused_pattern.captures(line) {
                let _kind = captures.get(1).map(|m| m.as_str()).unwrap_or("item");
                let _name = captures.get(2).map(|m| m.as_str()).unwrap_or("unknown");
            }
        }

        let mut tasks: Vec<ImprovementTask> = unused_by_file
            .into_iter()
            .take(config.max_tasks_per_strategy)
            .enumerate()
            .map(|(i, (file, warnings))| {
                let count = warnings.len();
                let context = warnings.join("\n");

                ImprovementTask {
                    id: format!("deadcode-{i}"),
                    strategy: "dead_code".to_string(),
                    category: ImprovementCategory::DeadCode,
                    description: format!(
                        "Remove {count} unused item(s) in {file}. \
                         Remove the dead code or mark it with appropriate \
                         #[allow(dead_code)] if it's intentionally kept for future use. \
                         Prefer removal over suppression."
                    ),
                    target_files: vec![file],
                    priority: 5,
                    estimated_diff_lines: (count * 5) as u32,
                    context,
                }
            })
            .collect();

        tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_name_is_dead_code() {
        assert_eq!(DeadCodeStrategy.name(), "dead_code");
    }

    #[test]
    fn strategy_category_is_dead_code() {
        assert_eq!(DeadCodeStrategy.category(), ImprovementCategory::DeadCode);
    }
}
