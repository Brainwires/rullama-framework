//! Clippy lint strategy — runs `cargo clippy` and generates fix tasks.

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;

use super::{ImprovementCategory, ImprovementStrategy, ImprovementTask};
use crate::config::StrategyConfig;

/// Strategy that runs `cargo clippy` and generates fix tasks from warnings.
///
/// Parses clippy JSON output to group warnings by file and produce tasks
/// that instruct the AI to fix the actual code rather than suppress lints.
pub struct ClippyStrategy;

#[derive(Deserialize)]
struct ClippyDiagnostic {
    reason: Option<String>,
    message: Option<ClippyMessage>,
}

#[derive(Deserialize)]
struct ClippyMessage {
    message: String,
    level: String,
    spans: Vec<ClippySpan>,
    code: Option<ClippyCode>,
}

#[derive(Deserialize)]
struct ClippySpan {
    file_name: String,
    line_start: u32,
    line_end: u32,
}

#[derive(Deserialize)]
struct ClippyCode {
    code: String,
}

#[async_trait]
impl ImprovementStrategy for ClippyStrategy {
    fn name(&self) -> &str {
        "clippy"
    }

    fn category(&self) -> ImprovementCategory {
        ImprovementCategory::Linting
    }

    async fn generate_tasks(
        &self,
        repo_path: &str,
        config: &StrategyConfig,
    ) -> Result<Vec<ImprovementTask>> {
        let output = tokio::process::Command::new("cargo")
            .args(["clippy", "--message-format=json", "--", "-W", "clippy::all"])
            .current_dir(repo_path)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut warnings_by_file: HashMap<String, Vec<String>> = HashMap::new();

        for line in stdout.lines() {
            let diag: ClippyDiagnostic = match serde_json::from_str(line) {
                Ok(d) => d,
                Err(_) => continue,
            };

            if diag.reason.as_deref() != Some("compiler-message") {
                continue;
            }

            let msg = match diag.message {
                Some(m) if m.level == "warning" => m,
                _ => continue,
            };

            if msg.spans.is_empty() {
                continue;
            }

            let file = &msg.spans[0].file_name;
            if !file.starts_with("src/") {
                continue;
            }

            let code_str = msg
                .code
                .as_ref()
                .map(|c| format!(" [{}]", c.code))
                .unwrap_or_default();

            let warning_text = format!(
                "Line {}-{}: {}{}",
                msg.spans[0].line_start, msg.spans[0].line_end, msg.message, code_str
            );

            warnings_by_file
                .entry(file.to_string())
                .or_default()
                .push(warning_text);
        }

        let mut tasks: Vec<ImprovementTask> = warnings_by_file
            .into_iter()
            .take(config.max_tasks_per_strategy)
            .enumerate()
            .map(|(i, (file, warnings))| {
                let warning_count = warnings.len();
                let context = warnings.join("\n");
                ImprovementTask {
                    id: format!("clippy-{i}"),
                    strategy: "clippy".to_string(),
                    category: ImprovementCategory::Linting,
                    description: format!(
                        "Fix {warning_count} clippy warning(s) in {file}. \
                         Apply the suggested fixes while maintaining correctness. \
                         Do NOT add #[allow(...)] attributes - fix the actual code."
                    ),
                    target_files: vec![file],
                    priority: 6,
                    estimated_diff_lines: (warning_count * 3) as u32,
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
    fn strategy_name_is_clippy() {
        assert_eq!(ClippyStrategy.name(), "clippy");
    }

    #[test]
    fn strategy_category_is_linting() {
        assert_eq!(ClippyStrategy.category(), ImprovementCategory::Linting);
    }
}
