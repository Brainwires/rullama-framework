//! Test coverage detection strategy.

use anyhow::Result;
use async_trait::async_trait;
use walkdir::WalkDir;

use super::{ImprovementCategory, ImprovementStrategy, ImprovementTask};
use crate::config::StrategyConfig;

/// Strategy that identifies source files without any test coverage.
///
/// Scans for Rust files in `src/` that lack `#[cfg(test)]` or `#[test]` blocks
/// and have public functions, then generates tasks to add unit tests.
pub struct TestCoverageStrategy;

#[async_trait]
impl ImprovementStrategy for TestCoverageStrategy {
    fn name(&self) -> &str {
        "test_coverage"
    }

    fn category(&self) -> ImprovementCategory {
        ImprovementCategory::Testing
    }

    async fn generate_tasks(
        &self,
        repo_path: &str,
        config: &StrategyConfig,
    ) -> Result<Vec<ImprovementTask>> {
        let src_path = format!("{repo_path}/src");
        let mut uncovered: Vec<(String, Vec<String>)> = Vec::new();

        for entry in WalkDir::new(&src_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
        {
            let path = entry.path();
            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let rel_path = path
                .strip_prefix(repo_path)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            if rel_path.contains("/tests/") || rel_path.ends_with("mod.rs") {
                continue;
            }

            let has_tests = content.contains("#[cfg(test)]") || content.contains("#[test]");

            if !has_tests {
                let pub_fns: Vec<String> = content
                    .lines()
                    .filter(|line| {
                        let trimmed = line.trim();
                        trimmed.starts_with("pub fn ") || trimmed.starts_with("pub async fn ")
                    })
                    .map(|line| {
                        line.trim()
                            .split('(')
                            .next()
                            .unwrap_or("")
                            .replace("pub fn ", "")
                            .replace("pub async fn ", "")
                            .trim()
                            .to_string()
                    })
                    .filter(|name| !name.is_empty())
                    .collect();

                if !pub_fns.is_empty() {
                    uncovered.push((rel_path, pub_fns));
                }
            }
        }

        let mut tasks: Vec<ImprovementTask> = uncovered
            .into_iter()
            .take(config.max_tasks_per_strategy)
            .enumerate()
            .map(|(i, (file, fns))| {
                let fn_count = fns.len();
                let context = format!(
                    "Public functions without tests:\n{}",
                    fns.iter()
                        .map(|f| format!("  - {f}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                );

                ImprovementTask {
                    id: format!("test-{i}"),
                    strategy: "test_coverage".to_string(),
                    category: ImprovementCategory::Testing,
                    description: format!(
                        "Add unit tests for {fn_count} public function(s) in {file}. \
                         Add a #[cfg(test)] module at the bottom of the file with \
                         tests for the key public functions. Focus on testing the \
                         happy path and one or two edge cases per function."
                    ),
                    target_files: vec![file],
                    priority: 5,
                    estimated_diff_lines: (fn_count * 15) as u32,
                    context,
                }
            })
            .collect();

        tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(tasks)
    }
}
