//! Refactoring strategy — detects large files and long functions.

use anyhow::Result;
use async_trait::async_trait;
use walkdir::WalkDir;

use super::{ImprovementCategory, ImprovementStrategy, ImprovementTask};
use crate::config::StrategyConfig;

/// Strategy that identifies refactoring opportunities.
///
/// Flags files over 500 lines and functions over 60 lines, generating tasks
/// to break them into smaller, more maintainable units.
pub struct RefactoringStrategy;

#[async_trait]
impl ImprovementStrategy for RefactoringStrategy {
    fn name(&self) -> &str {
        "refactoring"
    }

    fn category(&self) -> ImprovementCategory {
        ImprovementCategory::Refactoring
    }

    async fn generate_tasks(
        &self,
        repo_path: &str,
        config: &StrategyConfig,
    ) -> Result<Vec<ImprovementTask>> {
        let src_path = format!("{repo_path}/src");
        let mut smells: Vec<(String, String, u8, u32)> = Vec::new();

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

            let line_count = content.lines().count();

            if line_count > 500 {
                smells.push((
                    rel_path.clone(),
                    format!(
                        "Refactor {} ({line_count} lines) by extracting logical sections \
                         into separate modules or helper functions.",
                        rel_path
                    ),
                    2,
                    50,
                ));
            }

            // Long function detection
            let mut fn_name = String::new();
            let mut fn_start: Option<usize> = None;
            let mut brace_depth: i32 = 0;
            let mut in_function = false;

            for (i, line) in content.lines().enumerate() {
                let trimmed = line.trim();

                if !in_function
                    && (trimmed.starts_with("pub fn ")
                        || trimmed.starts_with("pub async fn ")
                        || trimmed.starts_with("fn ")
                        || trimmed.starts_with("async fn "))
                    && trimmed.contains('{')
                {
                    fn_name = trimmed
                        .replace("pub async fn ", "")
                        .replace("pub fn ", "")
                        .replace("async fn ", "")
                        .replace("fn ", "")
                        .split('(')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    fn_start = Some(i);
                    in_function = true;
                    brace_depth = 0;
                }

                if in_function {
                    for ch in trimmed.chars() {
                        match ch {
                            '{' => brace_depth += 1,
                            '}' => brace_depth -= 1,
                            _ => {}
                        }
                    }

                    if brace_depth == 0 && fn_start.is_some() {
                        let start = fn_start.expect("checked is_some above");
                        let fn_lines = i - start + 1;
                        if fn_lines > 60 {
                            smells.push((
                                rel_path.clone(),
                                format!(
                                    "Refactor function '{fn_name}' in {} ({fn_lines} lines) by \
                                     extracting logical steps into smaller helper functions.",
                                    rel_path
                                ),
                                4,
                                30,
                            ));
                        }
                        in_function = false;
                        fn_start = None;
                    }
                }
            }
        }

        let mut tasks: Vec<ImprovementTask> = smells
            .into_iter()
            .take(config.max_tasks_per_strategy)
            .enumerate()
            .map(
                |(i, (file, description, priority, estimated_diff))| ImprovementTask {
                    id: format!("refactor-{i}"),
                    strategy: "refactoring".to_string(),
                    category: ImprovementCategory::Refactoring,
                    description,
                    target_files: vec![file],
                    priority,
                    estimated_diff_lines: estimated_diff,
                    context: String::new(),
                },
            )
            .collect();

        tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_name_is_refactoring() {
        assert_eq!(RefactoringStrategy.name(), "refactoring");
    }

    #[test]
    fn strategy_category_is_refactoring() {
        assert_eq!(
            RefactoringStrategy.category(),
            ImprovementCategory::Refactoring
        );
    }
}
