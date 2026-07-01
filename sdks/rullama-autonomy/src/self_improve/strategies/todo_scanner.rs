//! TODO/FIXME scanner strategy.

use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;
use std::collections::HashMap;
use walkdir::WalkDir;

use super::{ImprovementCategory, ImprovementStrategy, ImprovementTask};
use crate::config::StrategyConfig;

/// Strategy that scans for TODO, FIXME, HACK, and XXX comments and generates
/// tasks to resolve them. FIXME-containing files receive higher priority.
pub struct TodoScannerStrategy;

#[async_trait]
impl ImprovementStrategy for TodoScannerStrategy {
    fn name(&self) -> &str {
        "todo_scanner"
    }

    fn category(&self) -> ImprovementCategory {
        ImprovementCategory::Refactoring
    }

    async fn generate_tasks(
        &self,
        repo_path: &str,
        config: &StrategyConfig,
    ) -> Result<Vec<ImprovementTask>> {
        let pattern = Regex::new(r"(?i)(TODO|FIXME|HACK|XXX)\s*:?\s*(.*)")?;
        let mut todos_by_file: HashMap<String, Vec<(u32, String, String)>> = HashMap::new();

        let src_path = format!("{repo_path}/src");
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

            for (line_num, line) in content.lines().enumerate() {
                if let Some(captures) = pattern.captures(line) {
                    let tag = captures.get(1).map(|m| m.as_str()).unwrap_or("TODO");
                    let text = captures
                        .get(2)
                        .map(|m| m.as_str().trim())
                        .unwrap_or("")
                        .to_string();
                    todos_by_file.entry(rel_path.clone()).or_default().push((
                        line_num as u32 + 1,
                        tag.to_uppercase(),
                        text,
                    ));
                }
            }
        }

        let mut tasks: Vec<ImprovementTask> = todos_by_file
            .into_iter()
            .take(config.max_tasks_per_strategy)
            .enumerate()
            .map(|(i, (file, todos))| {
                let todo_count = todos.len();
                let context = todos
                    .iter()
                    .map(|(line, tag, text)| format!("Line {line}: [{tag}] {text}"))
                    .collect::<Vec<_>>()
                    .join("\n");

                let has_fixme = todos.iter().any(|(_, tag, _)| tag == "FIXME");
                let priority = if has_fixme { 7 } else { 4 };

                ImprovementTask {
                    id: format!("todo-{i}"),
                    strategy: "todo_scanner".to_string(),
                    category: ImprovementCategory::Refactoring,
                    description: format!(
                        "Address {todo_count} TODO/FIXME comment(s) in {file}. \
                         Implement the described functionality or remove the \
                         TODO if it's already been addressed."
                    ),
                    target_files: vec![file],
                    priority,
                    estimated_diff_lines: (todo_count * 10) as u32,
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
    fn strategy_name_is_todo_scanner() {
        assert_eq!(TodoScannerStrategy.name(), "todo_scanner");
    }

    #[test]
    fn strategy_category_is_refactoring() {
        assert_eq!(
            TodoScannerStrategy.category(),
            ImprovementCategory::Refactoring
        );
    }
}
