//! Documentation gap detection strategy.

use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;
use std::collections::HashMap;
use walkdir::WalkDir;

use super::{ImprovementCategory, ImprovementStrategy, ImprovementTask};
use crate::config::StrategyConfig;

/// Strategy that detects missing documentation on public items.
///
/// Walks the `src/` directory, identifies `pub` items without preceding `///`
/// doc comments, and generates tasks for files with 3 or more gaps.
pub struct DocGapsStrategy;

#[async_trait]
impl ImprovementStrategy for DocGapsStrategy {
    fn name(&self) -> &str {
        "doc_gaps"
    }

    fn category(&self) -> ImprovementCategory {
        ImprovementCategory::Documentation
    }

    async fn generate_tasks(
        &self,
        repo_path: &str,
        config: &StrategyConfig,
    ) -> Result<Vec<ImprovementTask>> {
        let pub_item_pattern =
            Regex::new(r"^\s*pub\s+(fn|struct|enum|trait|type|const|static|mod)\s+(\w+)")?;

        let mut gaps_by_file: HashMap<String, Vec<(u32, String, String)>> = HashMap::new();

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

            let lines: Vec<&str> = content.lines().collect();

            for (i, line) in lines.iter().enumerate() {
                if let Some(captures) = pub_item_pattern.captures(line) {
                    let item_type = captures.get(1).map(|m| m.as_str()).unwrap_or("");
                    let item_name = captures.get(2).map(|m| m.as_str()).unwrap_or("");

                    let has_doc = if i > 0 {
                        let prev_line = lines[i - 1].trim();
                        prev_line.starts_with("///")
                            || prev_line.starts_with("//!")
                            || prev_line.starts_with("#[doc")
                            || prev_line.ends_with("*/")
                    } else {
                        false
                    };

                    let has_doc = has_doc
                        || (i > 1 && {
                            let prev2 = lines[i - 2].trim();
                            prev2.starts_with("///") || prev2.starts_with("//!")
                        });

                    if !has_doc {
                        gaps_by_file.entry(rel_path.clone()).or_default().push((
                            i as u32 + 1,
                            item_type.to_string(),
                            item_name.to_string(),
                        ));
                    }
                }
            }
        }

        let mut tasks: Vec<ImprovementTask> = gaps_by_file
            .into_iter()
            .filter(|(_, gaps)| gaps.len() >= 3)
            .take(config.max_tasks_per_strategy)
            .enumerate()
            .map(|(i, (file, gaps))| {
                let gap_count = gaps.len();
                let context = gaps
                    .iter()
                    .map(|(line, kind, name)| {
                        format!("Line {line}: pub {kind} {name} - missing doc comment")
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                ImprovementTask {
                    id: format!("docs-{i}"),
                    strategy: "doc_gaps".to_string(),
                    category: ImprovementCategory::Documentation,
                    description: format!(
                        "Add documentation comments (///) to {gap_count} public items in {file}. \
                         Read the code to understand what each item does and write concise, \
                         helpful doc comments. Focus on what the item does and when to use it."
                    ),
                    target_files: vec![file],
                    priority: 3,
                    estimated_diff_lines: (gap_count * 2) as u32,
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
    fn strategy_name_is_doc_gaps() {
        assert_eq!(DocGapsStrategy.name(), "doc_gaps");
    }

    #[test]
    fn strategy_category_is_documentation() {
        assert_eq!(
            DocGapsStrategy.category(),
            ImprovementCategory::Documentation
        );
    }
}
