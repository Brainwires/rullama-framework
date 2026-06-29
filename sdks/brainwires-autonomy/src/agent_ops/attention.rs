//! Attention mechanism — context focus and relevance scoring.
//!
//! Uses RAG integration to determine which parts of the codebase
//! are most relevant to the current task.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Relevance score for a code chunk or file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceScore {
    /// Path to the file or chunk.
    pub path: String,
    /// Relevance score (0.0 to 1.0).
    pub score: f64,
    /// Why this is relevant.
    pub reason: String,
}

/// Attention window — the subset of the codebase to focus on.
#[derive(Debug, Clone, Default)]
pub struct AttentionWindow {
    /// Files ranked by relevance.
    pub ranked_files: Vec<RelevanceScore>,
    /// Maximum number of files to include in context.
    pub max_files: usize,
    /// Maximum total tokens for the attention window.
    pub max_tokens: usize,
}

impl AttentionWindow {
    /// Create a new empty attention window with the given limits.
    pub fn new(max_files: usize, max_tokens: usize) -> Self {
        Self {
            ranked_files: Vec::new(),
            max_files,
            max_tokens,
        }
    }

    /// Get the top-N most relevant files.
    pub fn top_files(&self, n: usize) -> Vec<&RelevanceScore> {
        self.ranked_files
            .iter()
            .take(n.min(self.max_files))
            .collect()
    }

    /// Add a relevance score, maintaining sorted order (highest first).
    pub fn add(&mut self, score: RelevanceScore) {
        let pos = self
            .ranked_files
            .binary_search_by(|s| {
                s.score
                    .partial_cmp(&score.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .reverse()
            })
            .unwrap_or_else(|p| p);
        self.ranked_files.insert(pos, score);
    }
}

/// Attention mechanism that focuses agent context on the most relevant code.
///
/// Caches attention windows per task and integrates with RAG search results
/// to rank files by relevance score.
pub struct AttentionMechanism {
    /// Cache of previous attention computations.
    cache: HashMap<String, AttentionWindow>,
    /// Default attention window configuration.
    default_max_files: usize,
    default_max_tokens: usize,
}

impl AttentionMechanism {
    /// Create a new attention mechanism with default window limits.
    pub fn new(max_files: usize, max_tokens: usize) -> Self {
        Self {
            cache: HashMap::new(),
            default_max_files: max_files,
            default_max_tokens: max_tokens,
        }
    }

    /// Compute attention for a task description, returning relevant files.
    ///
    /// This is a framework-level method. Integration with RAG (brainwires-rag)
    /// is done at the application level by calling `query_codebase` and feeding
    /// results into `from_search_results`.
    pub fn from_search_results(
        &mut self,
        task_id: &str,
        results: Vec<(String, f64, String)>,
    ) -> &AttentionWindow {
        let mut window = AttentionWindow::new(self.default_max_files, self.default_max_tokens);

        for (path, score, reason) in results {
            window.add(RelevanceScore {
                path,
                score,
                reason,
            });
        }

        self.cache.insert(task_id.to_string(), window);
        self.cache.get(task_id).expect("just inserted")
    }

    /// Get a cached attention window for a task.
    pub fn get(&self, task_id: &str) -> Option<&AttentionWindow> {
        self.cache.get(task_id)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_window_new_starts_empty() {
        let w = AttentionWindow::new(10, 5000);
        assert!(w.ranked_files.is_empty());
        assert_eq!(w.max_files, 10);
        assert_eq!(w.max_tokens, 5000);
    }

    #[test]
    fn add_maintains_descending_score_order() {
        let mut w = AttentionWindow::new(10, 5000);
        w.add(RelevanceScore {
            path: "a.rs".into(),
            score: 0.5,
            reason: "".into(),
        });
        w.add(RelevanceScore {
            path: "b.rs".into(),
            score: 0.9,
            reason: "".into(),
        });
        w.add(RelevanceScore {
            path: "c.rs".into(),
            score: 0.7,
            reason: "".into(),
        });

        assert_eq!(w.ranked_files[0].path, "b.rs");
        assert_eq!(w.ranked_files[1].path, "c.rs");
        assert_eq!(w.ranked_files[2].path, "a.rs");
    }

    #[test]
    fn top_files_returns_correct_count() {
        let mut w = AttentionWindow::new(5, 5000);
        for i in 0..10 {
            w.add(RelevanceScore {
                path: format!("file{i}.rs"),
                score: i as f64 / 10.0,
                reason: "".into(),
            });
        }
        // top_files(n) returns min(n, max_files) items
        assert_eq!(w.top_files(3).len(), 3);
        assert_eq!(w.top_files(10).len(), 5); // capped by max_files=5
    }

    #[test]
    fn attention_mechanism_from_search_results_and_get() {
        let mut mech = AttentionMechanism::new(10, 5000);
        let results = vec![
            ("src/main.rs".to_string(), 0.9, "entry point".to_string()),
            ("src/lib.rs".to_string(), 0.7, "library".to_string()),
        ];
        let window = mech.from_search_results("task-1", results);
        assert_eq!(window.ranked_files.len(), 2);
        assert_eq!(window.ranked_files[0].path, "src/main.rs");

        // get returns the cached window
        let cached = mech.get("task-1").expect("should be cached");
        assert_eq!(cached.ranked_files.len(), 2);

        assert!(mech.get("nonexistent").is_none());
    }
}
