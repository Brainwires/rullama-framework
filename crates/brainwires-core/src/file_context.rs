//! File Context Manager
//!
//! Manages file content for context injection with smart chunking for large files.
//! Prevents re-injection of files already in context and retrieves relevant
//! portions of large files based on query context.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

/// Maximum characters before chunking a file
const MAX_DIRECT_FILE_CHARS: usize = 50_000;
/// Chunk size for large files (in characters)
const LARGE_FILE_CHUNK_SIZE: usize = 10_000;
/// Maximum number of chunks to return from a large file
const MAX_FILE_CHUNKS: usize = 5;

/// Content returned from file context manager
#[derive(Debug, Clone)]
pub enum FileContent {
    /// Small file - full content returned
    Full(String),
    /// Large file - only relevant chunks returned
    Chunked {
        /// File path
        path: String,
        /// Total file size in characters
        total_size: usize,
        /// Retrieved chunks with context
        chunks: Vec<FileChunk>,
        /// Whether there's more content available
        has_more: bool,
    },
    /// File already in context - just return a reference
    AlreadyInContext(String),
}

/// A chunk of file content with context
#[derive(Debug, Clone)]
pub struct FileChunk {
    /// Chunk content
    pub content: String,
    /// Starting line number (1-indexed)
    pub line_start: usize,
    /// Ending line number (1-indexed)
    pub line_end: usize,
    /// Relevance score (0.0 to 1.0)
    pub relevance_score: f32,
}

/// Manages file content for context injection
pub struct FileContextManager {
    /// Files already in current context (to avoid re-injection)
    context_files: HashSet<String>,
    /// Cache of indexed file chunks (path -> chunks)
    file_chunks: HashMap<String, Vec<FileChunk>>,
}

impl Default for FileContextManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FileContextManager {
    /// Create a new file context manager
    pub fn new() -> Self {
        Self {
            context_files: HashSet::new(),
            file_chunks: HashMap::new(),
        }
    }

    /// Compute SHA256 hash of content
    pub fn compute_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Check if a file is already in the current context
    pub fn is_in_context(&self, path: &str) -> bool {
        self.context_files.contains(path)
    }

    /// Mark a file as being in the current context
    pub fn mark_in_context(&mut self, path: &str) {
        self.context_files.insert(path.to_string());
    }

    /// Clear the context tracking (for new conversation turns)
    pub fn clear_context(&mut self) {
        self.context_files.clear();
    }

    /// Get the number of files currently in context
    pub fn context_file_count(&self) -> usize {
        self.context_files.len()
    }

    /// Get file content with smart routing based on size
    ///
    /// # Arguments
    /// * `path` - Path to the file
    /// * `query_context` - Optional query to use for finding relevant chunks
    ///
    /// # Returns
    /// * `FileContent::Full` for small files
    /// * `FileContent::Chunked` for large files with relevant portions
    /// * `FileContent::AlreadyInContext` if file was previously loaded
    pub async fn get_file_content(
        &mut self,
        path: &str,
        query_context: Option<&str>,
    ) -> Result<FileContent> {
        // Check if already in context
        if self.is_in_context(path) {
            return Ok(FileContent::AlreadyInContext(path.to_string()));
        }

        // Read the file
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read file: {}", path))?;

        // Small file: return full content
        if content.len() <= MAX_DIRECT_FILE_CHARS {
            self.mark_in_context(path);
            return Ok(FileContent::Full(content));
        }

        // Large file: get relevant chunks
        let chunks = self.get_relevant_chunks(path, &content, query_context)?;

        self.mark_in_context(path);

        Ok(FileContent::Chunked {
            path: path.to_string(),
            total_size: content.len(),
            chunks,
            has_more: content.len() > MAX_DIRECT_FILE_CHARS,
        })
    }

    /// Get relevant chunks from a large file
    fn get_relevant_chunks(
        &mut self,
        path: &str,
        content: &str,
        query_context: Option<&str>,
    ) -> Result<Vec<FileChunk>> {
        // Build all chunks from file
        let all_chunks = self.build_file_chunks(content);

        // Cache chunks for future reference
        self.file_chunks
            .insert(path.to_string(), all_chunks.clone());

        // If we have a query, try to find relevant chunks
        if let Some(query) = query_context {
            let relevant = self.find_relevant_chunks(&all_chunks, query);
            if !relevant.is_empty() {
                return Ok(relevant);
            }
        }

        // Fall back to first N chunks
        Ok(all_chunks.into_iter().take(MAX_FILE_CHUNKS).collect())
    }

    /// Build chunks from file content
    fn build_file_chunks(&self, content: &str) -> Vec<FileChunk> {
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();
        let mut current_line = 0;

        while current_line < lines.len() {
            let mut chunk_content = String::new();
            let start_line = current_line + 1; // 1-indexed

            // Build chunk up to target size
            while current_line < lines.len() && chunk_content.len() < LARGE_FILE_CHUNK_SIZE {
                if !chunk_content.is_empty() {
                    chunk_content.push('\n');
                }
                chunk_content.push_str(lines[current_line]);
                current_line += 1;
            }

            if !chunk_content.is_empty() {
                chunks.push(FileChunk {
                    content: chunk_content,
                    line_start: start_line,
                    line_end: current_line,
                    relevance_score: 1.0,
                });
            }
        }

        chunks
    }

    /// Find chunks relevant to a query using simple keyword matching
    fn find_relevant_chunks(&self, chunks: &[FileChunk], query: &str) -> Vec<FileChunk> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored_chunks: Vec<(FileChunk, f32)> = chunks
            .iter()
            .filter_map(|chunk| {
                let content_lower = chunk.content.to_lowercase();

                // Count matching words
                let matching_words = query_words
                    .iter()
                    .filter(|word| content_lower.contains(*word))
                    .count();

                if matching_words > 0 {
                    let score = matching_words as f32 / query_words.len() as f32;
                    Some((
                        FileChunk {
                            content: chunk.content.clone(),
                            line_start: chunk.line_start,
                            line_end: chunk.line_end,
                            relevance_score: score,
                        },
                        score,
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N chunks
        scored_chunks
            .into_iter()
            .take(MAX_FILE_CHUNKS)
            .map(|(chunk, _)| chunk)
            .collect()
    }

    /// Get specific lines from a file
    pub async fn get_file_lines(
        &mut self,
        path: &str,
        start_line: usize,
        end_line: usize,
    ) -> Result<FileContent> {
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read file: {}", path))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = (start_line.saturating_sub(1)).min(total_lines);
        let end = end_line.min(total_lines);

        if start >= end {
            return Ok(FileContent::Full(String::new()));
        }

        let selected_content: String = lines[start..end].join("\n");

        self.mark_in_context(path);

        if selected_content.len() <= MAX_DIRECT_FILE_CHARS {
            Ok(FileContent::Full(selected_content))
        } else {
            Ok(FileContent::Chunked {
                path: path.to_string(),
                total_size: content.len(),
                chunks: vec![FileChunk {
                    content: selected_content,
                    line_start: start + 1,
                    line_end: end,
                    relevance_score: 1.0,
                }],
                has_more: true,
            })
        }
    }

    /// Format chunked content for display in context
    pub fn format_content(file_content: &FileContent) -> String {
        match file_content {
            FileContent::Full(content) => content.clone(),
            FileContent::AlreadyInContext(path) => {
                format!("[File {} is already shown above]", path)
            }
            FileContent::Chunked {
                path,
                total_size,
                chunks,
                has_more,
            } => {
                let mut result = format!(
                    "[File: {} | Size: {} chars | Showing {} relevant sections]\n\n",
                    path,
                    total_size,
                    chunks.len()
                );

                for chunk in chunks {
                    result.push_str(&format!(
                        "--- Lines {}-{} (relevance: {:.2}) ---\n{}\n\n",
                        chunk.line_start, chunk.line_end, chunk.relevance_score, chunk.content
                    ));
                }

                if *has_more {
                    result.push_str(
                        "[... more content available, ask for specific sections or line numbers ...]\n",
                    );
                }

                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_chunk_creation() {
        let chunk = FileChunk {
            content: "fn main() {}".to_string(),
            line_start: 1,
            line_end: 1,
            relevance_score: 0.95,
        };

        assert_eq!(chunk.line_start, 1);
        assert_eq!(chunk.line_end, 1);
        assert!((chunk.relevance_score - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_format_full_content() {
        let content = FileContent::Full("hello world".to_string());
        let formatted = FileContextManager::format_content(&content);
        assert_eq!(formatted, "hello world");
    }

    #[test]
    fn test_format_already_in_context() {
        let content = FileContent::AlreadyInContext("/path/to/file.rs".to_string());
        let formatted = FileContextManager::format_content(&content);
        assert!(formatted.contains("already shown above"));
        assert!(formatted.contains("/path/to/file.rs"));
    }

    #[test]
    fn test_format_chunked_content() {
        let content = FileContent::Chunked {
            path: "/path/to/file.rs".to_string(),
            total_size: 50000,
            chunks: vec![
                FileChunk {
                    content: "fn main() {}".to_string(),
                    line_start: 1,
                    line_end: 1,
                    relevance_score: 0.95,
                },
                FileChunk {
                    content: "fn helper() {}".to_string(),
                    line_start: 10,
                    line_end: 10,
                    relevance_score: 0.85,
                },
            ],
            has_more: true,
        };

        let formatted = FileContextManager::format_content(&content);

        assert!(formatted.contains("/path/to/file.rs"));
        assert!(formatted.contains("50000 chars"));
        assert!(formatted.contains("2 relevant sections"));
        assert!(formatted.contains("fn main()"));
        assert!(formatted.contains("fn helper()"));
        assert!(formatted.contains("more content available"));
    }

    #[test]
    fn test_compute_hash() {
        let hash1 = FileContextManager::compute_hash("hello world");
        let hash2 = FileContextManager::compute_hash("hello world");
        let hash3 = FileContextManager::compute_hash("different content");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // SHA256 hex length
    }

    #[test]
    fn test_context_tracking() {
        let mut manager = FileContextManager::new();

        assert!(!manager.is_in_context("/some/file.rs"));
        assert_eq!(manager.context_file_count(), 0);

        manager.mark_in_context("/some/file.rs");
        assert!(manager.is_in_context("/some/file.rs"));
        assert_eq!(manager.context_file_count(), 1);

        manager.clear_context();
        assert!(!manager.is_in_context("/some/file.rs"));
        assert_eq!(manager.context_file_count(), 0);
    }

    #[test]
    fn test_build_file_chunks() {
        let manager = FileContextManager::new();
        let content = "line 1\nline 2\nline 3\nline 4\nline 5";

        let chunks = manager.build_file_chunks(content);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].line_start, 1);
    }

    #[test]
    fn test_find_relevant_chunks() {
        let manager = FileContextManager::new();
        let chunks = vec![
            FileChunk {
                content: "This is about authentication and login".to_string(),
                line_start: 1,
                line_end: 1,
                relevance_score: 1.0,
            },
            FileChunk {
                content: "This is about database queries".to_string(),
                line_start: 2,
                line_end: 2,
                relevance_score: 1.0,
            },
            FileChunk {
                content: "This handles user login flow".to_string(),
                line_start: 3,
                line_end: 3,
                relevance_score: 1.0,
            },
        ];

        let relevant = manager.find_relevant_chunks(&chunks, "login authentication");

        assert!(!relevant.is_empty());
        // The chunks containing "login" or "authentication" should be ranked higher
        assert!(
            relevant[0].content.contains("login") || relevant[0].content.contains("authentication")
        );
    }
}
