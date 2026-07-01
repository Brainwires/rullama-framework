//! File information structure for indexed files

use std::path::PathBuf;

/// Information about a discovered file
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// Absolute file path.
    pub path: PathBuf,
    /// Path relative to the index root.
    pub relative_path: String,
    /// Root path of the indexed directory.
    pub root_path: String,
    /// Project name, if specified.
    pub project: Option<String>,
    /// File extension.
    pub extension: Option<String>,
    /// Detected programming language.
    pub language: Option<String>,
    /// File content as UTF-8 string.
    pub content: String,
    /// Content hash for change detection.
    pub hash: String,
}
