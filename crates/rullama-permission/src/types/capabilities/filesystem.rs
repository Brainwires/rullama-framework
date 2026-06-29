use serde::{Deserialize, Serialize};

use crate::types::path_pattern::PathPattern;

/// File system capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemCapabilities {
    /// Allowed read paths (glob patterns)
    #[serde(default = "default_read_paths")]
    pub read_paths: Vec<PathPattern>,

    /// Allowed write paths (glob patterns)
    #[serde(default)]
    pub write_paths: Vec<PathPattern>,

    /// Denied paths (override allows)
    #[serde(default = "default_denied_paths")]
    pub denied_paths: Vec<PathPattern>,

    /// Can follow symlinks outside allowed paths
    #[serde(default = "default_true")]
    pub follow_symlinks: bool,

    /// Can access hidden files (dotfiles)
    #[serde(default = "default_true")]
    pub access_hidden: bool,

    /// Maximum file size for write operations (bytes)
    #[serde(default)]
    pub max_write_size: Option<u64>,

    /// Can delete files
    #[serde(default)]
    pub can_delete: bool,

    /// Can create directories
    #[serde(default = "default_true")]
    pub can_create_dirs: bool,
}

fn default_read_paths() -> Vec<PathPattern> {
    vec![PathPattern::new("**/*")]
}

fn default_denied_paths() -> Vec<PathPattern> {
    vec![
        PathPattern::new("**/.env*"),
        PathPattern::new("**/*credentials*"),
        PathPattern::new("**/*secret*"),
    ]
}

fn default_true() -> bool {
    true
}

impl Default for FilesystemCapabilities {
    fn default() -> Self {
        Self {
            read_paths: default_read_paths(),
            write_paths: Vec::new(),
            denied_paths: default_denied_paths(),
            follow_symlinks: true,
            access_hidden: true,
            max_write_size: None,
            can_delete: false,
            can_create_dirs: true,
        }
    }
}

impl FilesystemCapabilities {
    /// Create full access filesystem capabilities
    pub fn full() -> Self {
        Self {
            read_paths: vec![PathPattern::new("**/*")],
            write_paths: vec![PathPattern::new("**/*")],
            denied_paths: Vec::new(),
            follow_symlinks: true,
            access_hidden: true,
            max_write_size: None,
            can_delete: true,
            can_create_dirs: true,
        }
    }
}
