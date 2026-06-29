use serde::{Deserialize, Serialize};

/// Path pattern for glob matching
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PathPattern {
    pattern: String,
}

impl PathPattern {
    /// Create a new path pattern
    pub fn new(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
        }
    }

    /// Create a glob pattern
    pub fn glob(pattern: &str) -> Self {
        Self::new(pattern)
    }

    /// Check if a path matches this pattern
    #[cfg(feature = "native")]
    pub fn matches(&self, path: &str) -> bool {
        // Use glob matching
        if let Ok(pattern) = glob::Pattern::new(&self.pattern) {
            pattern.matches(path) || pattern.matches_path(std::path::Path::new(path))
        } else {
            // Fall back to simple string matching if pattern is invalid
            path.contains(&self.pattern)
        }
    }

    /// Check if a path matches this pattern (simple string matching for WASM)
    #[cfg(not(feature = "native"))]
    pub fn matches(&self, path: &str) -> bool {
        path.contains(&self.pattern)
    }

    /// Get the pattern string
    pub fn pattern(&self) -> &str {
        &self.pattern
    }
}
