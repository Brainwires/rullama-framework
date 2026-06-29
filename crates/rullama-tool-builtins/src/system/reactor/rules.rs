//! Rules engine for matching file system events to autonomous actions.

use serde::{Deserialize, Serialize};

/// File system event types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsEventType {
    /// A new file was created.
    Created,
    /// An existing file was modified.
    Modified,
    /// A file was deleted.
    Deleted,
    /// A file was renamed.
    Renamed,
}

impl std::fmt::Display for FsEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Created => write!(f, "created"),
            Self::Modified => write!(f, "modified"),
            Self::Deleted => write!(f, "deleted"),
            Self::Renamed => write!(f, "renamed"),
        }
    }
}

/// Action to take when a reactor rule matches.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ReactorAction {
    /// Execute a command.
    ExecuteCommand {
        /// Command to run.
        cmd: String,
        /// Arguments.
        args: Vec<String>,
        /// Working directory.
        working_dir: Option<String>,
    },
    /// Investigate a log error pattern.
    InvestigateLogError {
        /// Regex pattern to search for in the changed file.
        log_pattern: String,
    },
    /// Send a notification.
    Notify {
        /// Notification message (supports `${FILE_PATH}` and `${EVENT_TYPE}` variables).
        message: String,
    },
}

/// A rule that matches file system events and triggers an action.
///
/// Rules combine path patterns, event type filters, and per-rule debouncing
/// to determine which file changes should trigger autonomous actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactorRule {
    /// Unique rule identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Directories to watch.
    pub watch_paths: Vec<String>,
    /// Glob patterns for file matching (e.g., `*.log`, `src/**/*.rs`).
    #[serde(default)]
    pub patterns: Vec<String>,
    /// Glob patterns to exclude.
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
    /// Event types to react to.
    pub event_types: Vec<FsEventType>,
    /// Per-rule debounce in milliseconds.
    pub debounce_ms: u64,
    /// Action to take when the rule matches.
    pub action: ReactorAction,
    /// Whether this rule is enabled.
    pub enabled: bool,
}

impl ReactorRule {
    /// Check if a file path matches this rule's patterns.
    pub fn matches_path(&self, path: &str) -> bool {
        // If no patterns specified, match everything
        if self.patterns.is_empty() {
            return !self.is_excluded(path);
        }

        let matches = self
            .patterns
            .iter()
            .any(|pattern| glob_match(pattern, path));

        matches && !self.is_excluded(path)
    }

    /// Check if a file path matches any exclude pattern.
    fn is_excluded(&self, path: &str) -> bool {
        self.exclude_patterns
            .iter()
            .any(|pattern| glob_match(pattern, path))
    }

    /// Check if an event type matches this rule.
    pub fn matches_event_type(&self, event_type: &FsEventType) -> bool {
        self.event_types.is_empty() || self.event_types.contains(event_type)
    }
}

/// Simple glob matching (supports `*` and `**`).
fn glob_match(pattern: &str, path: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    // Handle extension-only patterns like "*.log"
    if let Some(ext) = pattern.strip_prefix("*.") {
        return path.ends_with(&format!(".{ext}"));
    }

    // Handle directory prefix patterns like "src/**"
    if let Some(prefix) = pattern.strip_suffix("/**") {
        return path.starts_with(prefix);
    }

    // Exact match
    pattern == path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rule(patterns: Vec<&str>, exclude: Vec<&str>) -> ReactorRule {
        ReactorRule {
            id: "test".to_string(),
            name: "Test".to_string(),
            watch_paths: vec![".".to_string()],
            patterns: patterns.into_iter().map(|s| s.to_string()).collect(),
            exclude_patterns: exclude.into_iter().map(|s| s.to_string()).collect(),
            event_types: vec![FsEventType::Modified],
            debounce_ms: 1000,
            action: ReactorAction::Notify {
                message: "test".to_string(),
            },
            enabled: true,
        }
    }

    #[test]
    fn matches_extension_pattern() {
        let rule = test_rule(vec!["*.log"], vec![]);
        assert!(rule.matches_path("app.log"));
        assert!(rule.matches_path("/var/log/app.log"));
        assert!(!rule.matches_path("app.txt"));
    }

    #[test]
    fn matches_directory_pattern() {
        let rule = test_rule(vec!["src/**"], vec![]);
        assert!(rule.matches_path("src/main.rs"));
        assert!(rule.matches_path("src/lib/utils.rs"));
        assert!(!rule.matches_path("tests/main.rs"));
    }

    #[test]
    fn excludes_patterns() {
        let rule = test_rule(vec!["*.rs"], vec!["*.generated.rs"]);
        assert!(rule.matches_path("main.rs"));
        assert!(!rule.matches_path("bindings.generated.rs"));
    }

    #[test]
    fn empty_patterns_match_everything() {
        let rule = test_rule(vec![], vec![]);
        assert!(rule.matches_path("anything.txt"));
    }

    #[test]
    fn matches_event_type() {
        let rule = test_rule(vec![], vec![]);
        assert!(rule.matches_event_type(&FsEventType::Modified));
        assert!(!rule.matches_event_type(&FsEventType::Created));
    }

    #[test]
    fn empty_event_types_match_all() {
        let mut rule = test_rule(vec![], vec![]);
        rule.event_types = vec![];
        assert!(rule.matches_event_type(&FsEventType::Created));
        assert!(rule.matches_event_type(&FsEventType::Deleted));
    }
}
