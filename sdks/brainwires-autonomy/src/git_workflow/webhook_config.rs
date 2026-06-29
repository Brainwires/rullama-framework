//! Rich per-repository webhook configuration with variable interpolation.

use std::collections::HashMap;

use crate::config::{CommandConfig, WebhookRepoConfig};

/// Interpolation context for webhook command templates.
///
/// Supports `${VARIABLE}` syntax in command args and working directories.
/// Pre-built constructors provide standard variables for issue and push events.
pub struct InterpolationContext {
    /// Variables available for interpolation.
    pub vars: HashMap<String, String>,
}

impl InterpolationContext {
    /// Create a context for an issue event.
    pub fn for_issue(repo_name: &str, issue_number: u64, issue_title: &str) -> Self {
        let mut vars = HashMap::new();
        vars.insert("REPO_NAME".to_string(), repo_name.to_string());
        vars.insert("ISSUE_NUMBER".to_string(), issue_number.to_string());
        vars.insert("ISSUE_TITLE".to_string(), issue_title.to_string());
        Self { vars }
    }

    /// Create a context for a push event.
    pub fn for_push(repo_name: &str, branch: &str, commit_sha: &str) -> Self {
        let mut vars = HashMap::new();
        vars.insert("REPO_NAME".to_string(), repo_name.to_string());
        vars.insert("BRANCH_NAME".to_string(), branch.to_string());
        vars.insert("COMMIT_SHA".to_string(), commit_sha.to_string());

        // Handle tag refs
        if branch.starts_with("refs/tags/") {
            let tag = branch.strip_prefix("refs/tags/").unwrap_or(branch);
            vars.insert("TAG_NAME".to_string(), tag.to_string());
            let version = tag.strip_prefix('v').unwrap_or(tag);
            vars.insert("VERSION".to_string(), version.to_string());
        }

        Self { vars }
    }

    /// Add a custom variable.
    pub fn with_var(mut self, key: &str, value: &str) -> Self {
        self.vars.insert(key.to_string(), value.to_string());
        self
    }

    /// Interpolate variables in a string (e.g., `${REPO_NAME}` -> actual value).
    pub fn interpolate(&self, template: &str) -> String {
        let mut result = template.to_string();
        for (key, value) in &self.vars {
            result = result.replace(&format!("${{{key}}}"), value);
        }
        result
    }

    /// Interpolate variables in a command config's args and working_dir.
    pub fn interpolate_command(&self, cmd: &CommandConfig) -> CommandConfig {
        CommandConfig {
            cmd: self.interpolate(&cmd.cmd),
            args: cmd.args.iter().map(|a| self.interpolate(a)).collect(),
            working_dir: cmd.working_dir.as_ref().map(|w| self.interpolate(w)),
        }
    }
}

/// Check whether a repo config should handle an event based on its type and labels.
///
/// Returns `false` if the event type is not in the config's event list or if
/// none of the event's labels match the config's label filter.
pub fn should_handle_event(
    config: &WebhookRepoConfig,
    event_type: &str,
    labels: &[String],
) -> bool {
    // Check event type filter
    if !config.events.is_empty() && !config.events.iter().any(|e| e == event_type) {
        return false;
    }

    // Check label filter
    if !config.labels_filter.is_empty() {
        let has_matching_label = labels
            .iter()
            .any(|l| config.labels_filter.iter().any(|f| f == l));
        if !has_matching_label {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_replaces_variables() {
        let ctx = InterpolationContext::for_issue("my-repo", 42, "Fix bug");
        assert_eq!(
            ctx.interpolate("Fixing ${REPO_NAME} issue #${ISSUE_NUMBER}"),
            "Fixing my-repo issue #42"
        );
    }

    #[test]
    fn interpolation_for_push_with_tag() {
        let ctx = InterpolationContext::for_push("my-repo", "refs/tags/v1.2.3", "abc123");
        assert_eq!(ctx.vars["TAG_NAME"], "v1.2.3");
        assert_eq!(ctx.vars["VERSION"], "1.2.3");
    }

    #[test]
    fn interpolation_for_push_without_tag() {
        let ctx = InterpolationContext::for_push("my-repo", "main", "abc123");
        assert!(!ctx.vars.contains_key("TAG_NAME"));
    }

    #[test]
    fn should_handle_event_with_empty_filters() {
        let config = WebhookRepoConfig::default();
        // Default events = ["issues"]
        assert!(should_handle_event(&config, "issues", &[]));
        assert!(!should_handle_event(&config, "push", &[]));
    }

    #[test]
    fn should_handle_event_respects_label_filter() {
        let config = WebhookRepoConfig {
            labels_filter: vec!["auto-fix".to_string()],
            ..Default::default()
        };
        assert!(!should_handle_event(&config, "issues", &[]));
        assert!(!should_handle_event(
            &config,
            "issues",
            &["bug".to_string()]
        ));
        assert!(should_handle_event(
            &config,
            "issues",
            &["auto-fix".to_string()]
        ));
    }
}
