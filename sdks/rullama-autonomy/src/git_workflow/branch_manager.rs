//! Branch management for autonomous fix workflows.

use serde::{Deserialize, Serialize};

/// Information about a created branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    /// Branch name.
    pub name: String,
    /// Base branch this was created from.
    pub base_branch: String,
    /// Path to the worktree, if using git worktrees.
    pub worktree_path: Option<String>,
}

/// Manages branch creation, pushing, and cleanup for autonomous fix workflows.
///
/// Branches are named using a configurable prefix, issue number, and sanitized slug.
pub struct BranchManager {
    branch_prefix: String,
}

impl BranchManager {
    /// Create a new branch manager with the given prefix.
    pub fn new(branch_prefix: String) -> Self {
        Self { branch_prefix }
    }

    /// Create a branch name from an issue number and slug.
    ///
    /// The slug is lowercased, non-alphanumeric characters replaced with hyphens,
    /// and truncated to 40 characters.
    pub fn branch_name(&self, issue_number: u64, slug: &str) -> String {
        let clean_slug: String = slug
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .take(40)
            .collect();
        let clean_slug = clean_slug.trim_matches('-').to_string();
        format!(
            "{}issue-{}-{}",
            self.branch_prefix, issue_number, clean_slug
        )
    }

    /// Create a new branch from a base branch, fetching latest from origin first.
    pub async fn create_branch(
        &self,
        repo_path: &str,
        branch_name: &str,
        base_branch: &str,
    ) -> anyhow::Result<BranchInfo> {
        // Fetch latest
        let _ = tokio::process::Command::new("git")
            .args(["fetch", "origin", base_branch])
            .current_dir(repo_path)
            .output()
            .await;

        // Create branch
        let output = tokio::process::Command::new("git")
            .args([
                "checkout",
                "-b",
                branch_name,
                &format!("origin/{base_branch}"),
            ])
            .current_dir(repo_path)
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to create branch {branch_name}: {stderr}");
        }

        Ok(BranchInfo {
            name: branch_name.to_string(),
            base_branch: base_branch.to_string(),
            worktree_path: None,
        })
    }

    /// Push a branch to the remote.
    pub async fn push_branch(&self, repo_path: &str, branch_name: &str) -> anyhow::Result<()> {
        let output = tokio::process::Command::new("git")
            .args(["push", "-u", "origin", branch_name])
            .current_dir(repo_path)
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to push branch {branch_name}: {stderr}");
        }

        Ok(())
    }

    /// Delete a branch both locally and on the remote (best-effort cleanup).
    pub async fn delete_branch(&self, repo_path: &str, branch_name: &str) -> anyhow::Result<()> {
        let _ = tokio::process::Command::new("git")
            .args(["branch", "-D", branch_name])
            .current_dir(repo_path)
            .output()
            .await;

        let _ = tokio::process::Command::new("git")
            .args(["push", "origin", "--delete", branch_name])
            .current_dir(repo_path)
            .output()
            .await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_name_basic() {
        let bm = BranchManager::new("autonomy/".to_string());
        let name = bm.branch_name(42, "fix login bug");
        assert_eq!(name, "autonomy/issue-42-fix-login-bug");
    }

    #[test]
    fn branch_name_sanitizes_special_chars() {
        let bm = BranchManager::new("fix/".to_string());
        let name = bm.branch_name(7, "Hello, World! @#$%");
        // Non-alphanumeric chars become hyphens, leading/trailing hyphens trimmed
        assert!(name.starts_with("fix/issue-7-"));
        assert!(!name.ends_with('-'));
        // Should not contain special characters
        let slug_part = name.strip_prefix("fix/issue-7-").unwrap();
        assert!(slug_part.chars().all(|c| c.is_alphanumeric() || c == '-'));
    }

    #[test]
    fn branch_name_truncates_long_slugs() {
        let bm = BranchManager::new("auto/".to_string());
        let long_slug = "a".repeat(100);
        let name = bm.branch_name(1, &long_slug);
        // Slug is truncated to 40 chars
        let slug_part = name.strip_prefix("auto/issue-1-").unwrap();
        assert!(slug_part.len() <= 40);
    }
}
