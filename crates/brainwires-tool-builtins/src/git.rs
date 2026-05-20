use anyhow::Result;
use git2::Repository;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

/// Git operations tool implementation
pub struct GitTool;

impl GitTool {
    /// Get all git tool definitions
    pub fn get_tools() -> Vec<Tool> {
        vec![
            Self::git_status_tool(),
            Self::git_diff_tool(),
            Self::git_log_tool(),
            Self::git_stage_tool(),
            Self::git_unstage_tool(),
            Self::git_commit_tool(),
            Self::git_push_tool(),
            Self::git_pull_tool(),
            Self::git_fetch_tool(),
            Self::git_discard_tool(),
            Self::git_branch_tool(),
        ]
    }

    fn git_status_tool() -> Tool {
        Tool {
            name: "git_status".to_string(),
            description: "Get git repository status".to_string(),
            input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn git_diff_tool() -> Tool {
        Tool {
            name: "git_diff".to_string(),
            description: "Get git diff of changes".to_string(),
            input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn git_log_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "limit".to_string(),
            json!({"type": "number", "description": "Number of commits", "default": 10}),
        );
        Tool {
            name: "git_log".to_string(),
            description: "Get git commit history".to_string(),
            input_schema: ToolInputSchema::object(properties, vec![]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn git_stage_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert("files".to_string(), json!({"type": "array", "items": {"type": "string"}, "description": "Files to stage. Use '.' for all."}));
        Tool {
            name: "git_stage".to_string(),
            description: "Stage files for commit.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["files".to_string()]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn git_unstage_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert("files".to_string(), json!({"type": "array", "items": {"type": "string"}, "description": "Files to unstage."}));
        Tool {
            name: "git_unstage".to_string(),
            description: "Unstage files from the staging area.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["files".to_string()]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn git_commit_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "message".to_string(),
            json!({"type": "string", "description": "Commit message"}),
        );
        properties.insert("all".to_string(), json!({"type": "boolean", "description": "Stage all modified files before committing", "default": false}));
        Tool {
            name: "git_commit".to_string(),
            description: "Create a git commit with staged changes.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["message".to_string()]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn git_push_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert("remote".to_string(), json!({"type": "string", "description": "Remote name (default: origin)", "default": "origin"}));
        properties.insert(
            "branch".to_string(),
            json!({"type": "string", "description": "Branch to push"}),
        );
        properties.insert("set_upstream".to_string(), json!({"type": "boolean", "description": "Set upstream tracking (-u)", "default": false}));
        Tool {
            name: "git_push".to_string(),
            description: "Push commits to a remote repository.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec![]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn git_pull_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert("remote".to_string(), json!({"type": "string", "description": "Remote name (default: origin)", "default": "origin"}));
        properties.insert(
            "branch".to_string(),
            json!({"type": "string", "description": "Branch to pull"}),
        );
        properties.insert("rebase".to_string(), json!({"type": "boolean", "description": "Use rebase instead of merge", "default": false}));
        Tool {
            name: "git_pull".to_string(),
            description: "Pull changes from a remote repository.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec![]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn git_fetch_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert("remote".to_string(), json!({"type": "string", "description": "Remote name (default: origin)", "default": "origin"}));
        properties.insert(
            "all".to_string(),
            json!({"type": "boolean", "description": "Fetch all remotes", "default": false}),
        );
        properties.insert("prune".to_string(), json!({"type": "boolean", "description": "Remove stale remote-tracking refs", "default": false}));
        Tool {
            name: "git_fetch".to_string(),
            description: "Fetch changes from a remote without merging.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec![]),
            requires_approval: false,
            serialize: true,
            ..Default::default()
        }
    }

    fn git_discard_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert("files".to_string(), json!({"type": "array", "items": {"type": "string"}, "description": "Files to discard changes for."}));
        Tool {
            name: "git_discard".to_string(),
            description: "Discard uncommitted changes. WARNING: Permanent!".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["files".to_string()]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn git_branch_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            json!({"type": "string", "description": "Branch name"}),
        );
        properties.insert("action".to_string(), json!({"type": "string", "enum": ["list", "create", "switch", "delete"], "description": "Action to perform", "default": "list"}));
        properties.insert(
            "force".to_string(),
            json!({"type": "boolean", "description": "Force the action", "default": false}),
        );
        Tool {
            name: "git_branch".to_string(),
            description: "Manage git branches: list, create, switch, or delete.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec![]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    /// Execute a git tool
    #[tracing::instrument(name = "tool.execute", skip(input, context), fields(tool_name))]
    pub fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "git_status" => Self::git_status(context),
            "git_diff" => Self::git_diff(context),
            "git_log" => Self::git_log(input, context),
            "git_stage" => Self::git_stage(input, context),
            "git_unstage" => Self::git_unstage(input, context),
            "git_commit" => Self::git_commit(input, context),
            "git_push" => Self::git_push(input, context),
            "git_pull" => Self::git_pull(input, context),
            "git_fetch" => Self::git_fetch(input, context),
            "git_discard" => Self::git_discard(input, context),
            "git_branch" => Self::git_branch(input, context),
            _ => Err(anyhow::anyhow!("Unknown git tool: {}", tool_name)),
        };
        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Git operation failed: {}", e),
            ),
        }
    }

    fn git_status(context: &ToolContext) -> Result<String> {
        let repo = Repository::discover(&context.working_directory)?;
        let statuses = repo.statuses(None)?;
        let mut output = String::from("Git Status:\n\n");
        for entry in statuses.iter() {
            let path = entry.path().unwrap_or("?");
            let status = entry.status();
            output.push_str(&format!("{:?} - {}\n", status, path));
        }
        Ok(output)
    }

    fn git_diff(context: &ToolContext) -> Result<String> {
        let repo = Repository::discover(&context.working_directory)?;
        let head = repo.head()?.peel_to_tree()?;
        let diff = repo.diff_tree_to_workdir_with_index(Some(&head), None)?;
        Ok(format!("Git Diff:\n{} files changed", diff.deltas().len()))
    }

    fn git_log(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            #[serde(default = "default_limit")]
            limit: usize,
        }
        fn default_limit() -> usize {
            10
        }
        let params: Input = serde_json::from_value(input.clone()).unwrap_or(Input { limit: 10 });
        let repo = Repository::discover(&context.working_directory)?;
        let mut revwalk = repo.revwalk()?;
        revwalk.push_head()?;
        let mut output = String::from("Git Log:\n\n");
        for (i, oid) in revwalk.enumerate() {
            if i >= params.limit {
                break;
            }
            let commit = repo.find_commit(oid?)?;
            output.push_str(&format!(
                "{} - {}\n",
                commit.id(),
                commit.summary().unwrap_or("No message")
            ));
        }
        Ok(output)
    }

    fn git_stage(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            files: Vec<String>,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(&context.working_directory).arg("add");
        for file in &params.files {
            cmd.arg(file);
        }
        let output = cmd.output()?;
        if output.status.success() {
            Ok(format!(
                "Successfully staged {} file(s)",
                params.files.len()
            ))
        } else {
            Err(anyhow::anyhow!(
                "Failed to stage files: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    fn git_unstage(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            files: Vec<String>,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(&context.working_directory)
            .args(["reset", "HEAD", "--"]);
        for file in &params.files {
            cmd.arg(file);
        }
        let output = cmd.output()?;
        if output.status.success() {
            Ok(format!(
                "Successfully unstaged {} file(s)",
                params.files.len()
            ))
        } else {
            Err(anyhow::anyhow!(
                "Failed to unstage files: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    fn git_commit(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            message: String,
            #[serde(default)]
            all: bool,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(&context.working_directory).arg("commit");
        if params.all {
            cmd.arg("-a");
        }
        cmd.args(["-m", &params.message]);
        let output = cmd.output()?;
        if output.status.success() {
            Ok(format!(
                "Commit successful:\n{}",
                String::from_utf8_lossy(&output.stdout)
            ))
        } else {
            Err(anyhow::anyhow!(
                "Commit failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    fn git_push(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            #[serde(default = "dr")]
            remote: String,
            branch: Option<String>,
            #[serde(default)]
            set_upstream: bool,
        }
        fn dr() -> String {
            "origin".to_string()
        }
        let params: Input = serde_json::from_value(input.clone()).unwrap_or(Input {
            remote: "origin".to_string(),
            branch: None,
            set_upstream: false,
        });
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(&context.working_directory).arg("push");
        if params.set_upstream {
            cmd.arg("-u");
        }
        cmd.arg(&params.remote);
        if let Some(ref branch) = params.branch {
            cmd.arg(branch);
        }
        let output = cmd.output()?;
        if output.status.success() {
            Ok(format!(
                "Push successful:\n{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ))
        } else {
            Err(anyhow::anyhow!(
                "Push failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    fn git_pull(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            #[serde(default = "dr")]
            remote: String,
            branch: Option<String>,
            #[serde(default)]
            rebase: bool,
        }
        fn dr() -> String {
            "origin".to_string()
        }
        let params: Input = serde_json::from_value(input.clone()).unwrap_or(Input {
            remote: "origin".to_string(),
            branch: None,
            rebase: false,
        });
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(&context.working_directory).arg("pull");
        if params.rebase {
            cmd.arg("--rebase");
        }
        cmd.arg(&params.remote);
        if let Some(ref branch) = params.branch {
            cmd.arg(branch);
        }
        let output = cmd.output()?;
        if output.status.success() {
            Ok(format!(
                "Pull successful:\n{}",
                String::from_utf8_lossy(&output.stdout)
            ))
        } else {
            Err(anyhow::anyhow!(
                "Pull failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    fn git_fetch(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            #[serde(default = "dr")]
            remote: String,
            #[serde(default)]
            all: bool,
            #[serde(default)]
            prune: bool,
        }
        fn dr() -> String {
            "origin".to_string()
        }
        let params: Input = serde_json::from_value(input.clone()).unwrap_or(Input {
            remote: "origin".to_string(),
            all: false,
            prune: false,
        });
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(&context.working_directory).arg("fetch");
        if params.all {
            cmd.arg("--all");
        } else {
            cmd.arg(&params.remote);
        }
        if params.prune {
            cmd.arg("--prune");
        }
        let output = cmd.output()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let fetch_output = if stdout.is_empty() && stderr.is_empty() {
                "Already up to date.".to_string()
            } else {
                format!("{}{}", stdout, stderr)
            };
            Ok(format!("Fetch successful:\n{}", fetch_output))
        } else {
            Err(anyhow::anyhow!(
                "Fetch failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    fn git_discard(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            files: Vec<String>,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(&context.working_directory)
            .args(["checkout", "--"]);
        for file in &params.files {
            cmd.arg(file);
        }
        let output = cmd.output()?;
        if output.status.success() {
            Ok(format!(
                "Successfully discarded changes to {} file(s)",
                params.files.len()
            ))
        } else {
            Err(anyhow::anyhow!(
                "Failed to discard changes: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    fn git_branch(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            name: Option<String>,
            #[serde(default = "da")]
            action: String,
            #[serde(default)]
            force: bool,
        }
        fn da() -> String {
            "list".to_string()
        }
        let params: Input = serde_json::from_value(input.clone()).unwrap_or(Input {
            name: None,
            action: "list".to_string(),
            force: false,
        });
        let branch_name = params.name.clone();
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(&context.working_directory);
        match params.action.as_str() {
            "list" => {
                cmd.args(["branch", "-a", "-v"]);
            }
            "create" => {
                let n = params
                    .name
                    .ok_or_else(|| anyhow::anyhow!("Branch name required"))?;
                cmd.args(["branch", &n]);
            }
            "switch" => {
                let n = params
                    .name
                    .ok_or_else(|| anyhow::anyhow!("Branch name required"))?;
                cmd.args(["checkout", &n]);
            }
            "delete" => {
                let n = params
                    .name
                    .ok_or_else(|| anyhow::anyhow!("Branch name required"))?;
                if params.force {
                    cmd.args(["branch", "-D", &n]);
                } else {
                    cmd.args(["branch", "-d", &n]);
                }
            }
            _ => return Err(anyhow::anyhow!("Unknown branch action: {}", params.action)),
        }
        let output = cmd.output()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            Ok(match params.action.as_str() {
                "list" => format!("Branches:\n{}", stdout),
                "create" => format!("Created branch '{}'", branch_name.unwrap_or_default()),
                "switch" => format!("Switched to branch '{}'", branch_name.unwrap_or_default()),
                "delete" => format!("Deleted branch '{}'", branch_name.unwrap_or_default()),
                _ => stdout.to_string(),
            })
        } else {
            Err(anyhow::anyhow!(
                "Branch operation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> ToolContext {
        ToolContext {
            working_directory: std::env::current_dir()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_get_tools() {
        let tools = GitTool::get_tools();
        assert_eq!(tools.len(), 11);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"git_status"));
        assert!(names.contains(&"git_commit"));
        assert!(names.contains(&"git_branch"));
    }

    #[test]
    fn test_execute_unknown_tool() {
        let context = create_test_context();
        let input = json!({});
        let result = GitTool::execute("1", "unknown_tool", &input, &context);
        assert!(result.is_error);
    }
}
