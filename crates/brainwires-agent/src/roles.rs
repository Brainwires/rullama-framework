//! Agent role definitions for constrained, least-privilege execution.
//!
//! Each `AgentRole` maps to a specific tool allow-list and a brief system-prompt
//! suffix that reinforces the role boundary to the model.
//!
//! # Enforcement
//!
//! Enforcement happens at *provider call time* — the model only receives the tools
//! it is allowed to use. This prevents hallucination on unavailable tools and wastes
//! fewer tokens on irrelevant tool descriptions.
//!
//! Use [`AgentRole::filter_tools`] to obtain the filtered tool slice before calling
//! the provider.

use brainwires_core::Tool;

/// Role assigned to a `TaskAgent` that restricts its available tools.
///
/// When no role is set an agent receives all tools in its context — equivalent to
/// `Execution`. Roles are opt-in so existing callers are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// Read-only exploration: file reads, directory listing, search, web fetch.
    ///
    /// Safe to run against untrusted or sensitive repositories. Cannot write,
    /// execute commands, or spawn additional agents.
    Exploration,

    /// Planning only: task management + read access.
    ///
    /// May create and query tasks but cannot modify files or run code. Intended
    /// for the planning phase before execution begins.
    Planning,

    /// Verification: read access and build/test execution.
    ///
    /// May read files and run validation tools (build, test, lint). Cannot write
    /// files or modify the repository. Used after `Execution` to confirm results.
    Verification,

    /// Full execution: all tools available. Requires explicit grant.
    ///
    /// Identical to having no role set. Named explicitly so callers are clear that
    /// they are granting unrestricted tool access.
    Execution,
}

impl AgentRole {
    /// Tool names allowed for this role. `None` means all tools are permitted.
    pub fn allowed_tools(self) -> Option<&'static [&'static str]> {
        match self {
            Self::Exploration => Some(&[
                "read_file",
                "list_directory",
                "search_code",
                "query_codebase",
                "fetch_url",
                "web_search",
                "glob",
                "grep",
                "context_recall",
                "task_get",
                "task_list",
            ]),
            Self::Planning => Some(&[
                "read_file",
                "list_directory",
                "glob",
                "grep",
                "task_create",
                "task_update",
                "task_add_subtask",
                "task_list",
                "task_get",
                "plan_task",
                "context_recall",
            ]),
            Self::Verification => Some(&[
                "read_file",
                "list_directory",
                "glob",
                "grep",
                "execute_command",
                "check_duplicates",
                "verify_build",
                "check_syntax",
                "task_get",
                "task_list",
                "context_recall",
            ]),
            Self::Execution => None,
        }
    }

    /// Filter a tool slice to only those permitted by this role.
    ///
    /// Returns a `Vec<Tool>` that can be passed directly to the provider. When
    /// the role is `Execution` the original slice is cloned in full.
    pub fn filter_tools(self, tools: &[Tool]) -> Vec<Tool> {
        match self.allowed_tools() {
            None => tools.to_vec(),
            Some(allow) => tools
                .iter()
                .filter(|t| allow.contains(&t.name.as_str()))
                .cloned()
                .collect(),
        }
    }

    /// Short system-prompt suffix that reminds the model of its constraints.
    pub fn system_prompt_suffix(self) -> &'static str {
        match self {
            Self::Exploration => {
                "\n\n[ROLE: Exploration] You may only read files and search. \
                Do not attempt to write files, run commands, or spawn agents."
            }
            Self::Planning => {
                "\n\n[ROLE: Planning] You may read files and manage tasks. \
                Do not write files or execute code — produce a plan only."
            }
            Self::Verification => {
                "\n\n[ROLE: Verification] You may read files and run build/test commands. \
                Do not write or delete files."
            }
            Self::Execution => "",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_tool(name: &str) -> Tool {
        Tool {
            name: name.to_string(),
            description: String::new(),
            input_schema: brainwires_core::ToolInputSchema::default(),
            requires_approval: false,
            defer_loading: false,
            allowed_callers: vec![],
            input_examples: vec![],
            serialize: false,
        }
    }

    #[test]
    fn exploration_filters_write_tools() {
        let tools = vec![
            fake_tool("read_file"),
            fake_tool("write_file"),
            fake_tool("execute_command"),
            fake_tool("glob"),
        ];
        let filtered = AgentRole::Exploration.filter_tools(&tools);
        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"glob"));
        assert!(!names.contains(&"write_file"));
        assert!(!names.contains(&"execute_command"));
    }

    #[test]
    fn execution_passes_all_tools() {
        let tools = vec![fake_tool("read_file"), fake_tool("write_file")];
        let filtered = AgentRole::Execution.filter_tools(&tools);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn planning_allows_task_tools_not_write_or_execute() {
        let tools = vec![
            fake_tool("read_file"),
            fake_tool("task_create"),
            fake_tool("task_update"),
            fake_tool("plan_task"),
            fake_tool("write_file"),
            fake_tool("execute_command"),
        ];
        let filtered = AgentRole::Planning.filter_tools(&tools);
        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"task_create"));
        assert!(names.contains(&"task_update"));
        assert!(names.contains(&"plan_task"));
        assert!(!names.contains(&"write_file"));
        assert!(!names.contains(&"execute_command"));
    }

    #[test]
    fn verification_allows_execute_command_not_write() {
        let tools = vec![
            fake_tool("read_file"),
            fake_tool("execute_command"),
            fake_tool("verify_build"),
            fake_tool("write_file"),
            fake_tool("task_create"),
        ];
        let filtered = AgentRole::Verification.filter_tools(&tools);
        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"execute_command"));
        assert!(names.contains(&"verify_build"));
        assert!(!names.contains(&"write_file"));
        assert!(!names.contains(&"task_create"));
    }

    #[test]
    fn system_prompt_suffix_non_empty_for_constrained_roles() {
        assert!(!AgentRole::Exploration.system_prompt_suffix().is_empty());
        assert!(!AgentRole::Planning.system_prompt_suffix().is_empty());
        assert!(!AgentRole::Verification.system_prompt_suffix().is_empty());
        assert_eq!(AgentRole::Execution.system_prompt_suffix(), "");
    }
}
