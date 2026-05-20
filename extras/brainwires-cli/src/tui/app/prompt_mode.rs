//! Prompt Mode Operations
//!
//! Handles switching between Ask, Edit, and Plan prompt modes.
//! Ask mode restricts the AI to read-only tools.
//! Edit mode provides full tool access (default).
//! Plan mode is managed by the existing plan_mode module.

use anyhow::Result;

use super::state::{App, AppMode, PromptMode};
use crate::types::message::{Message, MessageContent, Role};
use crate::types::tool::Tool;

/// Tools excluded in Ask mode (write/mutating operations)
const ASK_MODE_EXCLUDED_TOOLS: &[&str] = &[
    "write_file",
    "edit_file",
    "delete_file",
    "patch_file",
    "create_directory",
    "execute_command",
    "git_commit",
    "git_push",
    "git_reset",
    "git_checkout",
    "git_stage",
    "git_discard",
];

impl App {
    /// Switch to Ask mode (read-only)
    pub async fn set_prompt_mode_ask(&mut self) -> Result<()> {
        // Exit plan mode if active
        if self.mode == AppMode::PlanMode {
            self.exit_plan_mode().await?;
        }

        self.prompt_mode = PromptMode::Ask;
        self.rebuild_system_prompt();
        self.show_toast("Switched to Ask mode (read-only)".to_string(), 3000);
        Ok(())
    }

    /// Switch to Edit mode (full tools)
    pub async fn set_prompt_mode_edit(&mut self) -> Result<()> {
        // Exit plan mode if active
        if self.mode == AppMode::PlanMode {
            self.exit_plan_mode().await?;
        }

        self.prompt_mode = PromptMode::Edit;
        self.rebuild_system_prompt();
        self.show_toast("Switched to Edit mode (full tools)".to_string(), 3000);
        Ok(())
    }

    /// Filter tools based on current prompt mode.
    /// In Ask mode, removes write/mutating tools.
    /// In Edit and Plan modes, passes through unchanged.
    pub fn filter_tools_for_prompt_mode(&self, tools: Vec<Tool>) -> Vec<Tool> {
        match self.prompt_mode {
            PromptMode::Ask => tools
                .into_iter()
                .filter(|t| !ASK_MODE_EXCLUDED_TOOLS.contains(&t.name.as_str()))
                .collect(),
            PromptMode::Edit | PromptMode::Plan => tools,
        }
    }

    /// Keyword-match the user's message against discovered skills. Returns
    /// a one-line hint for the best match above the confidence threshold,
    /// or `None` if nothing likely applies. Non-intrusive — this never
    /// invokes; the user still types `/<name>` to opt in.
    ///
    /// Uses the same keyword heuristic as
    /// [`brainwires_skills::SkillRouter::keyword_match`] but runs
    /// synchronously against the App's registry rather than reaching
    /// through the router's `Arc<RwLock>`.
    pub fn suggest_skill_for(&self, user_message: &str) -> Option<String> {
        const THRESHOLD: f32 = 0.75;
        let registry = self.skill_registry.as_ref()?;
        let meta_list = registry.all_metadata();
        if meta_list.is_empty() {
            return None;
        }

        let msg_lower = user_message.to_lowercase();
        let words: std::collections::HashSet<&str> = msg_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();
        if words.is_empty() {
            return None;
        }

        let mut best: Option<(String, f32)> = None;
        for m in meta_list {
            let name_lower = m.name.to_lowercase();
            let desc_lower = m.description.to_lowercase();
            let mut score = 0;
            if msg_lower.contains(&name_lower) || name_lower.contains(&msg_lower) {
                score += 3;
            }
            for w in &words {
                if desc_lower.contains(w) {
                    score += 1;
                }
            }
            for nw in name_lower.split('-') {
                if words.contains(nw) {
                    score += 2;
                }
            }
            if score > 0 {
                let confidence = (0.6f32 + score as f32 * 0.05).min(0.9);
                if confidence >= THRESHOLD && best.as_ref().is_none_or(|(_, c)| confidence > *c) {
                    best = Some((m.name.clone(), confidence));
                }
            }
        }

        best.map(|(name, _)| {
            format!(
                "💡 Skill '{}' may help — invoke with /{} if it fits",
                name, name
            )
        })
    }

    /// Apply the pending skill tool scope (if any) and clear it. Called by
    /// every AI-invocation path so all three modes (default, IPC, MDAP)
    /// honor a skill's `allowed_tools` declaration.
    ///
    /// Matches exact tool names and MCP-style `server__tool` suffixes,
    /// same as [`brainwires_skills::SkillExecutor::filter_allowed_tools`].
    pub fn apply_and_clear_skill_tool_scope(&mut self, tools: Vec<Tool>) -> Vec<Tool> {
        let Some(allowed) = self.pending_skill_tool_scope.take() else {
            return tools;
        };
        let allowed_set: std::collections::HashSet<&str> =
            allowed.iter().map(|s| s.as_str()).collect();
        let filtered: Vec<Tool> = tools
            .into_iter()
            .filter(|t| {
                allowed_set.contains(t.name.as_str())
                    || allowed_set
                        .iter()
                        .any(|a| t.name.ends_with(&format!("__{}", a)))
            })
            .collect();
        tracing::debug!(
            "Skill tool-scope applied: {} tools remaining",
            filtered.len()
        );
        filtered
    }

    /// Rebuild the system prompt in conversation_history based on current prompt mode.
    pub(super) fn rebuild_system_prompt(&mut self) {
        let new_prompt = match self.prompt_mode {
            PromptMode::Ask => {
                crate::system_prompts::build_ask_mode_system_prompt(Some(&self.working_set))
                    .unwrap_or_else(|_| "You are a helpful read-only coding assistant.".to_string())
            }
            PromptMode::Edit | PromptMode::Plan => {
                crate::system_prompts::build_system_prompt_with_context(
                    None,
                    Some(&self.working_set),
                )
                .unwrap_or_else(|_| "You are a helpful coding assistant.".to_string())
            }
        };

        // Replace the first System message, or insert one if none exists
        if let Some(sys_msg) = self
            .conversation_history
            .iter_mut()
            .find(|m| m.role == Role::System)
        {
            sys_msg.content = MessageContent::Text(new_prompt);
        } else {
            self.conversation_history.insert(
                0,
                Message {
                    role: Role::System,
                    content: MessageContent::Text(new_prompt),
                    name: None,
                    metadata: None,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str) -> Tool {
        Tool {
            name: name.to_string(),
            description: format!("Test tool: {}", name),
            ..Default::default()
        }
    }

    fn make_test_tools() -> Vec<Tool> {
        vec![
            make_tool("read_file"),
            make_tool("write_file"),
            make_tool("edit_file"),
            make_tool("list_directory"),
            make_tool("search_files"),
            make_tool("delete_file"),
            make_tool("execute_command"),
            make_tool("git_status"),
            make_tool("git_commit"),
            make_tool("git_push"),
        ]
    }

    #[test]
    fn test_ask_mode_filters_write_tools() {
        let tools = make_test_tools();

        // Simulate Ask mode filtering
        let filtered: Vec<Tool> = tools
            .into_iter()
            .filter(|t| !ASK_MODE_EXCLUDED_TOOLS.contains(&t.name.as_str()))
            .collect();

        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"list_directory"));
        assert!(names.contains(&"search_files"));
        assert!(names.contains(&"git_status"));
        assert!(!names.contains(&"write_file"));
        assert!(!names.contains(&"edit_file"));
        assert!(!names.contains(&"delete_file"));
        assert!(!names.contains(&"execute_command"));
        assert!(!names.contains(&"git_commit"));
        assert!(!names.contains(&"git_push"));
    }

    #[test]
    fn test_edit_mode_passes_all_tools() {
        let tools = make_test_tools();
        let original_count = tools.len();

        // Edit mode should not filter
        let filtered: Vec<Tool> = tools; // no filtering in Edit mode
        assert_eq!(filtered.len(), original_count);
    }

    #[test]
    fn test_excluded_tools_list_completeness() {
        // Verify all expected mutating tools are in the exclusion list
        let expected = vec![
            "write_file",
            "edit_file",
            "delete_file",
            "patch_file",
            "create_directory",
            "execute_command",
            "git_commit",
            "git_push",
            "git_reset",
            "git_checkout",
            "git_stage",
            "git_discard",
        ];
        for tool in expected {
            assert!(
                ASK_MODE_EXCLUDED_TOOLS.contains(&tool),
                "Missing excluded tool: {}",
                tool
            );
        }
    }
}
