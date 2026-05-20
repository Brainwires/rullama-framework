//! Skill Executor
//!
//! Executes skills in one of three modes:
//! - **Inline**: Instructions returned for injection into the conversation
//! - **Subagent**: Execution info returned; caller spawns via AgentPool
//! - **Script**: Script content returned; caller executes via OrchestratorTool
//!
//! Tool restrictions from `allowed-tools` are enforced in `prepare_*` methods.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::metadata::{Skill, SkillExecutionMode, SkillResult};
use super::parser::render_template;
use super::registry::SkillRegistry;

/// Skill executor handles the execution of skills in various modes
pub struct SkillExecutor {
    /// Reference to skill registry for loading skills
    registry: Arc<RwLock<SkillRegistry>>,
}

impl SkillExecutor {
    /// Create a new skill executor
    pub fn new(registry: Arc<RwLock<SkillRegistry>>) -> Self {
        Self { registry }
    }

    /// Execute a skill by name
    ///
    /// Loads the skill from registry and executes it with the given arguments.
    pub async fn execute_by_name(
        &self,
        skill_name: &str,
        args: HashMap<String, String>,
    ) -> Result<SkillResult> {
        let mut registry = self.registry.write().await;
        let skill = registry
            .get_skill(skill_name)
            .with_context(|| format!("Failed to load skill '{}'", skill_name))?
            .clone();
        drop(registry);

        self.execute(&skill, args).await
    }

    /// Execute a skill
    ///
    /// Dispatches to the appropriate execution mode based on skill metadata.
    pub async fn execute(
        &self,
        skill: &Skill,
        args: HashMap<String, String>,
    ) -> Result<SkillResult> {
        let instructions = render_template(&skill.instructions, &args);

        match skill.execution_mode {
            SkillExecutionMode::Inline => self.execute_inline(skill, &instructions).await,
            SkillExecutionMode::Subagent => self.execute_subagent(skill, &instructions).await,
            SkillExecutionMode::Script => self.execute_script(skill, &instructions).await,
        }
    }

    /// Execute skill inline — returns instructions for injection into the conversation.
    async fn execute_inline(&self, skill: &Skill, instructions: &str) -> Result<SkillResult> {
        tracing::info!("Executing skill '{}' inline", skill.name());

        let full_instructions = format!(
            "## Skill: {}\n\n{}\n\n---\n\n{}",
            skill.name(),
            skill.description(),
            instructions
        );

        Ok(SkillResult::inline(
            full_instructions,
            skill.model().cloned(),
        ))
    }

    /// Execute skill as a subagent — returns an agent ID; caller spawns via AgentPool.
    async fn execute_subagent(&self, skill: &Skill, instructions: &str) -> Result<SkillResult> {
        tracing::info!("Executing skill '{}' as subagent", skill.name());

        let agent_id = format!("skill-{}-{}", skill.name(), uuid::Uuid::new_v4());

        tracing::debug!(
            "Prepared subagent task '{}' with {} instructions chars",
            agent_id,
            instructions.len()
        );

        Ok(SkillResult::Subagent { agent_id })
    }

    /// Execute skill as a Rhai script — returns script content; caller executes via OrchestratorTool.
    async fn execute_script(&self, skill: &Skill, script: &str) -> Result<SkillResult> {
        tracing::info!("Executing skill '{}' as script", skill.name());

        if !script.contains("let ") && !script.contains("fn ") && !script.contains(";") {
            tracing::warn!(
                "Script for skill '{}' doesn't look like valid Rhai code",
                skill.name()
            );
        }

        Ok(SkillResult::Script {
            output: script.to_string(),
            is_error: false,
        })
    }

    /// Filter available tool names to only those allowed by the skill.
    fn filter_allowed_tools(&self, skill: &Skill, available: &[String]) -> Vec<String> {
        if let Some(allowed_tools) = skill.allowed_tools() {
            available
                .iter()
                .filter(|name| {
                    allowed_tools.iter().any(|allowed| {
                        // Match exact name or MCP-style prefix (server__tool)
                        *name == allowed || name.ends_with(&format!("__{}", allowed))
                    })
                })
                .cloned()
                .collect()
        } else {
            available.to_vec()
        }
    }

    /// Prepare a subagent execution context.
    ///
    /// Returns task description, filtered tool names, and system prompt.
    /// Caller (who has AgentPool access) converts this into a Task + AgentContext.
    pub async fn prepare_subagent(
        &self,
        skill: &Skill,
        available_tool_names: &[String],
        args: HashMap<String, String>,
    ) -> Result<SubagentPrepared> {
        let instructions = render_template(&skill.instructions, &args);
        let allowed_tool_names = self.filter_allowed_tools(skill, available_tool_names);

        let system_prompt = format!(
            "You are executing the '{}' skill.\n\n\
             **Description**: {}\n\n\
             **Instructions**:\n{}",
            skill.name(),
            skill.description(),
            instructions
        );

        Ok(SubagentPrepared {
            task_description: instructions,
            allowed_tool_names,
            system_prompt,
            model_override: skill.model().cloned(),
        })
    }

    /// Prepare a script execution.
    ///
    /// Returns the rendered script and filtered tool names.
    /// Caller (who has OrchestratorTool access) handles execution.
    pub async fn prepare_script(
        &self,
        skill: &Skill,
        available_tool_names: &[String],
        args: HashMap<String, String>,
    ) -> Result<ScriptPrepared> {
        let script_content = render_template(&skill.instructions, &args);
        let allowed_tool_names = self.filter_allowed_tools(skill, available_tool_names);

        Ok(ScriptPrepared {
            script_content,
            allowed_tool_names,
            model_override: skill.model().cloned(),
            skill_name: skill.name().to_string(),
        })
    }

    /// Get the execution mode for a skill
    pub async fn get_execution_mode(&self, skill_name: &str) -> Result<SkillExecutionMode> {
        let registry = self.registry.read().await;
        let metadata = registry
            .get_metadata(skill_name)
            .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", skill_name))?;
        Ok(metadata.execution_mode())
    }
}

/// Prepared subagent execution — caller converts into Task + AgentContext.
#[derive(Debug, Clone)]
pub struct SubagentPrepared {
    /// Task description (rendered instructions)
    pub task_description: String,
    /// Tool names allowed for this skill (filtered from available tools)
    pub allowed_tool_names: Vec<String>,
    /// System prompt for the subagent
    pub system_prompt: String,
    /// Optional model override
    pub model_override: Option<String>,
}

/// Prepared script execution — caller executes via OrchestratorTool.
#[derive(Debug, Clone)]
pub struct ScriptPrepared {
    /// The rendered Rhai script content
    pub script_content: String,
    /// Tool names allowed for this skill (filtered from available tools)
    pub allowed_tool_names: Vec<String>,
    /// Optional model override
    pub model_override: Option<String>,
    /// Skill name for logging
    pub skill_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::SkillMetadata;

    fn create_available_tools() -> Vec<String> {
        vec![
            "Read".to_string(),
            "Write".to_string(),
            "Grep".to_string(),
            "git_diff".to_string(),
        ]
    }

    fn create_test_skill() -> Skill {
        let mut metadata = SkillMetadata::new("test-skill".to_string(), "A test skill".to_string());
        metadata.allowed_tools = Some(vec!["Read".to_string(), "Grep".to_string()]);

        Skill {
            metadata,
            instructions: "Do the test with {{arg1}}".to_string(),
            execution_mode: SkillExecutionMode::Inline,
        }
    }

    #[tokio::test]
    async fn test_execute_inline() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let executor = SkillExecutor::new(registry);

        let skill = create_test_skill();
        let mut args = HashMap::new();
        args.insert("arg1".to_string(), "value1".to_string());

        let result = executor.execute(&skill, args).await.unwrap();

        match result {
            SkillResult::Inline { instructions, .. } => {
                assert!(instructions.contains("test-skill"));
                assert!(instructions.contains("value1"));
            }
            _ => panic!("Expected inline result"),
        }
    }

    #[tokio::test]
    async fn test_execute_subagent() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let executor = SkillExecutor::new(registry);

        let mut skill = create_test_skill();
        skill.execution_mode = SkillExecutionMode::Subagent;

        let args = HashMap::new();
        let result = executor.execute(&skill, args).await.unwrap();

        match result {
            SkillResult::Subagent { agent_id } => {
                assert!(agent_id.starts_with("skill-test-skill-"));
            }
            _ => panic!("Expected subagent result"),
        }
    }

    #[tokio::test]
    async fn test_execute_script() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let executor = SkillExecutor::new(registry);

        let mut skill = create_test_skill();
        skill.execution_mode = SkillExecutionMode::Script;
        skill.instructions = "let x = 1; x + 1".to_string();

        let args = HashMap::new();
        let result = executor.execute(&skill, args).await.unwrap();

        match result {
            SkillResult::Script { output, is_error } => {
                assert!(!is_error);
                assert!(output.contains("let x = 1"));
            }
            _ => panic!("Expected script result"),
        }
    }

    #[tokio::test]
    async fn test_filter_allowed_tools() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let executor = SkillExecutor::new(registry);

        let skill = create_test_skill(); // allowed: Read, Grep
        let available = create_available_tools();

        let filtered = executor.filter_allowed_tools(&skill, &available);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&"Read".to_string()));
        assert!(filtered.contains(&"Grep".to_string()));
        assert!(!filtered.contains(&"Write".to_string()));
        assert!(!filtered.contains(&"git_diff".to_string()));
    }

    #[tokio::test]
    async fn test_no_tool_restrictions() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let executor = SkillExecutor::new(registry);

        let mut skill = create_test_skill();
        skill.metadata.allowed_tools = None; // No restrictions

        let available = create_available_tools();
        let filtered = executor.filter_allowed_tools(&skill, &available);

        // Should have all tools
        assert_eq!(filtered.len(), 4);
    }

    #[tokio::test]
    async fn test_prepare_subagent() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let executor = SkillExecutor::new(registry);

        let mut skill = create_test_skill();
        skill.execution_mode = SkillExecutionMode::Subagent;

        let available = create_available_tools();
        let mut args = HashMap::new();
        args.insert("arg1".to_string(), "test_value".to_string());

        let prepared = executor
            .prepare_subagent(&skill, &available, args)
            .await
            .unwrap();

        assert!(prepared.task_description.contains("test_value"));
        assert!(prepared.system_prompt.contains("test-skill"));
        // Context should be restricted to Read + Grep
        assert_eq!(prepared.allowed_tool_names.len(), 2);
        assert!(prepared.allowed_tool_names.contains(&"Read".to_string()));
        assert!(prepared.allowed_tool_names.contains(&"Grep".to_string()));
    }

    #[tokio::test]
    async fn test_prepare_script() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let executor = SkillExecutor::new(registry);

        let mut skill = create_test_skill();
        skill.execution_mode = SkillExecutionMode::Script;
        skill.instructions = "let result = {{value}}; result".to_string();

        let available = create_available_tools();
        let mut args = HashMap::new();
        args.insert("value".to_string(), "42".to_string());

        let prepared = executor
            .prepare_script(&skill, &available, args)
            .await
            .unwrap();

        assert!(prepared.script_content.contains("let result = 42"));
        assert_eq!(prepared.skill_name, "test-skill");
        // Context should be restricted to Read + Grep
        assert_eq!(prepared.allowed_tool_names.len(), 2);
    }
}
