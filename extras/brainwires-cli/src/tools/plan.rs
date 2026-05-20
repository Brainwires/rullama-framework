//! Plan Tool - Spawns a planning agent with isolated context
//!
//! Creates an execution plan for a task by spawning a dedicated planning agent
//! that researches and analyzes in its own context, returning only the final plan.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agents::{CommunicationHub, FileLockManager, TaskAgent, TaskAgentConfig};
use crate::config::PlatformPaths;
use crate::providers::Provider;
use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};
use crate::types::agent::{AgentContext, PermissionMode, Task};
use crate::types::plan::{PlanMetadata, PlanStatus};
use crate::types::tool::{Tool, ToolInputSchema, ToolResult};
use crate::utils::entity_extraction::EntityExtractor;
use crate::utils::logger::Logger;

/// Plan tool - spawns planning agent with isolated context
pub struct PlanTool {
    provider: Arc<dyn Provider>,
}

impl PlanTool {
    /// Create a new PlanTool
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self { provider }
    }

    /// Get all plan tool definitions (static, no provider needed)
    pub fn get_tools() -> Vec<Tool> {
        vec![Self::plan_task_tool()]
    }

    /// Plan task tool definition
    fn plan_task_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "task".to_string(),
            json!({
                "type": "string",
                "description": "The task to create a plan for"
            }),
        );
        properties.insert(
            "context".to_string(),
            json!({
                "type": "string",
                "description": "Optional additional context or constraints for the planning",
                "default": ""
            }),
        );
        properties.insert(
            "max_iterations".to_string(),
            json!({
                "type": "integer",
                "description": "Maximum iterations for research/planning (default: 10)",
                "default": 10
            }),
        );

        Tool {
            name: "plan_task".to_string(),
            description: "Create an execution plan for a task. Spawns a planning agent that \
                researches the codebase and returns a structured plan. The planning agent \
                runs in its own isolated context with read-only access, so all research \
                tokens are spent separately. Only the final plan is returned."
                .to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["task".to_string()]),
            requires_approval: false, // Read-only planning is safe
            defer_loading: true,      // Plan tool is deferred
            ..Default::default()
        }
    }

    /// Execute the plan_task tool
    ///
    /// Returns a boxed future to break async recursion cycle
    /// (ToolExecutor -> PlanTool -> TaskAgent -> ToolExecutor)
    pub fn execute(
        &self,
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
    ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + '_>> {
        let tool_use_id = tool_use_id.to_string();
        let tool_name = tool_name.to_string();
        let input = input.clone();

        Box::pin(async move {
            let result = match tool_name.as_str() {
                "plan_task" => self.execute_plan_task(&input).await,
                _ => Err(anyhow::anyhow!("Unknown plan tool: {}", tool_name)),
            };

            match result {
                Ok(output) => ToolResult::success(tool_use_id, output),
                Err(e) => ToolResult::error(tool_use_id, format!("Planning failed: {}", e)),
            }
        })
    }

    /// Execute plan_task - spawns agent, waits for completion, returns plan
    async fn execute_plan_task(&self, input: &Value) -> anyhow::Result<String> {
        #[derive(Deserialize)]
        struct Input {
            task: String,
            #[serde(default)]
            context: String,
            #[serde(default = "default_max_iterations")]
            max_iterations: u32,
        }

        fn default_max_iterations() -> u32 {
            10
        }

        let params: Input = serde_json::from_value(input.clone())?;

        // Build planning prompt
        let planning_prompt = if params.context.is_empty() {
            params.task.clone()
        } else {
            format!("{}\n\nAdditional context:\n{}", params.task, params.context)
        };

        // Create isolated context for the planning agent
        let working_dir = std::env::current_dir()?.to_string_lossy().to_string();

        // Get read-only tools for the planning agent
        let all_tools = brainwires_tool_builtins::registry_with_builtins()
            .get_all()
            .to_vec();
        let read_only_tools: Vec<_> = all_tools
            .into_iter()
            .filter(|t| {
                // Include read-only tools
                let name = t.name.as_str();
                matches!(
                    name,
                    "read_file"
                        | "list_directory"
                        | "search_code"
                        | "search_files"
                        | "query_codebase"
                        | "index_codebase"
                        | "web_search"
                        | "web_fetch"
                        | "git_status"
                        | "git_log"
                        | "git_diff"
                        | "git_show"
                )
            })
            .collect();

        let context = AgentContext {
            working_directory: working_dir.clone(),
            conversation_history: Vec::new(),
            tools: read_only_tools,
            user_id: None,
            metadata: HashMap::new(),
            working_set: crate::types::WorkingSet::new(),
            capabilities: brainwires::permissions::AgentCapabilities::read_only(),
        };

        // Configure the planning agent
        let config = TaskAgentConfig {
            max_iterations: params.max_iterations,
            permission_mode: PermissionMode::ReadOnly,
            system_prompt: Some(crate::system_prompts::planning_agent_system_prompt(
                &working_dir,
            )),
            temperature: 0.7,
            max_tokens: 4096,
            validation_config: None, // No validation for read-only planning agents
            mdap_config: None,       // MDAP not used for planning agents
            analytics_collector: crate::utils::logger::analytics_collector()
                .map(std::sync::Arc::new),
            role: None,
            max_total_tokens: None,
            max_cost_usd: None,
            timeout_secs: None,
            session_budget: None,
        };

        // Create task
        let task = Task::new(format!("plan-{}", uuid::Uuid::new_v4()), planning_prompt);

        // Create communication hub and file lock manager for this agent
        // (isolated - not shared with the main agent pool)
        let communication_hub = Arc::new(CommunicationHub::new());
        let file_lock_manager = Arc::new(FileLockManager::new());

        // Create and execute the planning agent synchronously
        let agent = TaskAgent::new(
            format!("planner-{}", uuid::Uuid::new_v4()),
            task,
            Arc::clone(&self.provider),
            communication_hub,
            file_lock_manager,
            context,
            config,
        );

        // Execute and wait for completion (this is blocking, not background)
        let result = agent.execute().await?;

        if result.success {
            // Save the plan to storage
            let plan_id = self
                .save_plan_to_storage(&params.task, &result.summary, result.iterations)
                .await
                .unwrap_or_else(|e| {
                    Logger::debug(format!("Failed to save plan: {}", e));
                    "unsaved".to_string()
                });

            // Format the plan output with plan ID
            Ok(format!(
                "## Execution Plan\n\n{}\n\n---\n*Planning completed in {} iterations*\n*Plan ID: {}*",
                result.summary, result.iterations, plan_id
            ))
        } else {
            Err(anyhow::anyhow!("Planning failed: {}", result.summary))
        }
    }

    /// Save a plan to storage and return the plan ID
    async fn save_plan_to_storage(
        &self,
        task: &str,
        plan_content: &str,
        iterations: u32,
    ) -> anyhow::Result<String> {
        // Initialize storage
        let db_path = PlatformPaths::conversations_db_path()?;
        let client = Arc::new(
            LanceDatabase::new(
                db_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Invalid DB path"))?,
            )
            .await?,
        );

        let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
        client.initialize(embeddings.dimension()).await?;

        let plan_store = PlanStore::new(client, embeddings);

        // Create plan metadata (agent-generated plans don't have a conversation ID)
        let conversation_id = format!("agent-{}", uuid::Uuid::new_v4());
        let mut plan =
            PlanMetadata::new(conversation_id, task.to_string(), plan_content.to_string());
        plan = plan.with_iterations(iterations);
        plan.set_status(PlanStatus::Active);

        // Extract entities for context
        let entity_extractor = EntityExtractor::new();
        let extraction = entity_extractor.extract(plan_content, &plan.plan_id);

        if !extraction.entities.is_empty() {
            Logger::debug(format!(
                "Extracted {} entities from agent plan",
                extraction.entities.len()
            ));
        }

        // Save to database and export to markdown
        let _file_path = plan_store.save_and_export(&mut plan).await?;

        Ok(plan.plan_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tools() {
        let tools = PlanTool::get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "plan_task");
    }

    #[test]
    fn test_plan_task_tool_definition() {
        let tool = PlanTool::plan_task_tool();
        assert_eq!(tool.name, "plan_task");
        assert!(!tool.requires_approval); // Read-only is safe
        assert!(tool.description.contains("isolated context"));
    }

    #[test]
    fn test_planning_system_prompt() {
        let prompt = crate::system_prompts::planning_agent_system_prompt("/test/dir");
        assert!(prompt.contains("/test/dir"));
        assert!(prompt.contains("read-only"));
        assert!(prompt.contains("Do NOT execute"));
    }
}
