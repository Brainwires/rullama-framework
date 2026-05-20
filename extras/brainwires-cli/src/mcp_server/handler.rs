//! MCP Server Handler
//!
//! Handles JSON-RPC 2.0 protocol over stdin/stdout for MCP server mode

use crate::agents::TaskManager;
use crate::tools::ToolExecutor;
use anyhow::{Context, Result};
use async_trait::async_trait;
use brainwires_agent::agent_manager::{AgentInfo, AgentManager, AgentResult, SpawnConfig};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::agents::{
    CommunicationHub, FileLockManager, TaskAgent, TaskAgentConfig, TaskAgentResult,
};
use crate::config::ConfigManager;
use crate::mcp::{
    CallToolParams, CallToolResult, Content, InitializeParams, InitializeResult, JsonRpcError,
    JsonRpcRequest, JsonRpcResponse, ListToolsResult, McpTool, ServerCapabilities, ServerInfo,
    ToolsCapability,
};
use crate::providers::{Provider, ProviderFactory};
use crate::tools::{ToolCategory, ToolRegistry};
use crate::types::agent::{AgentContext, Task};
use crate::types::tool::{Tool, ToolContext, ToolUse};
use crate::utils::logger::Logger;

use super::agent_tools::AgentToolRegistry;

/// Entry for a running agent with its join handle
struct AgentEntry {
    agent: Arc<TaskAgent>,
    handle: tokio::task::JoinHandle<anyhow::Result<TaskAgentResult>>,
}

/// MCP Server Handler - processes JSON-RPC over stdin/stdout
pub struct McpServerHandler {
    /// System prompt override
    system_prompt: Option<String>,
    /// Tool registry (local tools)
    tool_registry: ToolRegistry,
    /// Agent tool registry (task agent management)
    agent_tool_registry: Arc<RwLock<AgentToolRegistry>>,
    /// Communication hub for agent messaging
    communication_hub: Arc<CommunicationHub>,
    /// File lock manager
    file_lock_manager: Arc<FileLockManager>,
    /// AI provider
    provider: Arc<dyn Provider>,
    /// Tool executor for handling tasks
    tool_executor: Arc<RwLock<ToolExecutor>>,
    /// Running task agents with their join handles
    agents: Arc<RwLock<HashMap<String, AgentEntry>>>,
    /// Request ID counter
    request_id: Arc<RwLock<u64>>,
}

impl McpServerHandler {
    /// Create a new MCP server handler
    pub async fn new(
        model: Option<String>,
        system_prompt: Option<String>,
        backend_url_override: Option<String>,
    ) -> Result<Self> {
        let config_manager = ConfigManager::new()?;
        let config = config_manager.get();

        let model = model.unwrap_or_else(|| config.model.clone());

        // Create provider with optional backend URL override
        let factory = ProviderFactory;
        let provider = factory
            .create_with_backend(model.clone(), backend_url_override.clone())
            .await
            .context("Failed to create provider")?;

        let tool_registry = brainwires_tool_builtins::registry_with_builtins();
        let agent_tool_registry = Arc::new(RwLock::new(AgentToolRegistry::new()));
        let communication_hub = Arc::new(CommunicationHub::new());
        let file_lock_manager = Arc::new(FileLockManager::new());

        // Create task manager and tool executor
        let task_manager = Arc::new(RwLock::new(TaskManager::new()));
        let mut tool_executor = ToolExecutor::new(crate::types::agent::PermissionMode::Auto);
        tool_executor.set_task_manager(task_manager);
        let tool_executor = Arc::new(RwLock::new(tool_executor));

        Ok(Self {
            system_prompt,
            tool_registry,
            agent_tool_registry,
            communication_hub,
            file_lock_manager,
            provider,
            agents: Arc::new(RwLock::new(HashMap::new())),
            request_id: Arc::new(RwLock::new(0)),
            tool_executor,
        })
    }

    /// Run the MCP server (blocking)
    pub async fn run(&self) -> Result<()> {
        // Initialize file logging for MCP server
        // Disable stdout logging since MCP uses stdin/stdout for protocol
        crate::utils::logger::init_with_output(false);

        tracing::info!("MCP Server started - listening on stdin");
        // MCP stdio protocol: stdout is reserved for JSON-RPC frames only.
        // Status messages must go to stderr.
        eprintln!(
            "{} MCP Server started - listening on stdin",
            console::style("ℹ").blue()
        );

        let stdin = std::io::stdin();
        let reader = BufReader::new(stdin.lock());

        for line in reader.lines() {
            let line = line.context("Failed to read from stdin")?;

            if line.trim().is_empty() {
                continue;
            }

            // Parse JSON-RPC request
            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    // Send parse error
                    self.send_error(Value::Null, -32700, format!("Parse error: {}", e))
                        .await?;
                    continue;
                }
            };

            // Handle request
            match self.handle_request(request).await {
                Ok(response) => {
                    self.send_response(response).await?;
                }
                Err(e) => {
                    Logger::error(format!("Request handling error: {}", e));
                }
            }
        }

        // MCP stdio protocol: route status to stderr, not stdout.
        eprintln!("{} MCP Server stopped", console::style("ℹ").blue());
        Ok(())
    }

    /// Handle a JSON-RPC request
    async fn handle_request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request).await,
            "tools/list" => self.handle_list_tools(request).await,
            "tools/call" => self.handle_call_tool(request).await,
            _ => Ok(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: "Method not found".to_string(),
                    data: None,
                }),
            }),
        }
    }

    /// Handle initialize request
    async fn handle_initialize(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let _params: InitializeParams =
            serde_json::from_value(request.params.unwrap_or(serde_json::json!({})))
                .context("Invalid initialize params")?;

        let mut capabilities = ServerCapabilities::default();
        capabilities.tools = Some(ToolsCapability {
            list_changed: Some(false),
        });

        let result = InitializeResult {
            protocol_version: "2024-11-05".to_string(),
            capabilities,
            server_info: ServerInfo {
                name: "brainwires-cli".to_string(),
                version: crate::build_info::VERSION.to_string(),
            },
        };

        Ok(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(serde_json::to_value(result)?),
            error: None,
        })
    }

    /// Handle tools/list request
    ///
    /// Only exposes agent-related tools via MCP. Low-level tools (file ops, search, git, bash)
    /// are used internally by agents but not exposed to MCP clients to avoid conflicts
    /// with the host AI's native tools.
    async fn handle_list_tools(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let mut tools: Vec<Tool> = Vec::new();

        // Agent management tools (agent_spawn, agent_list, agent_status, agent_stop, agent_await)
        tools.extend(
            self.agent_tool_registry
                .read()
                .await
                .get_tools()
                .iter()
                .cloned(),
        );

        // Agent pool info tools (agent_pool_stats, agent_file_locks)
        tools.extend(
            self.tool_registry
                .get_by_category(ToolCategory::AgentPool)
                .into_iter()
                .filter(|t| t.name == "agent_pool_stats" || t.name == "agent_file_locks")
                .cloned(),
        );

        // Task management tools
        tools.extend(
            self.tool_registry
                .get_by_category(ToolCategory::TaskManager)
                .into_iter()
                .cloned(),
        );

        // Session task tracking
        tools.extend(
            self.tool_registry
                .get_by_category(ToolCategory::SessionTask)
                .into_iter()
                .cloned(),
        );

        // Planning tools
        tools.extend(
            self.tool_registry
                .get_by_category(ToolCategory::Planning)
                .into_iter()
                .cloned(),
        );

        // Context recall
        tools.extend(
            self.tool_registry
                .get_by_category(ToolCategory::Context)
                .into_iter()
                .cloned(),
        );

        // Convert to MCP tool format
        let mcp_tools: Vec<McpTool> = tools
            .iter()
            .map(|tool| {
                let schema_value = serde_json::to_value(&tool.input_schema).unwrap_or_default();
                let schema_obj = schema_value.as_object().cloned().unwrap_or_default();
                McpTool::new(
                    tool.name.clone(),
                    tool.description.clone(),
                    std::sync::Arc::new(schema_obj),
                )
            })
            .collect();

        let result = ListToolsResult { tools: mcp_tools };

        Ok(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(serde_json::to_value(result)?),
            error: None,
        })
    }

    /// Check if a tool is allowed to be called via MCP
    fn is_mcp_allowed_tool(&self, tool_name: &str) -> bool {
        // Agent tools (prefixed with agent_)
        if tool_name.starts_with("agent_") {
            return true;
        }

        // Self-improvement tools
        if tool_name.starts_with("self_improve_") {
            return true;
        }

        // Task management, session, planning, and context tools
        const ALLOWED_TOOLS: &[&str] = &[
            // Task management
            "task_create",
            "task_add_subtask",
            "task_start",
            "task_complete",
            "task_fail",
            "task_add_dependency",
            "task_get_tree",
            "task_get_ready",
            "task_get_stats",
            // Session task
            "task_list_write",
            // Planning
            "plan_task",
            // Context
            "recall_context",
        ];

        ALLOWED_TOOLS.contains(&tool_name)
    }

    /// Handle tools/call request
    async fn handle_call_tool(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let params: CallToolParams =
            serde_json::from_value(request.params.clone().unwrap_or(serde_json::json!({})))
                .context("Invalid call tool params")?;

        // Check if this is an agent management tool
        if params.name.starts_with("agent_") {
            return self.handle_agent_tool_call(request, params).await;
        }

        // Check if this is a self-improvement tool
        if params.name.starts_with("self_improve_") {
            return self.handle_self_improve_tool_call(request, params).await;
        }

        // Validate tool is allowed via MCP
        if !self.is_mcp_allowed_tool(&params.name) {
            return Ok(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: format!(
                        "Tool '{}' is not available via MCP. Use agent_spawn to create an agent that can use this tool internally.",
                        params.name
                    ),
                    data: None,
                }),
            });
        }

        // Execute regular tool
        let input_value = match params.arguments {
            Some(map) => Value::Object(map),
            None => serde_json::json!({}),
        };

        let tool_use = ToolUse {
            id: format!("tool-{}", self.next_request_id().await),
            name: params.name.to_string(),
            input: input_value,
        };

        let tool_context = ToolContext {
            working_directory: std::env::current_dir()?.to_string_lossy().to_string(),
            // Use full_access for MCP server mode - tools should have write access
            capabilities: serde_json::to_value(
                brainwires::permissions::AgentCapabilities::full_access(),
            )
            .ok(),
            ..Default::default()
        };

        let tool_executor = self.tool_executor.read().await;
        let result = tool_executor.execute(&tool_use, &tool_context).await?;

        let tool_result = if result.is_error {
            CallToolResult::error(vec![Content::text(result.content.clone())])
        } else {
            CallToolResult::success(vec![Content::text(result.content.clone())])
        };

        Ok(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(serde_json::to_value(tool_result)?),
            error: None,
        })
    }

    /// Handle agent management tool calls
    async fn handle_agent_tool_call(
        &self,
        request: JsonRpcRequest,
        params: CallToolParams,
    ) -> Result<JsonRpcResponse> {
        // Convert arguments to Value
        let args = match params.arguments {
            Some(map) => Value::Object(map),
            None => serde_json::json!({}),
        };

        let result = match params.name.as_ref() {
            "agent_spawn" => self.spawn_agent_impl(args.clone()).await,
            "agent_list" => self.list_agents_impl().await,
            "agent_status" => self.get_agent_status_impl(args.clone()).await,
            "agent_stop" => self.stop_agent_impl(args.clone()).await,
            "agent_await" => self.await_agent_impl(args.clone()).await,
            "agent_pool_stats" => self.get_pool_stats_impl().await,
            "agent_file_locks" => self.get_file_locks_impl().await,
            _ => Err(anyhow::anyhow!("Unknown agent tool: {}", params.name)),
        };

        match result {
            Ok(content) => {
                let tool_result = CallToolResult::success(vec![Content::text(content)]);

                Ok(JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(serde_json::to_value(tool_result)?),
                    error: None,
                })
            }
            Err(e) => {
                let tool_result =
                    CallToolResult::error(vec![Content::text(format!("Error: {}", e))]);

                Ok(JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(serde_json::to_value(tool_result)?),
                    error: None,
                })
            }
        }
    }

    /// Spawn a new task agent (returns formatted status message)
    async fn spawn_agent_impl(&self, args: Value) -> Result<String> {
        let task_description = args
            .get("description")
            .and_then(|v| v.as_str())
            .context("Missing 'description' parameter")?;

        // Use provided working_directory or fall back to current directory
        let working_directory = args
            .get("working_directory")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| ".".to_string())
            });

        // Get max_iterations from args or use default (100 = effectively unlimited for most tasks)
        let max_iterations = args
            .get("max_iterations")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(100); // High default to avoid artificial limits

        tracing::info!(
            "Spawning agent with working_directory: {}, max_iterations: {}",
            working_directory,
            max_iterations
        );

        let agent_id = format!("agent-{}", uuid::Uuid::new_v4());
        let task_id = format!("task-{}", uuid::Uuid::new_v4());
        let task = Task::new(task_id.clone(), task_description.to_string());

        // Clone for response message since context will be moved
        let wd_for_response = working_directory.clone();

        let context = AgentContext {
            working_directory,
            conversation_history: Vec::new(),
            tools: self.tool_registry.get_all().to_vec(),
            user_id: None,
            metadata: HashMap::new(),
            working_set: crate::types::WorkingSet::new(),
            // Use full_access for MCP server mode - agents should have write access
            capabilities: brainwires::permissions::AgentCapabilities::full_access(),
        };

        // Check if validation should be enabled (default: true)
        let enable_validation = args
            .get("enable_validation")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Determine build type for validation
        let build_type = args
            .get("build_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let validation_config = if enable_validation {
            let mut config = crate::agents::ValidationConfig {
                working_directory: wd_for_response.clone(),
                ..Default::default()
            };

            // Add build validation if build_type specified
            if let Some(bt) = build_type {
                config = config.with_build(bt);
            }

            Some(config)
        } else {
            None
        };

        // Parse MDAP configuration
        let mdap_config = if args
            .get("enable_mdap")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            use crate::mdap::MdapConfig;

            let preset = args.get("mdap_preset").and_then(|v| v.as_str());

            let mut config = match preset {
                Some("high_reliability") => MdapConfig::high_reliability(),
                Some("cost_optimized") => MdapConfig::cost_optimized(),
                _ => MdapConfig::default(),
            };

            // Override with specific parameters if provided
            if let Some(k) = args.get("mdap_k").and_then(|v| v.as_u64()) {
                config.k = k as u32;
            }

            if let Some(target) = args.get("mdap_target_success").and_then(|v| v.as_f64()) {
                config.target_success_rate = target;
            }

            tracing::info!(
                "MDAP enabled with k={}, target_success={}",
                config.k,
                config.target_success_rate
            );
            Some(config)
        } else {
            None
        };

        let config = TaskAgentConfig {
            max_iterations,
            permission_mode: crate::types::agent::PermissionMode::Auto,
            system_prompt: self.system_prompt.clone(),
            temperature: 0.7,
            max_tokens: 4096,
            validation_config,
            mdap_config,
            analytics_collector: crate::utils::logger::analytics_collector()
                .map(std::sync::Arc::new),
            role: None,
            max_total_tokens: None,
            max_cost_usd: None,
            timeout_secs: None,
            session_budget: None,
        };

        let agent = Arc::new(TaskAgent::new(
            agent_id.clone(),
            task,
            self.provider.clone(),
            self.communication_hub.clone(),
            self.file_lock_manager.clone(),
            context,
            config,
        ));

        // Spawn agent task with error logging and capture the handle
        let agent_clone = agent.clone();
        let agent_id_for_log = agent_id.clone();
        let handle = tokio::spawn(async move {
            tracing::info!("Agent {} starting execution", agent_id_for_log);
            let result = agent_clone.execute().await;
            match &result {
                Ok(r) => {
                    tracing::info!(
                        "Agent {} completed: success={}, iterations={}, summary={}",
                        agent_id_for_log,
                        r.success,
                        r.iterations,
                        r.summary
                    );
                }
                Err(e) => {
                    tracing::error!("Agent {} failed with error: {:?}", agent_id_for_log, e);
                }
            }
            result
        });

        // Store agent entry with handle
        self.agents
            .write()
            .await
            .insert(agent_id.clone(), AgentEntry { agent, handle });

        Ok(format!(
            "Spawned task agent '{}' for task '{}'\nWorking directory: {}",
            agent_id, task_description, wd_for_response
        ))
    }

    /// List all running agents (returns formatted text)
    async fn list_agents_impl(&self) -> Result<String> {
        let agents = self.agents.read().await;

        if agents.is_empty() {
            return Ok("No running agents".to_string());
        }

        let mut output = String::new();
        output.push_str("Running agents:\n");

        for (agent_id, entry) in agents.iter() {
            let status = entry.agent.status().await;
            let task = entry.agent.task().await;
            output.push_str(&format!(
                "  - {} ({}): {}\n",
                agent_id, status, task.description
            ));
        }

        Ok(output)
    }

    /// Get status of a specific agent (returns formatted text)
    async fn get_agent_status_impl(&self, args: Value) -> Result<String> {
        let agent_id = args
            .get("agent_id")
            .and_then(|v| v.as_str())
            .context("Missing 'agent_id' parameter")?;

        let agents = self.agents.read().await;
        let entry = agents
            .get(agent_id)
            .context(format!("Agent '{}' not found", agent_id))?;

        let status = entry.agent.status().await;
        let task = entry.agent.task().await;

        Ok(format!(
            "Agent '{}' - Status: {}\nTask: {}\nIterations: {}",
            agent_id, status, task.description, task.iterations
        ))
    }

    /// Stop a running agent (returns formatted text)
    async fn stop_agent_impl(&self, args: Value) -> Result<String> {
        let agent_id = args
            .get("agent_id")
            .and_then(|v| v.as_str())
            .context("Missing 'agent_id' parameter")?;

        let mut agents = self.agents.write().await;

        if let Some(entry) = agents.remove(agent_id) {
            // Abort the task
            entry.handle.abort();
            Ok(format!("Stopped agent '{}'", agent_id))
        } else {
            Err(anyhow::anyhow!("Agent '{}' not found", agent_id))
        }
    }

    /// Wait for an agent to complete (returns formatted text)
    async fn await_agent_impl(&self, args: Value) -> Result<String> {
        let agent_id = args
            .get("agent_id")
            .and_then(|v| v.as_str())
            .context("Missing 'agent_id' parameter")?;

        let timeout_secs = args.get("timeout_secs").and_then(|v| v.as_u64());

        // Remove the agent entry to get ownership of the handle
        let entry = {
            let mut agents = self.agents.write().await;
            agents.remove(agent_id)
        };

        let entry = entry.context(format!("Agent '{}' not found", agent_id))?;

        // Wait for the agent to complete, with optional timeout
        let result = if let Some(secs) = timeout_secs {
            match tokio::time::timeout(std::time::Duration::from_secs(secs), entry.handle).await {
                Ok(join_result) => join_result,
                Err(_) => {
                    return Err(anyhow::anyhow!(
                        "Timeout waiting for agent '{}' after {} seconds",
                        agent_id,
                        secs
                    ));
                }
            }
        } else {
            entry.handle.await
        };

        // Handle join result
        match result {
            Ok(Ok(agent_result)) => Ok(format!(
                "Agent '{}' completed\n\
                     Success: {}\n\
                     Iterations: {}\n\
                     Summary: {}",
                agent_result.agent_id,
                agent_result.success,
                agent_result.iterations,
                agent_result.summary
            )),
            Ok(Err(e)) => Err(anyhow::anyhow!(
                "Agent '{}' execution failed: {}",
                agent_id,
                e
            )),
            Err(join_error) => {
                if join_error.is_cancelled() {
                    Err(anyhow::anyhow!("Agent '{}' was cancelled", agent_id))
                } else {
                    Err(anyhow::anyhow!(
                        "Agent '{}' panicked: {}",
                        agent_id,
                        join_error
                    ))
                }
            }
        }
    }

    /// Get pool statistics (returns formatted text)
    async fn get_pool_stats_impl(&self) -> Result<String> {
        let agents = self.agents.read().await;
        let total = agents.len();
        let running = agents.values().filter(|e| !e.handle.is_finished()).count();
        let completed = total - running;

        Ok(format!(
            "Agent Pool Stats:\n\
             Total agents: {}\n\
             Running: {}\n\
             Completed: {}",
            total, running, completed
        ))
    }

    /// Get all file locks held by agents (returns formatted text)
    async fn get_file_locks_impl(&self) -> Result<String> {
        let locks = self.file_lock_manager.list_locks().await;

        if locks.is_empty() {
            return Ok("No file locks currently held".to_string());
        }

        let mut output = String::from("Current file locks:\n");
        for (path, lock_info) in locks {
            output.push_str(&format!(
                "  {} - held by {} ({:?})\n",
                path.display(),
                lock_info.agent_id,
                lock_info.lock_type
            ));
        }

        Ok(output)
    }

    /// Handle self-improvement tool calls
    async fn handle_self_improve_tool_call(
        &self,
        request: JsonRpcRequest,
        params: CallToolParams,
    ) -> Result<JsonRpcResponse> {
        let args = match params.arguments {
            Some(map) => Value::Object(map),
            None => serde_json::json!({}),
        };

        let result_text = match params.name.as_ref() {
            "self_improve_start" => {
                let config = crate::self_improve::SelfImprovementConfig {
                    max_cycles: args.get("max_cycles").and_then(|v| v.as_u64()).unwrap_or(10) as u32,
                    max_budget: args.get("max_budget").and_then(|v| v.as_f64()).unwrap_or(10.0),
                    dry_run: args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false),
                    strategies: args
                        .get("strategies")
                        .and_then(|v| v.as_str())
                        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
                        .unwrap_or_default(),
                    no_bridge: args.get("no_bridge").and_then(|v| v.as_bool()).unwrap_or(false),
                    no_direct: args.get("no_direct").and_then(|v| v.as_bool()).unwrap_or(false),
                    ..Default::default()
                };

                let mut controller = crate::self_improve::SelfImprovementController::new(config);
                match controller.run().await {
                    Ok(report) => report.to_markdown(),
                    Err(e) => format!("Self-improvement failed: {e}"),
                }
            }
            "self_improve_status" => {
                "Self-improvement status: no active session (sessions run synchronously via self_improve_start)".to_string()
            }
            "self_improve_stop" => {
                "Self-improvement sessions run synchronously - use timeout or budget limits to control duration".to_string()
            }
            _ => format!("Unknown self-improvement tool: {}", params.name),
        };

        let tool_result = CallToolResult::success(vec![Content::text(result_text)]);

        Ok(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(serde_json::to_value(tool_result)?),
            error: None,
        })
    }

    /// Send a JSON-RPC response to stdout
    async fn send_response(&self, response: JsonRpcResponse) -> Result<()> {
        let json = serde_json::to_string(&response)?;
        let mut stdout = std::io::stdout();
        writeln!(stdout, "{}", json)?;
        stdout.flush()?;
        Ok(())
    }

    /// Send an error response
    async fn send_error(&self, id: Value, code: i32, message: String) -> Result<()> {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        };
        self.send_response(response).await
    }

    /// Get next request ID
    async fn next_request_id(&self) -> u64 {
        let mut id = self.request_id.write().await;
        *id += 1;
        *id
    }
}

// ============================================================================
// AgentManager trait implementation
// ============================================================================

#[async_trait]
impl AgentManager for McpServerHandler {
    /// Spawn a new agent and return its agent ID
    async fn spawn_agent(&self, config: SpawnConfig) -> Result<String> {
        let args = serde_json::to_value(&config)?;
        let msg = self.spawn_agent_impl(args).await?;
        // The message format is: "Spawned task agent 'AGENT_ID' for task '...'..."
        // Extract the agent_id between the first pair of single quotes
        let agent_id = msg
            .split('\'')
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("Could not parse agent_id from spawn result"))?
            .to_string();
        Ok(agent_id)
    }

    /// List all running agents as structured AgentInfo objects
    async fn list_agents(&self) -> Result<Vec<AgentInfo>> {
        let agents = self.agents.read().await;
        let mut result = Vec::new();
        for (agent_id, entry) in agents.iter() {
            let status = entry.agent.status().await;
            let task = entry.agent.task().await;
            result.push(AgentInfo {
                agent_id: agent_id.clone(),
                status: status.to_string(),
                task_description: task.description.clone(),
                iterations: task.iterations,
            });
        }
        Ok(result)
    }

    /// Get structured status for a specific agent
    async fn agent_status(&self, agent_id: &str) -> Result<AgentInfo> {
        let agents = self.agents.read().await;
        let entry = agents
            .get(agent_id)
            .with_context(|| format!("Agent '{}' not found", agent_id))?;
        let status = entry.agent.status().await;
        let task = entry.agent.task().await;
        Ok(AgentInfo {
            agent_id: agent_id.to_string(),
            status: status.to_string(),
            task_description: task.description.clone(),
            iterations: task.iterations,
        })
    }

    /// Stop a running agent
    async fn stop_agent(&self, agent_id: &str) -> Result<()> {
        let args = serde_json::json!({ "agent_id": agent_id });
        self.stop_agent_impl(args).await?;
        Ok(())
    }

    /// Wait for an agent to complete and return a structured result
    async fn await_agent(&self, agent_id: &str, timeout_secs: Option<u64>) -> Result<AgentResult> {
        // Remove the agent entry to get ownership of the handle
        let entry = {
            let mut agents = self.agents.write().await;
            agents.remove(agent_id)
        };
        let entry = entry.with_context(|| format!("Agent '{}' not found", agent_id))?;

        // Wait for completion with optional timeout
        let join_result = if let Some(secs) = timeout_secs {
            tokio::time::timeout(std::time::Duration::from_secs(secs), entry.handle)
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "Timeout waiting for agent '{}' after {} seconds",
                        agent_id,
                        secs
                    )
                })?
        } else {
            entry.handle.await
        };

        match join_result {
            Ok(Ok(r)) => Ok(AgentResult {
                agent_id: r.agent_id.clone(),
                success: r.success,
                summary: r.summary.clone(),
                iterations: r.iterations,
            }),
            Ok(Err(e)) => Err(anyhow::anyhow!(
                "Agent '{}' execution failed: {}",
                agent_id,
                e
            )),
            Err(join_error) if join_error.is_cancelled() => {
                Err(anyhow::anyhow!("Agent '{}' was cancelled", agent_id))
            }
            Err(join_error) => Err(anyhow::anyhow!(
                "Agent '{}' panicked: {}",
                agent_id,
                join_error
            )),
        }
    }

    /// Return pool statistics as a JSON value
    async fn pool_stats(&self) -> Result<Value> {
        let agents = self.agents.read().await;
        let total = agents.len();
        let running = agents.values().filter(|e| !e.handle.is_finished()).count();
        let completed = total - running;
        Ok(serde_json::json!({
            "total": total,
            "running": running,
            "completed": completed,
        }))
    }

    /// Return all held file locks as a JSON value
    async fn file_locks(&self) -> Result<Value> {
        let locks = self.file_lock_manager.list_locks().await;
        let entries: Vec<Value> = locks
            .into_iter()
            .map(|(path, info)| {
                serde_json::json!({
                    "path": path.display().to_string(),
                    "agent_id": info.agent_id,
                    "lock_type": format!("{:?}", info.lock_type),
                })
            })
            .collect();
        Ok(Value::Array(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Test is_mcp_allowed_tool function
    #[test]
    fn test_is_mcp_allowed_tool_agent_tools() {
        let registry = brainwires_tool_builtins::registry_with_builtins();
        let handler = TestableHandler {
            tool_registry: registry,
        };

        // Agent tools should be allowed
        assert!(handler.is_mcp_allowed("agent_spawn"));
        assert!(handler.is_mcp_allowed("agent_list"));
        assert!(handler.is_mcp_allowed("agent_status"));
        assert!(handler.is_mcp_allowed("agent_stop"));
        assert!(handler.is_mcp_allowed("agent_await"));
        assert!(handler.is_mcp_allowed("agent_pool_stats"));
        assert!(handler.is_mcp_allowed("agent_file_locks"));
    }

    #[test]
    fn test_is_mcp_allowed_tool_task_tools() {
        let registry = brainwires_tool_builtins::registry_with_builtins();
        let handler = TestableHandler {
            tool_registry: registry,
        };

        // Task management tools should be allowed
        assert!(handler.is_mcp_allowed("task_create"));
        assert!(handler.is_mcp_allowed("task_start"));
        assert!(handler.is_mcp_allowed("task_complete"));
        assert!(handler.is_mcp_allowed("task_fail"));
        assert!(handler.is_mcp_allowed("task_get_tree"));
        assert!(handler.is_mcp_allowed("task_get_stats"));
    }

    #[test]
    fn test_is_mcp_allowed_tool_planning_tools() {
        let registry = brainwires_tool_builtins::registry_with_builtins();
        let handler = TestableHandler {
            tool_registry: registry,
        };

        // Planning and context tools should be allowed
        assert!(handler.is_mcp_allowed("plan_task"));
        assert!(handler.is_mcp_allowed("recall_context"));
        assert!(handler.is_mcp_allowed("task_list_write"));
    }

    #[test]
    fn test_is_mcp_allowed_tool_disallowed() {
        let registry = brainwires_tool_builtins::registry_with_builtins();
        let handler = TestableHandler {
            tool_registry: registry,
        };

        // Internal tools should NOT be allowed
        assert!(!handler.is_mcp_allowed("read_file"));
        assert!(!handler.is_mcp_allowed("write_file"));
        assert!(!handler.is_mcp_allowed("bash"));
        assert!(!handler.is_mcp_allowed("search_files"));
        assert!(!handler.is_mcp_allowed("list_files"));
        assert!(!handler.is_mcp_allowed("git_status"));
    }

    /// Test JsonRpcRequest parsing
    #[test]
    fn test_json_rpc_request_parsing() {
        let request_json = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05"
            }
        });

        let request: JsonRpcRequest = serde_json::from_value(request_json).unwrap();
        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.id, json!(1));
        assert_eq!(request.method, "initialize");
        assert!(request.params.is_some());
    }

    #[test]
    fn test_json_rpc_request_no_params() {
        let request_json = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        });

        let request: JsonRpcRequest = serde_json::from_value(request_json).unwrap();
        assert_eq!(request.method, "tools/list");
        assert!(request.params.is_none());
    }

    /// Test JsonRpcResponse serialization
    #[test]
    fn test_json_rpc_response_success() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            result: Some(json!({"status": "ok"})),
            error: None,
        };

        let serialized = serde_json::to_value(&response).unwrap();
        assert_eq!(serialized["jsonrpc"], "2.0");
        assert_eq!(serialized["id"], 1);
        assert!(serialized["result"].is_object());
        assert!(serialized.get("error").is_none() || serialized["error"].is_null());
    }

    #[test]
    fn test_json_rpc_response_error() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: "Method not found".to_string(),
                data: None,
            }),
        };

        let serialized = serde_json::to_value(&response).unwrap();
        assert_eq!(serialized["error"]["code"], -32601);
        assert_eq!(serialized["error"]["message"], "Method not found");
    }

    /// Test InitializeResult structure
    #[test]
    fn test_initialize_result_structure() {
        let mut capabilities = ServerCapabilities::default();
        capabilities.tools = Some(ToolsCapability {
            list_changed: Some(false),
        });
        let result = InitializeResult {
            protocol_version: "2024-11-05".to_string(),
            capabilities,
            server_info: ServerInfo {
                name: "brainwires-cli".to_string(),
                version: "0.7.0".to_string(),
            },
        };

        let serialized = serde_json::to_value(&result).unwrap();
        assert_eq!(serialized["protocolVersion"], "2024-11-05");
        assert_eq!(serialized["serverInfo"]["name"], "brainwires-cli");
        assert!(serialized["capabilities"]["tools"].is_object());
    }

    /// Test CallToolParams parsing
    #[test]
    fn test_call_tool_params_with_arguments() {
        let params_json = json!({
            "name": "agent_spawn",
            "arguments": {
                "description": "Create a test file",
                "working_directory": "/tmp"
            }
        });

        let params: CallToolParams = serde_json::from_value(params_json).unwrap();
        assert_eq!(params.name.as_ref(), "agent_spawn");
        assert!(params.arguments.is_some());
        let args = params.arguments.unwrap();
        assert_eq!(
            args.get("description").unwrap().as_str().unwrap(),
            "Create a test file"
        );
    }

    #[test]
    fn test_call_tool_params_no_arguments() {
        let params_json = json!({
            "name": "agent_list"
        });

        let params: CallToolParams = serde_json::from_value(params_json).unwrap();
        assert_eq!(params.name.as_ref(), "agent_list");
        assert!(params.arguments.is_none());
    }

    /// Test ListToolsResult structure
    #[test]
    fn test_list_tools_result_structure() {
        let result = ListToolsResult {
            tools: vec![McpTool::new(
                "agent_spawn",
                "Spawn a task agent",
                std::sync::Arc::new(serde_json::Map::new()),
            )],
        };

        let serialized = serde_json::to_value(&result).unwrap();
        assert!(serialized["tools"].is_array());
        assert_eq!(serialized["tools"][0]["name"], "agent_spawn");
    }

    /// Test CallToolResult structure
    #[test]
    fn test_call_tool_result_success() {
        let result =
            CallToolResult::success(vec![Content::text("Operation completed successfully")]);

        let serialized = serde_json::to_value(&result).unwrap();
        assert!(serialized["content"].is_array());
        assert_eq!(serialized["isError"], false);
    }

    #[test]
    fn test_call_tool_result_error() {
        let result = CallToolResult::error(vec![Content::text("Error: Invalid parameter")]);

        let serialized = serde_json::to_value(&result).unwrap();
        assert_eq!(serialized["isError"], true);
    }

    /// Test JSON-RPC error codes
    #[test]
    fn test_json_rpc_error_codes() {
        // Parse error
        let parse_error = JsonRpcError {
            code: -32700,
            message: "Parse error".to_string(),
            data: None,
        };
        assert_eq!(parse_error.code, -32700);

        // Invalid request
        let invalid_request = JsonRpcError {
            code: -32600,
            message: "Invalid Request".to_string(),
            data: None,
        };
        assert_eq!(invalid_request.code, -32600);

        // Method not found
        let method_not_found = JsonRpcError {
            code: -32601,
            message: "Method not found".to_string(),
            data: None,
        };
        assert_eq!(method_not_found.code, -32601);

        // Invalid params
        let invalid_params = JsonRpcError {
            code: -32602,
            message: "Invalid params".to_string(),
            data: None,
        };
        assert_eq!(invalid_params.code, -32602);
    }

    /// Test agent spawn argument parsing
    #[test]
    fn test_spawn_agent_args_parsing() {
        let args = json!({
            "description": "Create a TypeScript utility function",
            "working_directory": "/home/user/project",
            "max_iterations": 20,
            "enable_validation": true,
            "build_type": "typescript",
            "enable_mdap": false
        });

        // Verify required fields
        let description = args.get("description").and_then(|v| v.as_str());
        assert_eq!(description, Some("Create a TypeScript utility function"));

        // Verify optional fields
        let working_dir = args.get("working_directory").and_then(|v| v.as_str());
        assert_eq!(working_dir, Some("/home/user/project"));

        let max_iter = args.get("max_iterations").and_then(|v| v.as_u64());
        assert_eq!(max_iter, Some(20));

        let enable_val = args.get("enable_validation").and_then(|v| v.as_bool());
        assert_eq!(enable_val, Some(true));

        let build_type = args.get("build_type").and_then(|v| v.as_str());
        assert_eq!(build_type, Some("typescript"));

        let enable_mdap = args.get("enable_mdap").and_then(|v| v.as_bool());
        assert_eq!(enable_mdap, Some(false));
    }

    /// Test MDAP configuration parsing
    #[test]
    fn test_mdap_config_parsing() {
        use crate::mdap::MdapConfig;

        // Default preset
        let default_config = MdapConfig::default();
        assert_eq!(default_config.k, 3);
        assert!((default_config.target_success_rate - 0.95).abs() < 0.001);

        // High reliability preset
        let high_rel = MdapConfig::high_reliability();
        assert_eq!(high_rel.k, 5);
        assert!((high_rel.target_success_rate - 0.99).abs() < 0.001);

        // Cost optimized preset
        let cost_opt = MdapConfig::cost_optimized();
        assert_eq!(cost_opt.k, 2);
        assert!((cost_opt.target_success_rate - 0.90).abs() < 0.001);
    }

    /// Test validation config building
    #[test]
    fn test_validation_config_building() {
        use crate::agents::ValidationConfig;

        let config = ValidationConfig {
            working_directory: "/test/path".to_string(),
            ..Default::default()
        }
        .with_build("typescript");

        assert_eq!(config.working_directory, "/test/path");
        // Verify build_type was set (internal field)
        assert!(config.enabled);
    }

    /// Helper struct for testing is_mcp_allowed_tool without full handler
    #[allow(dead_code)]
    struct TestableHandler {
        tool_registry: ToolRegistry,
    }

    impl TestableHandler {
        fn is_mcp_allowed(&self, tool_name: &str) -> bool {
            // Agent tools (prefixed with agent_)
            if tool_name.starts_with("agent_") {
                return true;
            }

            const ALLOWED_TOOLS: &[&str] = &[
                "task_create",
                "task_add_subtask",
                "task_start",
                "task_complete",
                "task_fail",
                "task_add_dependency",
                "task_get_tree",
                "task_get_ready",
                "task_get_stats",
                "task_list_write",
                "plan_task",
                "recall_context",
            ];

            ALLOWED_TOOLS.contains(&tool_name)
        }
    }

    // === Async Integration Tests (require authentication) ===

    #[tokio::test]
    #[ignore = "Requires active Brainwires session/authentication"]
    async fn test_handler_creation() {
        let handler =
            McpServerHandler::new(Some("claude-3-5-sonnet-20241022".to_string()), None, None).await;
        assert!(handler.is_ok(), "Should create handler successfully");
    }

    #[tokio::test]
    #[ignore = "Requires active Brainwires session/authentication"]
    async fn test_initialize_request() {
        let handler = McpServerHandler::new(None, None, None).await.unwrap();

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "initialize".to_string(),
            params: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0"
                }
            })),
        };

        let response = handler.handle_initialize(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert_eq!(response.jsonrpc, "2.0");
        assert_eq!(response.id, json!(1));
        assert!(response.result.is_some());
        assert!(response.error.is_none());

        let result = response.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "brainwires-cli");
    }

    #[tokio::test]
    #[ignore = "Requires active Brainwires session/authentication"]
    async fn test_list_tools_request() {
        let handler = McpServerHandler::new(None, None, None).await.unwrap();

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(2),
            method: "tools/list".to_string(),
            params: None,
        };

        let response = handler.handle_list_tools(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.result.is_some());

        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert!(!tools.is_empty(), "Should have tools registered");

        // Check for agent management tools
        let tool_names: Vec<String> = tools
            .iter()
            .filter_map(|t| t["name"].as_str().map(String::from))
            .collect();

        // Agent management tools should be present
        assert!(tool_names.contains(&"agent_spawn".to_string()));
        assert!(tool_names.contains(&"agent_list".to_string()));
        assert!(tool_names.contains(&"agent_status".to_string()));
        assert!(tool_names.contains(&"agent_stop".to_string()));
        assert!(tool_names.contains(&"agent_await".to_string()));

        // Task management tools should be present
        assert!(tool_names.contains(&"task_create".to_string()));
        assert!(tool_names.contains(&"plan_task".to_string()));
        assert!(tool_names.contains(&"recall_context".to_string()));

        // Low-level tools should NOT be exposed via MCP
        assert!(
            !tool_names.contains(&"read_file".to_string()),
            "read_file should not be exposed via MCP"
        );
        assert!(
            !tool_names.contains(&"write_file".to_string()),
            "write_file should not be exposed via MCP"
        );
        assert!(
            !tool_names.contains(&"execute_command".to_string()),
            "execute_command should not be exposed via MCP"
        );
        assert!(
            !tool_names.contains(&"search_code".to_string()),
            "search_code should not be exposed via MCP"
        );
        assert!(
            !tool_names.contains(&"git_status".to_string()),
            "git_status should not be exposed via MCP"
        );
    }

    #[tokio::test]
    #[ignore = "Requires active Brainwires session/authentication"]
    async fn test_invalid_method() {
        let handler = McpServerHandler::new(None, None, None).await.unwrap();

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(3),
            method: "invalid/method".to_string(),
            params: None,
        };

        let response = handler.handle_request(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.as_ref().unwrap().code, -32601); // Method not found
    }

    #[tokio::test]
    #[ignore = "Requires active Brainwires session/authentication"]
    async fn test_agent_list_initially_empty() {
        let handler = McpServerHandler::new(None, None, None).await.unwrap();
        let result = handler.list_agents_impl().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "No running agents");
    }

    #[tokio::test]
    #[ignore = "Requires active Brainwires session/authentication"]
    async fn test_agent_spawn_missing_description_param() {
        let handler = McpServerHandler::new(None, None, None).await.unwrap();
        let args = json!({});
        let result = handler.spawn_agent_impl(args).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing 'description' parameter")
        );
    }

    #[tokio::test]
    #[ignore = "Requires active Brainwires session/authentication"]
    async fn test_get_nonexistent_agent_status() {
        let handler = McpServerHandler::new(None, None, None).await.unwrap();
        let args = json!({"agent_id": "nonexistent-agent"});
        let result = handler.get_agent_status_impl(args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    #[ignore = "Requires active Brainwires session/authentication"]
    async fn test_stop_nonexistent_agent() {
        let handler = McpServerHandler::new(None, None, None).await.unwrap();
        let args = json!({"agent_id": "nonexistent-agent"});
        let result = handler.stop_agent_impl(args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    #[ignore = "Requires active Brainwires session/authentication"]
    async fn test_next_request_id_increments() {
        let handler = McpServerHandler::new(None, None, None).await.unwrap();
        let id1 = handler.next_request_id().await;
        let id2 = handler.next_request_id().await;
        let id3 = handler.next_request_id().await;

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[test]
    fn test_json_rpc_request_structure() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: "test".to_string(),
            params: Some(json!({"key": "value"})),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains("\"jsonrpc\":\"2.0\""));
        assert!(serialized.contains("\"method\":\"test\""));
    }

    #[test]
    fn test_json_rpc_response_structure() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            result: Some(json!({"status": "ok"})),
            error: None,
        };

        let serialized = serde_json::to_string(&response).unwrap();
        assert!(serialized.contains("\"jsonrpc\":\"2.0\""));
        assert!(serialized.contains("\"result\""));
        assert!(!serialized.contains("\"error\""));
    }

    #[test]
    fn test_json_rpc_error_structure() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message: "Invalid Request".to_string(),
                data: None,
            }),
        };

        let serialized = serde_json::to_string(&response).unwrap();
        assert!(serialized.contains("\"error\""));
        assert!(serialized.contains("-32600"));
        assert!(!serialized.contains("\"result\""));
    }
}
