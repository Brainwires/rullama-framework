use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::AgentNetworkClient;
use super::error::AgentNetworkClientError;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
/// Configuration for spawning an agent.
pub struct AgentConfig {
    /// Maximum number of iterations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<u32>,
    /// Whether to enable validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_validation: Option<bool>,
    /// Build system type (e.g. "typescript", "cargo").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_type: Option<String>,
    /// Whether to enable MDAP.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_mdap: Option<bool>,
    /// MDAP preset name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mdap_preset: Option<String>,
}

/// Result of an agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Agent unique identifier.
    pub agent_id: String,
    /// Whether the agent completed successfully.
    pub success: bool,
    /// Number of iterations used.
    pub iterations: u32,
    /// Summary of the result.
    pub summary: String,
    /// Raw output text.
    pub raw_output: String,
}

/// Information about a running or completed agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent unique identifier.
    pub agent_id: String,
    /// Current status.
    pub status: String,
    /// Description of the assigned task.
    pub task_description: String,
}

impl AgentNetworkClient {
    /// Spawn a new agent with the given description and config.
    pub async fn spawn_agent(
        &mut self,
        description: &str,
        working_dir: &str,
        config: AgentConfig,
    ) -> Result<String, AgentNetworkClientError> {
        let mut args = json!({
            "description": description,
            "working_directory": working_dir,
        });

        if let Some(max_iter) = config.max_iterations {
            args["max_iterations"] = json!(max_iter);
        }
        if let Some(enable_val) = config.enable_validation {
            args["enable_validation"] = json!(enable_val);
        }
        if let Some(ref build_type) = config.build_type {
            args["build_type"] = json!(build_type);
        }
        if let Some(enable_mdap) = config.enable_mdap {
            args["enable_mdap"] = json!(enable_mdap);
        }
        if let Some(ref preset) = config.mdap_preset {
            args["mdap_preset"] = json!(preset);
        }

        let result = self.call_tool("agent_spawn", args).await?;

        // Extract agent_id from result
        // The result from the MCP server is typically a CallToolResult with content
        let agent_id = extract_agent_id(&result)?;
        Ok(agent_id)
    }

    /// Wait for an agent to complete, with optional timeout.
    pub async fn await_agent(
        &mut self,
        agent_id: &str,
        timeout_secs: Option<u64>,
    ) -> Result<AgentResult, AgentNetworkClientError> {
        let mut args = json!({ "agent_id": agent_id });
        if let Some(timeout) = timeout_secs {
            args["timeout_secs"] = json!(timeout);
        }

        let result = self.call_tool("agent_await", args).await?;
        parse_agent_result(&result, agent_id)
    }

    /// List all agents.
    pub async fn list_agents(&mut self) -> Result<Vec<AgentInfo>, AgentNetworkClientError> {
        let result = self.call_tool("agent_list", json!({})).await?;
        parse_agent_list(&result)
    }

    /// Stop a running agent by ID.
    pub async fn stop_agent(&mut self, agent_id: &str) -> Result<(), AgentNetworkClientError> {
        self.call_tool("agent_stop", json!({ "agent_id": agent_id }))
            .await?;
        Ok(())
    }

    /// Get the current status of an agent by ID.
    pub async fn get_agent_status(
        &mut self,
        agent_id: &str,
    ) -> Result<AgentInfo, AgentNetworkClientError> {
        let result = self
            .call_tool("agent_status", json!({ "agent_id": agent_id }))
            .await?;
        parse_agent_info(&result, agent_id)
    }
}

fn extract_agent_id(result: &serde_json::Value) -> Result<String, AgentNetworkClientError> {
    // Try to extract from content array (CallToolResult format)
    if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
        for item in content {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                // Parse the text to find agent_id
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text)
                    && let Some(id) = parsed.get("agent_id").and_then(|i| i.as_str())
                {
                    return Ok(id.to_string());
                }
                // Try to find agent_id pattern in text
                if text.contains("agent_id") {
                    // Simple extraction
                    if let Some(start) = text.find("agent_id") {
                        let rest = &text[start..];
                        if let Some(colon) = rest.find(':') {
                            let value_part = rest[colon + 1..].trim();
                            let id = value_part
                                .trim_start_matches('"')
                                .split('"')
                                .next()
                                .unwrap_or(value_part.split_whitespace().next().unwrap_or(""));
                            if !id.is_empty() {
                                return Ok(id.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    // Direct field access
    if let Some(id) = result.get("agent_id").and_then(|i| i.as_str()) {
        return Ok(id.to_string());
    }

    Err(AgentNetworkClientError::Protocol(
        "Could not extract agent_id from spawn result".to_string(),
    ))
}

fn parse_agent_result(
    result: &serde_json::Value,
    agent_id: &str,
) -> Result<AgentResult, AgentNetworkClientError> {
    // Try to parse from content text
    let raw = result.to_string();

    let text = extract_text_content(result).unwrap_or_default();

    // Try JSON parse
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
        return Ok(AgentResult {
            agent_id: parsed
                .get("agent_id")
                .and_then(|i| i.as_str())
                .unwrap_or(agent_id)
                .to_string(),
            success: parsed
                .get("success")
                .and_then(|s| s.as_bool())
                .unwrap_or(false),
            iterations: parsed
                .get("iterations")
                .and_then(|i| i.as_u64())
                .unwrap_or(0) as u32,
            summary: parsed
                .get("summary")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string(),
            raw_output: raw,
        });
    }

    // Fallback
    Ok(AgentResult {
        agent_id: agent_id.to_string(),
        success: text.contains("success") || text.contains("completed"),
        iterations: 0,
        summary: text,
        raw_output: raw,
    })
}

fn parse_agent_list(result: &serde_json::Value) -> Result<Vec<AgentInfo>, AgentNetworkClientError> {
    let text = extract_text_content(result).unwrap_or_default();

    if let Ok(agents) = serde_json::from_str::<Vec<AgentInfo>>(&text) {
        return Ok(agents);
    }

    // Single agent info
    if let Ok(info) = serde_json::from_str::<AgentInfo>(&text) {
        return Ok(vec![info]);
    }

    Ok(Vec::new())
}

fn parse_agent_info(
    result: &serde_json::Value,
    agent_id: &str,
) -> Result<AgentInfo, AgentNetworkClientError> {
    let text = extract_text_content(result).unwrap_or_default();

    if let Ok(info) = serde_json::from_str::<AgentInfo>(&text) {
        return Ok(info);
    }

    Ok(AgentInfo {
        agent_id: agent_id.to_string(),
        status: "unknown".to_string(),
        task_description: text,
    })
}

fn extract_text_content(result: &serde_json::Value) -> Option<String> {
    if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
        for item in content {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                return Some(text.to_string());
            }
        }
    }
    // If result is a string itself
    if let Some(s) = result.as_str() {
        return Some(s.to_string());
    }
    None
}
