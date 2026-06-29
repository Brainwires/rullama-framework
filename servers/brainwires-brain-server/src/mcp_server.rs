use brainwires_knowledge::knowledge::brain_client::BrainClient;
use brainwires_knowledge::knowledge::types::*;

use anyhow::{Context, Result};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler, ServiceExt,
    handler::server::{router::prompt::PromptRouter, tool::ToolRouter, wrapper::Parameters},
    model::*,
    prompt, prompt_handler, prompt_router,
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct BrainMcpServer {
    client: Arc<Mutex<BrainClient>>,
    tool_router: ToolRouter<Self>,
    #[allow(dead_code)]
    prompt_router: PromptRouter<Self>,
}

impl BrainMcpServer {
    /// Create with default paths.
    pub async fn new() -> Result<Self> {
        let client = BrainClient::new().await?;
        Self::with_client(client)
    }

    /// Create from an existing BrainClient.
    pub fn with_client(client: BrainClient) -> Result<Self> {
        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
        })
    }

    /// Serve on stdin/stdout (MCP standard I/O transport).
    pub async fn serve_stdio() -> Result<()> {
        tracing::info!("Starting Open Brain MCP server");

        let server = Self::new()
            .await
            .context("Failed to create Brain MCP server")?;

        let transport = rmcp::transport::io::stdio();

        server.serve(transport).await?.waiting().await?;

        Ok(())
    }
}

// ── Tool definitions ─────────────────────────────────────────────────────

#[tool_router(router = tool_router)]
impl BrainMcpServer {
    #[tool(
        description = "Capture a thought, insight, decision, or note into persistent memory. Automatically detects category, extracts tags, embeds for semantic search, and extracts personal knowledge facts."
    )]
    async fn capture_thought(
        &self,
        Parameters(req): Parameters<CaptureThoughtRequest>,
    ) -> Result<String, String> {
        let mut client = self.client.lock().await;
        let response = client
            .capture_thought(req)
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Semantic search across all memory — thoughts, personal facts, and behavioral knowledge. Returns results ranked by relevance."
    )]
    async fn search_memory(
        &self,
        Parameters(req): Parameters<SearchMemoryRequest>,
    ) -> Result<String, String> {
        let client = self.client.lock().await;
        let response = client
            .search_memory(req)
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Browse recently captured thoughts, optionally filtered by category and time range."
    )]
    async fn list_recent(
        &self,
        Parameters(req): Parameters<ListRecentRequest>,
    ) -> Result<String, String> {
        let client = self.client.lock().await;
        let response = client
            .list_recent(req)
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Retrieve a specific thought by its UUID.")]
    async fn get_thought(
        &self,
        Parameters(req): Parameters<GetThoughtRequest>,
    ) -> Result<String, String> {
        let client = self.client.lock().await;
        let response = client
            .get_thought(&req.id, req.owner_id.as_deref())
            .await
            .map_err(|e| format!("{:#}", e))?;
        match response {
            Some(thought) => serde_json::to_string_pretty(&thought)
                .map_err(|e| format!("Serialization failed: {}", e)),
            None => Ok(format!("{{\"error\": \"Thought not found: {}\"}}", req.id)),
        }
    }

    #[tool(
        description = "Query personal knowledge (PKS) facts and behavioral knowledge (BKS) truths. Filter by source, category, and confidence."
    )]
    async fn search_knowledge(
        &self,
        Parameters(req): Parameters<SearchKnowledgeRequest>,
    ) -> Result<String, String> {
        let client = self.client.lock().await;
        let response = client
            .search_knowledge(req)
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Dashboard of knowledge statistics — thought counts by category, PKS fact counts, BKS truth counts, capture frequency, and top tags."
    )]
    async fn memory_stats(
        &self,
        Parameters(_req): Parameters<MemoryStatsRequest>,
    ) -> Result<String, String> {
        let client = self.client.lock().await;
        let response = client
            .memory_stats()
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(
        description = "Delete a thought by its UUID. Does not remove any PKS facts that were extracted from it."
    )]
    async fn delete_thought(
        &self,
        Parameters(req): Parameters<DeleteThoughtRequest>,
    ) -> Result<String, String> {
        let client = self.client.lock().await;
        let response = client
            .delete_thought(&req.id, req.owner_id.as_deref())
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }
}

// ── Prompt definitions (slash commands) ──────────────────────────────────

#[prompt_router]
impl BrainMcpServer {
    #[prompt(
        name = "capture",
        description = "Capture a new thought into persistent memory"
    )]
    async fn capture_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please capture this thought into persistent memory: {}",
                content
            ),
        )])
    }

    #[prompt(name = "search", description = "Semantic search across all memory")]
    async fn search_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!("Please search memory for: {}", query),
        )])
    }

    #[prompt(name = "recent", description = "List recently captured thoughts")]
    async fn recent_prompt(&self) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            PromptMessageRole::User,
            "Please list recent thoughts from memory.",
        )]
    }

    #[prompt(name = "stats", description = "Show memory statistics dashboard")]
    async fn stats_prompt(&self) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            PromptMessageRole::User,
            "Please show memory statistics.",
        )]
    }

    #[prompt(
        name = "knowledge",
        description = "Search personal and behavioral knowledge"
    )]
    async fn knowledge_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!("Please search knowledge base for: {}", query),
        )])
    }
}

// ── ServerHandler ────────────────────────────────────────────────────────

#[tool_handler(router = self.tool_router)]
#[prompt_handler]
impl ServerHandler for BrainMcpServer {
    fn get_info(&self) -> ServerInfo {
        {
            let mut info = ServerInfo::default();
            info.capabilities = ServerCapabilities::builder()
                .enable_tools()
                .enable_prompts()
                .build();
            info.server_info = Implementation::new("brainwires-brain", env!("CARGO_PKG_VERSION"))
                .with_title("Open Brain — Persistent Knowledge for Any AI Tool");
            info.instructions = Some(
                "Open Brain MCP server — persistent knowledge storage with semantic search. \
                Use capture_thought to store thoughts/decisions/insights, \
                search_memory for semantic retrieval, \
                search_knowledge for PKS/BKS facts, \
                and memory_stats for a dashboard."
                    .into(),
            );
            info
        }
    }
}
