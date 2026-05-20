//! MCP server — exposes context management tools to Claude Code.

use anyhow::{Context, Result};
use rmcp::{
    ServerHandler, ServiceExt, handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, tool, tool_handler, tool_router,
};

use crate::config::ClaudeBrainConfig;
use crate::context_manager::ContextManager;

/// Request to recall context from conversation history.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallContextRequest {
    /// Natural language query to search conversation history.
    pub query: String,
    /// Maximum results (default: 20).
    #[serde(default = "default_recall_limit")]
    pub limit: usize,
    /// Minimum relevance score 0.0-1.0 (default: 0.45).
    #[serde(default = "default_recall_min_score")]
    pub min_score: f32,
}

/// Request to capture a thought.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct CaptureRequest {
    /// The thought, decision, or insight to persist.
    pub content: String,
    /// Category: decision, insight, preference, action_item, reference, general.
    #[serde(default)]
    pub category: Option<String>,
    /// Tags for organization.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Importance 0.0-1.0 (default: 0.5).
    #[serde(default)]
    pub importance: Option<f32>,
}

/// Request to search memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    /// Natural language search query.
    pub query: String,
    /// Maximum results (default: 10).
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum relevance score (default: 0.6).
    #[serde(default = "default_min_score")]
    pub min_score: f32,
}

/// Request to search knowledge base (PKS/BKS).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct KnowledgeSearchRequest {
    /// Query for PKS/BKS knowledge.
    pub query: String,
    /// Maximum results (default: 10).
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// Request to consolidate session thoughts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ConsolidateRequest {
    /// Optional session ID to consolidate. If omitted, consolidates all sessions.
    #[serde(default)]
    pub session_id: Option<String>,
}

/// Request to teach a behavioral rule.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct LearnRequest {
    /// The rule to learn (e.g., "always use --no-stream with pm2 logs").
    pub rule: String,
    /// Category: command_usage, task_strategy, tool_behavior, error_recovery, resource_management, pattern_avoidance, prompting_technique.
    #[serde(default)]
    pub category: Option<String>,
    /// Why this rule exists.
    #[serde(default)]
    pub rationale: Option<String>,
    /// Context pattern where this rule applies.
    #[serde(default)]
    pub context: Option<String>,
}

/// Empty request for stats.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct StatsRequest {}

fn default_limit() -> usize {
    10
}
fn default_recall_limit() -> usize {
    20
}
fn default_min_score() -> f32 {
    0.3
}
fn default_recall_min_score() -> f32 {
    0.45
}

/// Claude Brain MCP server.
#[derive(Clone)]
pub struct ClaudeBrainMcpServer {
    ctx: std::sync::Arc<ContextManager>,
    tool_router: ToolRouter<Self>,
}

impl ClaudeBrainMcpServer {
    /// Create from config.
    pub async fn new(config: ClaudeBrainConfig) -> Result<Self> {
        let ctx = std::sync::Arc::new(
            ContextManager::new(config)
                .await
                .context("Failed to create ContextManager")?,
        );
        Ok(Self {
            ctx,
            tool_router: Self::tool_router(),
        })
    }

    /// Serve on stdin/stdout.
    pub async fn serve_stdio() -> Result<()> {
        tracing::info!("Starting Claude Brain MCP server");

        let config = ClaudeBrainConfig::load()?;
        let server = Self::new(config)
            .await
            .context("Failed to create Claude Brain MCP server")?;

        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;

        Ok(())
    }
}

// ── Tool definitions ─────────────────────────────────────────────────────

#[tool_router(router = tool_router)]
impl ClaudeBrainMcpServer {
    #[tool(
        description = "Search captured thoughts (hot-tier only) for conversation context outside the current window. Use when recalling earlier discussion details, decisions, or code from previous turns/sessions. Does NOT search PKS/BKS knowledge — use search_knowledge for that."
    )]
    async fn recall_context(
        &self,
        Parameters(req): Parameters<RecallContextRequest>,
    ) -> Result<String, String> {
        let response = self
            .ctx
            .search_thoughts(&req.query, req.limit, req.min_score)
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {e}"))
    }

    #[tool(
        description = "Capture a thought, decision, insight, or important context into persistent memory. Automatically categorizes, extracts tags, embeds for semantic search, and extracts knowledge facts."
    )]
    async fn capture_thought(
        &self,
        Parameters(req): Parameters<CaptureRequest>,
    ) -> Result<String, String> {
        use brainwires_knowledge::knowledge::types::CaptureThoughtRequest;

        let mut client = self.ctx.client().lock_owned().await;
        let response = client
            .capture_thought(CaptureThoughtRequest {
                content: req.content,
                category: req.category,
                tags: req.tags,
                importance: req.importance,
                source: Some("claude-brain-mcp".to_string()),
                owner_id: None,
            })
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {e}"))
    }

    #[tool(
        description = "Search across ALL memory tiers — thoughts, personal facts (PKS), and behavioral knowledge (BKS). Broader than recall_context. Use for cross-tier semantic search when you need the full picture."
    )]
    async fn search_memory(
        &self,
        Parameters(req): Parameters<SearchRequest>,
    ) -> Result<String, String> {
        let response = self
            .ctx
            .search_memory(&req.query, req.limit, req.min_score)
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {e}"))
    }

    #[tool(
        description = "Query personal knowledge (PKS) facts and behavioral knowledge (BKS) truths. Use this before making choices to check for known preferences."
    )]
    async fn search_knowledge(
        &self,
        Parameters(req): Parameters<KnowledgeSearchRequest>,
    ) -> Result<String, String> {
        let response = self
            .ctx
            .search_knowledge(&req.query, req.limit)
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {e}"))
    }

    #[tool(
        description = "Dashboard of knowledge statistics — thought counts by category, PKS fact counts, BKS truth counts, capture frequency, and top tags."
    )]
    async fn memory_stats(
        &self,
        Parameters(_req): Parameters<StatsRequest>,
    ) -> Result<String, String> {
        let response = self
            .ctx
            .memory_stats()
            .await
            .map_err(|e| format!("{:#}", e))?;
        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {e}"))
    }

    #[tool(
        description = "Compact session thoughts — concatenates auto-captured messages into a single record and deletes originals. Reduces storage bloat. Does NOT summarize (no LLM pass). Optionally target a specific session_id."
    )]
    async fn consolidate_now(
        &self,
        Parameters(req): Parameters<ConsolidateRequest>,
    ) -> Result<String, String> {
        use crate::session_adapter::BrainSessionAdapter;
        use brainwires_memory::dream::consolidator::DreamSessionStore;

        let adapter = BrainSessionAdapter::new(self.ctx.client());

        let sessions = if let Some(ref sid) = req.session_id {
            vec![sid.clone()]
        } else {
            adapter
                .list_sessions()
                .await
                .map_err(|e| format!("{:#}", e))?
        };

        let mut total_consolidated = 0;
        for session_key in &sessions {
            let messages = adapter
                .load(session_key)
                .await
                .map_err(|e| format!("{:#}", e))?;
            if let Some(msgs) = messages
                && !msgs.is_empty()
            {
                adapter
                    .save(session_key, &msgs)
                    .await
                    .map_err(|e| format!("{:#}", e))?;
                total_consolidated += 1;
            }
        }

        Ok(format!(
            "{{\"consolidated_sessions\": {}, \"total_sessions\": {}}}",
            total_consolidated,
            sessions.len()
        ))
    }

    #[tool(
        description = "Teach a behavioral rule to the BKS (Behavioral Knowledge System). Use this when you discover patterns about how to work effectively — command usage, error recovery, tool behavior, etc. Rules persist across sessions and inform future behavior."
    )]
    async fn learn(&self, Parameters(req): Parameters<LearnRequest>) -> Result<String, String> {
        use brainwires_knowledge::knowledge::bks_pks::{
            BehavioralTruth, TruthCategory, TruthSource,
        };

        let category = match req.category.as_deref() {
            Some("command_usage") => TruthCategory::CommandUsage,
            Some("task_strategy") => TruthCategory::TaskStrategy,
            Some("tool_behavior") => TruthCategory::ToolBehavior,
            Some("error_recovery") => TruthCategory::ErrorRecovery,
            Some("resource_management") => TruthCategory::ResourceManagement,
            Some("pattern_avoidance") => TruthCategory::PatternAvoidance,
            Some("prompting_technique") => TruthCategory::PromptingTechnique,
            _ => TruthCategory::TaskStrategy,
        };

        let truth = BehavioralTruth::new(
            category,
            req.context.unwrap_or_else(|| "*".to_string()),
            req.rule,
            req.rationale.unwrap_or_default(),
            TruthSource::ExplicitCommand,
            Some("claude-brain-mcp".to_string()),
        );
        let truth_id = truth.id.clone();

        let mut client = self.ctx.client().lock_owned().await;
        client
            .add_behavioral_truth(truth)
            .map_err(|e| format!("{:#}", e))?;

        Ok(format!(
            "{{\"learned\": true, \"truth_id\": \"{}\"}}",
            truth_id
        ))
    }
}

// ── ServerHandler ────────────────────────────────────────────────────────

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ClaudeBrainMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new("claude-brain", env!("CARGO_PKG_VERSION"))
            .with_title("Claude Brain — Brainwires Context Management for Claude Code");
        info.instructions = Some(
            "Claude Brain replaces Claude Code's default compaction with Brainwires \
             research-grade context management. Use recall_context to search past \
             conversation history, capture_thought to persist decisions and insights, \
             search_memory for semantic retrieval across all tiers, search_knowledge \
             for PKS/BKS facts, and memory_stats for a dashboard."
                .into(),
        );
        info
    }
}
