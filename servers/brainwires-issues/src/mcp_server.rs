//! MCP server implementation for the issue tracker.

use anyhow::{Context, Result};
use brainwires_core::paths::PlatformPaths;
use brainwires_storage::{LanceDatabase, bm25_search::BM25Search};
use rmcp::{
    RoleServer, ServerHandler, ServiceExt,
    handler::server::{router::prompt::PromptRouter, tool::ToolRouter, wrapper::Parameters},
    model::*,
    prompt, prompt_handler, prompt_router,
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    store::{CommentStore, IssueStore},
    types::{Comment, Issue, IssuePatch, IssuePriority, IssueStatus},
};

// ── Request / Response types ─────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateIssueRequest {
    /// Short title for the issue.
    pub title: String,
    /// Full description in Markdown (optional).
    pub description: Option<String>,
    /// Priority level: no_priority | low | medium | high | urgent
    pub priority: Option<String>,
    /// Assignee name or identifier.
    pub assignee: Option<String>,
    /// Project or milestone name.
    pub project: Option<String>,
    /// Parent issue UUID for sub-issues.
    pub parent_id: Option<String>,
    /// Labels to attach.
    pub labels: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetIssueRequest {
    /// Issue UUID or display number prefixed with `#` (e.g. `#42`).
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListIssuesRequest {
    /// Filter by project name.
    pub project: Option<String>,
    /// Filter by status: backlog | todo | in_progress | in_review | done | cancelled
    pub status: Option<String>,
    /// Filter by assignee.
    pub assignee: Option<String>,
    /// Filter by label.
    pub label: Option<String>,
    /// Number of records to skip for pagination (0-based). Use `next_offset` from previous response.
    pub offset: Option<usize>,
    /// Maximum number of issues to return (1–100, default 25).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateIssueRequest {
    /// Issue UUID to update.
    pub id: String,
    /// New title.
    pub title: Option<String>,
    /// New description.
    pub description: Option<String>,
    /// New status: backlog | todo | in_progress | in_review | done | cancelled
    pub status: Option<String>,
    /// New priority: no_priority | low | medium | high | urgent
    pub priority: Option<String>,
    /// Replace labels entirely.
    pub labels: Option<Vec<String>>,
    /// New assignee.
    pub assignee: Option<String>,
    /// Set to true to remove the assignee.
    pub clear_assignee: Option<bool>,
    /// New project.
    pub project: Option<String>,
    /// Set to true to remove the project.
    pub clear_project: Option<bool>,
    /// New parent issue UUID.
    pub parent_id: Option<String>,
    /// Set to true to detach from parent.
    pub clear_parent: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CloseIssueRequest {
    /// Issue UUID to close.
    pub id: String,
    /// Resolution: `done` (default) or `cancelled`.
    pub resolution: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteIssueRequest {
    /// Issue UUID to delete.
    pub id: String,
    /// If true, also delete all comments on the issue. Defaults to false (comments are kept).
    pub delete_comments: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchIssuesRequest {
    /// Natural language or keyword query.
    pub query: String,
    /// Maximum number of results (default 10).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddCommentRequest {
    /// Issue UUID to comment on.
    pub issue_id: String,
    /// Comment body in Markdown.
    pub body: String,
    /// Author name or identifier.
    pub author: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListCommentsRequest {
    /// Issue UUID.
    pub issue_id: String,
    /// Number of records to skip for pagination (0-based). Use `next_offset` from previous response.
    pub offset: Option<usize>,
    /// Maximum number of comments to return (1–200, default 50).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteCommentRequest {
    /// Comment UUID to delete.
    pub id: String,
}

// ── Paginated list response ───────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct PagedIssues {
    issues: Vec<Issue>,
    count: usize,
    next_offset: Option<usize>,
}

#[derive(Debug, Serialize)]
struct PagedComments {
    comments: Vec<Comment>,
    count: usize,
    next_offset: Option<usize>,
}

// ── IssuesMcpServer ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct IssuesMcpServer {
    issues: Arc<IssueStore<LanceDatabase>>,
    comments: Arc<CommentStore<LanceDatabase>>,
    tool_router: ToolRouter<Self>,
    #[allow(dead_code)]
    prompt_router: PromptRouter<Self>,
}

impl IssuesMcpServer {
    /// Create a new server with the default LanceDB backend.
    pub async fn new() -> Result<Self> {
        let data_dir = PlatformPaths::data_dir().join("brainwires-issues");
        Self::with_data_dir(&data_dir).await
    }

    /// Create a new server rooted at `data_dir`. The directory will be created
    /// if needed, with a `lancedb/` subdir for the table backend and a `bm25/`
    /// subdir for the keyword index. Exposed for integration tests that need
    /// an isolated `TempDir` per run.
    pub async fn with_data_dir(data_dir: &std::path::Path) -> Result<Self> {
        let db_path = data_dir.join("lancedb");
        let backend = Arc::new(
            LanceDatabase::new(db_path.to_string_lossy())
                .await
                .context("Failed to connect to LanceDB")?,
        );

        let bm25_path = data_dir.join("bm25");
        let issues = Arc::new(match BM25Search::new(&bm25_path) {
            Ok(bm25) => IssueStore::new_with_bm25(Arc::clone(&backend), bm25),
            Err(e) => {
                tracing::warn!(
                    "BM25 index unavailable (falling back to in-memory search): {}",
                    e
                );
                IssueStore::new(Arc::clone(&backend))
            }
        });
        let comments = Arc::new(CommentStore::new(Arc::clone(&backend)));

        issues
            .ensure_table()
            .await
            .context("Failed to ensure issues table")?;
        comments
            .ensure_table()
            .await
            .context("Failed to ensure comments table")?;

        Ok(Self {
            issues,
            comments,
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
        })
    }

    pub async fn serve_stdio() -> Result<()> {
        tracing::info!("Starting Issues MCP server");

        let server = Self::new().await.context("Failed to create MCP server")?;
        let transport = rmcp::transport::io::stdio();
        server.serve(transport).await?.waiting().await?;

        Ok(())
    }

    // ── Helper to resolve an issue by UUID or #number ────────────────────

    async fn resolve_issue(&self, id: &str) -> Result<Issue, String> {
        if let Some(num_str) = id.strip_prefix('#') {
            let num: u64 = num_str
                .parse()
                .map_err(|_| format!("Invalid issue number: {}", id))?;
            self.issues
                .get_by_number(num)
                .await
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("Issue {} not found", id))
        } else {
            self.issues
                .get(id)
                .await
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("Issue {} not found", id))
        }
    }
}

// ── Tool implementations ─────────────────────────────────────────────────

#[tool_router(router = tool_router)]
impl IssuesMcpServer {
    #[tool(description = "Create a new issue")]
    async fn create_issue(
        &self,
        Parameters(req): Parameters<CreateIssueRequest>,
    ) -> Result<String, String> {
        // C9: validate title
        if req.title.trim().is_empty() {
            return Err("title must not be empty".to_string());
        }

        let number = self.issues.next_number().await.map_err(|e| e.to_string())?;
        let mut issue = Issue::new(number, req.title);

        if let Some(d) = req.description {
            issue.description = d;
        }
        if let Some(p) = req.priority {
            issue.priority = IssuePriority::parse(&p);
        }
        if let Some(a) = req.assignee {
            issue.assignee = Some(a);
        }
        if let Some(p) = req.project {
            issue.project = Some(p);
        }
        if let Some(p) = req.parent_id {
            issue.parent_id = Some(p);
        }
        if let Some(l) = req.labels {
            issue.labels = l;
        }

        self.issues
            .create(&issue)
            .await
            .map_err(|e| e.to_string())?;

        serde_json::to_string_pretty(&issue).map_err(|e| e.to_string())
    }

    #[tool(description = "Get an issue by UUID or display number (e.g. #42)")]
    async fn get_issue(
        &self,
        Parameters(req): Parameters<GetIssueRequest>,
    ) -> Result<String, String> {
        let issue = self.resolve_issue(&req.id).await?;
        serde_json::to_string_pretty(&issue).map_err(|e| e.to_string())
    }

    #[tool(
        description = "List issues with optional filters for project, status, assignee, and label. Supports offset-based pagination."
    )]
    async fn list_issues(
        &self,
        Parameters(req): Parameters<ListIssuesRequest>,
    ) -> Result<String, String> {
        // C9: clamp limit
        let limit = req.limit.unwrap_or(25).clamp(1, 100);
        let status = req.status.as_deref().map(IssueStatus::parse);

        let (issues, next_offset) = self
            .issues
            .list(
                req.project.as_deref(),
                status.as_ref(),
                req.assignee.as_deref(),
                req.label.as_deref(),
                req.offset,
                limit,
            )
            .await
            .map_err(|e| e.to_string())?;

        let count = issues.len();
        let result = PagedIssues {
            issues,
            count,
            next_offset,
        };

        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }

    #[tool(description = "Update fields on an existing issue")]
    async fn update_issue(
        &self,
        Parameters(req): Parameters<UpdateIssueRequest>,
    ) -> Result<String, String> {
        let patch = IssuePatch {
            title: req.title,
            description: req.description,
            status: req.status.as_deref().map(IssueStatus::parse),
            priority: req.priority.as_deref().map(IssuePriority::parse),
            labels: req.labels,
            assignee: req.assignee,
            clear_assignee: req.clear_assignee,
            project: req.project,
            clear_project: req.clear_project,
            parent_id: req.parent_id,
            clear_parent: req.clear_parent,
        };

        let updated = self
            .issues
            .update(&req.id, patch)
            .await
            .map_err(|e| e.to_string())?;

        serde_json::to_string_pretty(&updated).map_err(|e| e.to_string())
    }

    #[tool(description = "Close or cancel an issue")]
    async fn close_issue(
        &self,
        Parameters(req): Parameters<CloseIssueRequest>,
    ) -> Result<String, String> {
        let status = match req.resolution.as_deref() {
            Some("cancelled") => IssueStatus::Cancelled,
            _ => IssueStatus::Done,
        };

        let patch = IssuePatch {
            status: Some(status),
            ..Default::default()
        };

        let updated = self
            .issues
            .update(&req.id, patch)
            .await
            .map_err(|e| e.to_string())?;

        serde_json::to_string_pretty(&updated).map_err(|e| e.to_string())
    }

    #[tool(description = "Delete an issue permanently")]
    async fn delete_issue(
        &self,
        Parameters(req): Parameters<DeleteIssueRequest>,
    ) -> Result<String, String> {
        // Verify it exists first
        self.resolve_issue(&req.id).await?;

        if req.delete_comments.unwrap_or(false) {
            self.comments
                .delete_by_issue(&req.id)
                .await
                .map_err(|e| e.to_string())?;
        }

        self.issues
            .delete(&req.id)
            .await
            .map_err(|e| e.to_string())?;

        // C8: safe JSON delete response
        serde_json::to_string(&serde_json::json!({"deleted": req.id})).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Search issues by keyword using BM25 full-text search. Matches against title and description."
    )]
    async fn search_issues(
        &self,
        Parameters(req): Parameters<SearchIssuesRequest>,
    ) -> Result<String, String> {
        // C9: clamp limit
        let limit = req.limit.unwrap_or(10).clamp(1, 50);

        let matches = self
            .issues
            .search(&req.query, limit)
            .await
            .map_err(|e| e.to_string())?;

        let count = matches.len();
        let result = PagedIssues {
            count,
            issues: matches,
            next_offset: None,
        };

        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }

    #[tool(description = "Add a comment to an issue")]
    async fn add_comment(
        &self,
        Parameters(req): Parameters<AddCommentRequest>,
    ) -> Result<String, String> {
        // Verify issue exists
        self.resolve_issue(&req.issue_id).await?;

        let mut comment = Comment::new(&req.issue_id, req.body);
        if let Some(a) = req.author {
            comment.author = Some(a);
        }

        self.comments
            .add(&comment)
            .await
            .map_err(|e| e.to_string())?;

        serde_json::to_string_pretty(&comment).map_err(|e| e.to_string())
    }

    #[tool(description = "List comments on an issue with offset-based pagination")]
    async fn list_comments(
        &self,
        Parameters(req): Parameters<ListCommentsRequest>,
    ) -> Result<String, String> {
        // C9: clamp limit
        let limit = req.limit.unwrap_or(50).clamp(1, 200);

        let (comments, next_offset) = self
            .comments
            .list_for_issue(&req.issue_id, req.offset, limit)
            .await
            .map_err(|e| e.to_string())?;

        let count = comments.len();
        let result = PagedComments {
            comments,
            count,
            next_offset,
        };

        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }

    #[tool(description = "Delete a comment by its UUID")]
    async fn delete_comment(
        &self,
        Parameters(req): Parameters<DeleteCommentRequest>,
    ) -> Result<String, String> {
        // C7: verify comment exists before deleting
        self.comments
            .get(&req.id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Comment {} not found", req.id))?;

        self.comments
            .delete(&req.id)
            .await
            .map_err(|e| e.to_string())?;

        // C8: safe JSON delete response
        serde_json::to_string(&serde_json::json!({"deleted": req.id})).map_err(|e| e.to_string())
    }
}

// ── Prompts ───────────────────────────────────────────────────────────────

#[prompt_router]
impl IssuesMcpServer {
    #[prompt(name = "create", description = "Create a new issue")]
    async fn create_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Vec<PromptMessage> {
        let title = args.get("title").and_then(|v| v.as_str()).unwrap_or("");
        let body = if title.is_empty() {
            "Please create a new issue. Ask me for the title, description, priority, and assignee."
                .to_string()
        } else {
            format!("Please create a new issue titled: \"{}\"", title)
        };
        vec![PromptMessage::new_text(PromptMessageRole::User, body)]
    }

    #[prompt(name = "list", description = "List open issues")]
    async fn list_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Vec<PromptMessage> {
        let project = args.get("project").and_then(|v| v.as_str()).unwrap_or("");
        let body = if project.is_empty() {
            "Please list all open issues (status: backlog, todo, in_progress, in_review)."
                .to_string()
        } else {
            format!("Please list all open issues for project \"{}\".", project)
        };
        vec![PromptMessage::new_text(PromptMessageRole::User, body)]
    }

    #[prompt(
        name = "search",
        description = "Search issues by keyword or description"
    )]
    async fn search_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Vec<PromptMessage> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!("Please search issues for: {}", query),
        )]
    }

    #[prompt(
        name = "triage",
        description = "Triage backlog issues — review and assign priority, status, and assignee"
    )]
    async fn triage_prompt(&self) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            PromptMessageRole::User,
            "Please list all issues in the backlog and help me triage them. \
             For each issue, suggest a priority, status, and assignee based on the title and description.",
        )]
    }
}

// ── ServerHandler ─────────────────────────────────────────────────────────

#[tool_handler(router = self.tool_router)]
#[prompt_handler]
impl ServerHandler for IssuesMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_prompts()
            .build();
        info.server_info = Implementation::new("brainwires-issues", env!("CARGO_PKG_VERSION"))
            .with_title("Issue Tracker — lightweight project issue tracking");
        info.instructions = Some(
            "Issue tracking MCP server. \
             Use create_issue to file new issues, list_issues to browse with filters, \
             update_issue to change fields, close_issue to resolve, \
             and add_comment / list_comments for discussion threads. \
             Use search_issues for keyword search."
                .into(),
        );
        info
    }
}
