use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Specifies which contexts can invoke a tool.
/// Implements Anthropic's `allowed_callers` pattern for programmatic tool calling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ToolCaller {
    /// Tool can be called directly by the AI
    #[default]
    Direct,
    /// Tool can only be called from within code/script execution
    CodeExecution,
}

/// A tool that can be used by the AI agent
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Tool {
    /// Name of the tool
    #[serde(default)]
    pub name: String,
    /// Description of what the tool does
    #[serde(default)]
    pub description: String,
    /// Input schema (JSON Schema)
    #[serde(default)]
    pub input_schema: ToolInputSchema,
    /// Whether this tool requires user approval before execution
    #[serde(default)]
    pub requires_approval: bool,
    /// Whether this tool should be deferred from initial context loading.
    #[serde(default)]
    pub defer_loading: bool,
    /// Specifies which contexts can call this tool.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_callers: Vec<ToolCaller>,
    /// Example inputs that teach the AI proper parameter usage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_examples: Vec<Value>,
    /// When `true`, the agent loop MUST execute this tool sequentially — never
    /// concurrently with other tools in the same round.
    ///
    /// Use for tools that mutate shared state (file writes, git operations,
    /// registry updates) where concurrent execution could corrupt data or
    /// interleave side effects. Read-only tools (read_file, search, web_fetch)
    /// should leave this `false`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub serialize: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// JSON Schema for tool input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    /// Schema type (typically "object").
    #[serde(rename = "type", default = "default_schema_type")]
    pub schema_type: String,
    /// Property definitions mapping name to JSON Schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Value>>,
    /// List of required property names.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

fn default_schema_type() -> String {
    "object".to_string()
}

impl Default for ToolInputSchema {
    fn default() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: None,
            required: None,
        }
    }
}

impl ToolInputSchema {
    /// Create a new object schema
    pub fn object(properties: HashMap<String, Value>, required: Vec<String>) -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: Some(properties),
            required: Some(required),
        }
    }
}

/// A tool use request from the AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUse {
    /// Unique ID for this tool use
    pub id: String,
    /// Name of the tool to use
    pub name: String,
    /// Input parameters for the tool
    pub input: Value,
}

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID of the tool use this is a result for
    pub tool_use_id: String,
    /// Result content
    pub content: String,
    /// Whether this is an error result
    #[serde(default)]
    pub is_error: bool,
}

impl ToolResult {
    /// Create a successful tool result
    pub fn success<S: Into<String>>(tool_use_id: S, content: S) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: false,
        }
    }

    /// Create an error tool result
    pub fn error<S: Into<String>>(tool_use_id: S, error: S) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: error.into(),
            is_error: true,
        }
    }
}

// ── Idempotency registry ─────────────────────────────────────────────────────

/// Record of a completed idempotent write operation.
#[derive(Debug, Clone)]
pub struct IdempotencyRecord {
    /// Unix timestamp of first execution.
    pub executed_at: i64,
    /// The success message returned on first execution (returned verbatim on retries).
    pub cached_result: String,
}

/// Shared registry that deduplicates mutating file-system tool calls within a run.
///
/// Create one per agent run and attach it to `ToolContext` via
/// `ToolContext::with_idempotency_registry`.  All clones of the `ToolContext`
/// share the same underlying map so that idempotency is enforced across the
/// entire run regardless of how many times the context is cloned.
#[derive(Debug, Clone, Default)]
pub struct IdempotencyRegistry(Arc<Mutex<HashMap<String, IdempotencyRecord>>>);

impl IdempotencyRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the cached result for `key`, or `None` if not yet executed.
    pub fn get(&self, key: &str) -> Option<IdempotencyRecord> {
        self.0
            .lock()
            .expect("idempotency registry lock poisoned")
            .get(key)
            .cloned()
    }

    /// Record that `key` produced `result`.
    ///
    /// If `key` was already recorded (concurrent retry), the first result wins.
    pub fn record(&self, key: String, result: String) {
        let mut map = self.0.lock().expect("idempotency registry lock poisoned");
        map.entry(key).or_insert_with(|| {
            use chrono::Utc;
            IdempotencyRecord {
                executed_at: Utc::now().timestamp(),
                cached_result: result,
            }
        });
    }

    /// Number of recorded operations.
    pub fn len(&self) -> usize {
        self.0
            .lock()
            .expect("idempotency registry lock poisoned")
            .len()
    }

    /// Returns `true` if no operations have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Side-effect staging (two-phase commit) ────────────────────────────────────

/// A single write operation that has been staged but not yet committed.
#[derive(Debug, Clone)]
pub struct StagedWrite {
    /// Content-addressed key — used to deduplicate identical staged writes.
    pub key: String,
    /// The absolute target path on the filesystem.
    pub target_path: PathBuf,
    /// UTF-8 content to write on commit.
    pub content: String,
}

/// Result returned by a successful [`StagingBackend::commit`].
#[derive(Debug, Clone)]
pub struct CommitResult {
    /// Number of writes successfully committed to disk.
    pub committed: usize,
    /// The target paths that were written.
    pub paths: Vec<PathBuf>,
}

/// Trait for staging write operations before committing to the filesystem.
///
/// Defined in `brainwires-core` so that [`ToolContext`] can hold an
/// `Arc<dyn StagingBackend>` without depending on `brainwires-tools`,
/// which would create a circular crate dependency.
///
/// The concrete implementation lives in `brainwires-tools::transaction::TransactionManager`.
pub trait StagingBackend: std::fmt::Debug + Send + Sync {
    /// Stage a write operation.
    ///
    /// Returns `true` if newly staged, `false` if `key` was already present
    /// (idempotent — same key staged twice is a no-op).
    fn stage(&self, write: StagedWrite) -> bool;

    /// Commit all staged writes to the filesystem.
    ///
    /// Each staged file is moved (or copied) to its target path atomically.
    /// On success the staging queue is cleared.
    fn commit(&self) -> anyhow::Result<CommitResult>;

    /// Discard all staged writes without touching the filesystem.
    fn rollback(&self);

    /// Return the number of pending staged writes.
    fn pending_count(&self) -> usize;
}

// ── Intended-write hash registry ─────────────────────────────────────────────

/// Shared map of `path -> SHA-256 of most recently written content`.
///
/// Populated by `write_file` (in `brainwires-tools`) after its post-write
/// read-back succeeds, and read by the validation loop (in `brainwires-agent`)
/// to detect *post-validation* clobber by a concurrent writer.
///
/// Why this exists (in addition to the tool-level read-back check):
///   - The read-back check catches interleaved writes within a single
///     `write_file` call.
///   - It does NOT catch: agent A writes, agent A's validation passes,
///     agent B writes, agent A finalises `Success: true` — at which point
///     the content A claims to have written is no longer on disk.
///
/// By recording the intended hash and re-reading at agent-finalisation time,
/// A sees the mismatch and its retry/failure machinery kicks in — so at most
/// one of two racing agents can legitimately report success.
#[derive(Debug, Clone, Default)]
pub struct IntendedWrites(Arc<Mutex<HashMap<PathBuf, [u8; 32]>>>);

impl IntendedWrites {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the SHA-256 of content written to `path`.  The most recent
    /// write wins (overwrite semantics — consistent with filesystem reality).
    pub fn record(&self, path: PathBuf, hash: [u8; 32]) {
        let mut map = self.0.lock().expect("intended writes lock poisoned");
        map.insert(path, hash);
    }

    /// Return the hash recorded for `path`, or `None` if none.
    pub fn get(&self, path: &Path) -> Option<[u8; 32]> {
        let map = self.0.lock().expect("intended writes lock poisoned");
        map.get(path).copied()
    }

    /// Snapshot all `(path, hash)` pairs currently recorded.
    pub fn snapshot(&self) -> Vec<(PathBuf, [u8; 32])> {
        let map = self.0.lock().expect("intended writes lock poisoned");
        map.iter().map(|(p, h)| (p.clone(), *h)).collect()
    }

    /// Number of recorded paths.
    pub fn len(&self) -> usize {
        self.0.lock().expect("intended writes lock poisoned").len()
    }

    /// Returns `true` if no writes have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── ToolContext ───────────────────────────────────────────────────────────────

/// Execution context for a tool.
///
/// Provides the working directory, optional metadata, and permission capabilities
/// to tool implementations.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Current working directory for resolving relative paths
    pub working_directory: String,
    /// User ID (if authenticated)
    pub user_id: Option<String>,
    /// Additional context data (application-specific key-value pairs)
    pub metadata: HashMap<String, String>,
    /// Agent capabilities for permission checks (serialized as JSON value).
    ///
    /// Consumers should serialize their concrete capability types into this field
    /// and deserialize when reading. This keeps the core crate free of capability
    /// type definitions.
    pub capabilities: Option<Value>,
    /// Per-run idempotency registry for mutating file operations.
    ///
    /// When `Some`, write/delete/edit operations derive a content-addressed key
    /// and skip re-execution if the same key has already been processed in this
    /// run.  `None` disables idempotency tracking (useful for tests or simple
    /// single-call use cases).
    pub idempotency_registry: Option<IdempotencyRegistry>,
    /// Optional two-phase commit staging backend.
    ///
    /// When `Some`, mutating file operations (`write_file`, `edit_file`,
    /// `patch_file`) stage their writes instead of applying them immediately.
    /// The caller is responsible for calling `backend.commit()` to finalize the
    /// writes, or `backend.rollback()` to discard them.
    ///
    /// Staging is checked *after* the idempotency registry: if the same
    /// operation key is already cached, the cached result is returned without
    /// staging again.
    pub staging_backend: Option<Arc<dyn StagingBackend>>,
    /// Optional shared registry that tracks `(path -> SHA-256)` for every
    /// successful `write_file` in this run.
    ///
    /// When `Some`, `write_file` records the hash of its content after the
    /// post-write read-back succeeds.  The agent's validation loop then
    /// re-reads each tracked file at finalisation and compares to detect
    /// post-validation clobber by a concurrent writer.
    ///
    /// `None` disables hash tracking (CLI-driven tool invocations outside an
    /// agent).  The cost of tracking is a `HashMap` insert per write.
    pub intended_writes: Option<IntendedWrites>,
}

impl ToolContext {
    /// Attach a fresh idempotency registry to this context (builder pattern).
    pub fn with_idempotency_registry(mut self) -> Self {
        self.idempotency_registry = Some(IdempotencyRegistry::new());
        self
    }

    /// Attach a staging backend for two-phase commit file writes (builder pattern).
    pub fn with_staging_backend(mut self, backend: Arc<dyn StagingBackend>) -> Self {
        self.staging_backend = Some(backend);
        self
    }

    /// Attach a fresh intended-writes registry (builder pattern).
    pub fn with_intended_writes(mut self) -> Self {
        self.intended_writes = Some(IntendedWrites::new());
        self
    }

    /// Attach an existing intended-writes registry — useful when the agent
    /// owns the registry and wants tool calls to share it (builder pattern).
    pub fn with_intended_writes_registry(mut self, registry: IntendedWrites) -> Self {
        self.intended_writes = Some(registry);
        self
    }

    /// Record the SHA-256 of content written to `path` in the attached
    /// intended-writes registry.  No-op when no registry is attached
    /// (e.g., CLI-driven tool invocations outside an agent).
    pub fn record_write(&self, path: PathBuf, hash: [u8; 32]) {
        if let Some(ref reg) = self.intended_writes {
            reg.record(path, hash);
        }
    }
}

impl Default for ToolContext {
    fn default() -> Self {
        Self {
            working_directory: std::env::current_dir()
                .ok()
                .and_then(|p| p.to_str().map(|s| s.to_string()))
                .unwrap_or_else(|| ".".to_string()),
            user_id: None,
            metadata: HashMap::new(),
            capabilities: None,
            idempotency_registry: None,
            staging_backend: None,
            intended_writes: None,
        }
    }
}

/// Tool selection mode
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ToolMode {
    /// All tools from registry
    Full,
    /// User-selected specific tools (stores tool names)
    Explicit(Vec<String>),
    /// Smart routing based on query analysis (default)
    #[default]
    Smart,
    /// Core tools only
    Core,
    /// No tools enabled
    None,
}

impl ToolMode {
    /// Get a display name for the mode
    pub fn display_name(&self) -> &'static str {
        match self {
            ToolMode::Full => "full",
            ToolMode::Explicit(_) => "explicit",
            ToolMode::Smart => "smart",
            ToolMode::Core => "core",
            ToolMode::None => "none",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("tool-1", "Success!");
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("tool-2", "Failed!");
        assert!(result.is_error);
    }

    #[test]
    fn test_tool_input_schema_object() {
        let mut props = HashMap::new();
        props.insert("name".to_string(), json!({"type": "string"}));
        let schema = ToolInputSchema::object(props, vec!["name".to_string()]);
        assert_eq!(schema.schema_type, "object");
        assert!(schema.properties.is_some());
    }

    #[test]
    fn test_idempotency_registry_basic() {
        let registry = IdempotencyRegistry::new();
        assert!(registry.is_empty());

        registry.record("key-1".to_string(), "result-1".to_string());
        assert_eq!(registry.len(), 1);

        let record = registry.get("key-1").unwrap();
        assert_eq!(record.cached_result, "result-1");
        assert!(record.executed_at > 0);

        // Second record call with same key is a no-op (first result wins)
        registry.record("key-1".to_string(), "result-DIFFERENT".to_string());
        assert_eq!(registry.get("key-1").unwrap().cached_result, "result-1");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_idempotency_registry_clone_shares_state() {
        let registry = IdempotencyRegistry::new();
        let clone = registry.clone();

        registry.record("k".to_string(), "v".to_string());
        // Clone sees the same entry because it shares the Arc<Mutex<...>>
        assert!(clone.get("k").is_some());
    }

    #[test]
    fn test_tool_context_default_has_no_registry() {
        let ctx = ToolContext::default();
        assert!(ctx.idempotency_registry.is_none());
    }

    #[test]
    fn test_tool_context_with_registry() {
        let ctx = ToolContext::default().with_idempotency_registry();
        assert!(ctx.idempotency_registry.is_some());
        assert!(ctx.idempotency_registry.unwrap().is_empty());
    }
}
