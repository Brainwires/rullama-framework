//! Browser automation tool — spawns Thalora as a subprocess and forwards calls via MCP.
//!
//! `BrowserTool` lazily initializes a `thalora --brainclaw` child process on first use and
//! reuses it for the lifetime of the daemon. If the process crashes, the next call
//! transparently re-spawns and reconnects so agent sessions are not interrupted.
//!
//! # Configuration
//!
//! Read from `ToolContext.metadata["browser_config"]` as a JSON string:
//!
//! ```json
//! { "thalora_binary": "thalora", "session_timeout_secs": 300 }
//! ```
//!
//! If no config is present, `thalora` is looked up via `PATH` and a 300-second timeout is used.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::Result;
use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};
use brainwires_mcp_client::{McpClient, McpServerConfig};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

// ── Per-process singleton ────────────────────────────────────────────────────

/// Global Thalora MCP client.  `None` means not yet connected (or connection lost).
///
/// Uses `tokio::sync::Mutex` so the guard is `Send` and can be held across `.await`.
static THALORA: OnceLock<Mutex<Option<McpClient>>> = OnceLock::new();

const SERVER_NAME: &str = "thalora";

fn global_client() -> &'static Mutex<Option<McpClient>> {
    THALORA.get_or_init(|| Mutex::new(None))
}

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, serde::Serialize)]
struct BrowserToolConfig {
    #[serde(default = "default_binary")]
    thalora_binary: String,
    #[serde(default = "default_timeout")]
    session_timeout_secs: u64,
}

fn default_binary() -> String {
    "thalora".to_string()
}
fn default_timeout() -> u64 {
    300
}

impl BrowserToolConfig {
    fn from_context(context: &ToolContext) -> Self {
        context
            .metadata
            .get("browser_config")
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(BrowserToolConfig {
                thalora_binary: default_binary(),
                session_timeout_secs: default_timeout(),
            })
    }

    fn server_config(&self) -> McpServerConfig {
        McpServerConfig {
            name: SERVER_NAME.to_string(),
            command: self.thalora_binary.clone(),
            args: vec!["--brainclaw".to_string()],
            env: Some(
                [(
                    "THALORA_SESSION_TIMEOUT_SECS".to_string(),
                    self.session_timeout_secs.to_string(),
                )]
                .into_iter()
                .collect(),
            ),
        }
    }
}

// ── Connection + call helper ─────────────────────────────────────────────────

/// Call a tool on Thalora, spawning a fresh subprocess on first call or after failure.
async fn call_thalora_tool(
    context: &ToolContext,
    tool_name: &str,
    arguments: Value,
) -> Result<String> {
    let cfg = BrowserToolConfig::from_context(context);
    let server_cfg = cfg.server_config();

    // Lock the global client for the duration of the call.
    // tokio::sync::MutexGuard is Send, so holding it across await is fine.
    let mut guard = global_client().lock().await;

    // Lazily spawn the Thalora subprocess on first use
    if guard.is_none() {
        tracing::info!(binary = %cfg.thalora_binary, "Spawning Thalora subprocess");
        let client = McpClient::new("brainclaw", env!("CARGO_PKG_VERSION"));
        client.connect(&server_cfg).await?;
        *guard = Some(client);
    }

    let client = guard.as_ref().expect("just initialized");
    let result = client
        .call_tool(SERVER_NAME, tool_name, Some(arguments.clone()))
        .await;

    match result {
        Ok(r) => Ok(extract_text_content(r.content)),
        Err(e) => {
            // Connection may be dead — clear so next call re-spawns
            tracing::warn!(error = %e, "Thalora call failed; resetting connection for next call");
            *guard = None;

            // Retry once with a fresh connection
            let client = McpClient::new("brainclaw", env!("CARGO_PKG_VERSION"));
            client.connect(&server_cfg).await?;
            let retry = client
                .call_tool(SERVER_NAME, tool_name, Some(arguments))
                .await?;
            *guard = Some(client);
            Ok(extract_text_content(retry.content))
        }
    }
}

/// Extract text from an MCP `CallToolResult` content list.
///
/// Serialises each `Content` item and extracts the `"text"` field — works for all
/// `RawContent::Text` variants without requiring rmcp as a direct dependency.
fn extract_text_content(content: Vec<brainwires_mcp_client::Content>) -> String {
    content
        .into_iter()
        .filter_map(|c| {
            if let Ok(obj) = serde_json::to_value(&*c) {
                obj.get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Tool definitions ─────────────────────────────────────────────────────────

/// Browser automation tools powered by Thalora (spawned as a subprocess via MCP).
pub struct BrowserTool;

impl BrowserTool {
    /// Return all browser tool definitions.
    pub fn get_tools() -> Vec<Tool> {
        vec![
            Self::browser_read_url_tool(),
            Self::browser_navigate_tool(),
            Self::browser_click_tool(),
            Self::browser_fill_tool(),
            Self::browser_eval_tool(),
            Self::browser_screenshot_tool(),
            Self::browser_search_tool(),
        ]
    }

    fn browser_read_url_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "url".into(),
            json!({"type":"string","description":"URL to navigate to and extract as markdown"}),
        );
        props.insert("wait_for_js".into(), json!({"type":"boolean","description":"Wait for JavaScript before extracting (default: false)"}));
        props.insert(
            "include_images".into(),
            json!({"type":"boolean","description":"Include image alt text (default: true)"}),
        );
        props.insert("max_output_size".into(), json!({"type":"number","description":"Max output characters; 0 = unlimited (default: 50000)"}));
        Tool {
            name: "browser_read_url".into(),
            description: "Navigate to a URL and return its content as clean readable markdown. One-shot — no session required.".into(),
            input_schema: ToolInputSchema::object(props, vec!["url".into()]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn browser_navigate_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "url".into(),
            json!({"type":"string","description":"URL to navigate to"}),
        );
        props.insert(
            "session_id".into(),
            json!({"type":"string","description":"Browser session ID"}),
        );
        props.insert(
            "wait_for_js".into(),
            json!({"type":"boolean","description":"Wait for JS to settle after navigation"}),
        );
        Tool {
            name: "browser_navigate".into(),
            description: "Navigate a browser session to a URL.".into(),
            input_schema: ToolInputSchema::object(props, vec!["url".into(), "session_id".into()]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn browser_click_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "selector".into(),
            json!({"type":"string","description":"CSS selector for the element to click"}),
        );
        props.insert(
            "session_id".into(),
            json!({"type":"string","description":"Browser session ID"}),
        );
        Tool {
            name: "browser_click".into(),
            description: "Click an element in a browser session by CSS selector.".into(),
            input_schema: ToolInputSchema::object(
                props,
                vec!["selector".into(), "session_id".into()],
            ),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn browser_fill_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "selector".into(),
            json!({"type":"string","description":"CSS selector for the input field"}),
        );
        props.insert(
            "value".into(),
            json!({"type":"string","description":"Value to fill into the field"}),
        );
        props.insert(
            "session_id".into(),
            json!({"type":"string","description":"Browser session ID"}),
        );
        Tool {
            name: "browser_fill".into(),
            description: "Fill an input field in a browser session.".into(),
            input_schema: ToolInputSchema::object(
                props,
                vec!["selector".into(), "value".into(), "session_id".into()],
            ),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn browser_eval_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "expression".into(),
            json!({"type":"string","description":"JavaScript expression to evaluate"}),
        );
        props.insert(
            "session_id".into(),
            json!({"type":"string","description":"Browser session ID (optional)"}),
        );
        props.insert(
            "return_by_value".into(),
            json!({"type":"boolean","description":"Return result as JSON value (default: true)"}),
        );
        Tool {
            name: "browser_eval".into(),
            description: "Execute JavaScript in the browser and return the result.".into(),
            input_schema: ToolInputSchema::object(props, vec!["expression".into()]),
            requires_approval: true, // JS execution always requires approval
            ..Default::default()
        }
    }

    fn browser_screenshot_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "session_id".into(),
            json!({"type":"string","description":"Browser session ID (optional)"}),
        );
        props.insert("format".into(), json!({"type":"string","enum":["png","jpeg"],"description":"Image format (default: png)"}));
        props.insert("quality".into(), json!({"type":"number","description":"JPEG quality 0–100 (only for format=jpeg, default: 80)"}));
        Tool {
            name: "browser_screenshot".into(),
            description: "Capture a screenshot of the current browser page as base64-encoded PNG."
                .into(),
            input_schema: ToolInputSchema::object(props, vec![]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn browser_search_tool() -> Tool {
        let mut props = HashMap::new();
        props.insert(
            "query".into(),
            json!({"type":"string","description":"Search query"}),
        );
        props.insert(
            "num_results".into(),
            json!({"type":"number","description":"Number of results (default: 10, max: 20)"}),
        );
        props.insert("search_engine".into(), json!({"type":"string","enum":["duckduckgo","bing","google","startpage"],"description":"Search engine (default: duckduckgo)"}));
        props.insert("time_range".into(), json!({"type":"string","enum":["day","week","month","year"],"description":"Limit results by recency"}));
        Tool {
            name: "browser_search".into(),
            description: "Search the web and return top results as titles, URLs, and snippets."
                .into(),
            input_schema: ToolInputSchema::object(props, vec!["query".into()]),
            requires_approval: false,
            ..Default::default()
        }
    }

    /// Execute a browser tool by name, forwarding to the Thalora subprocess via MCP.
    pub async fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        let result = call_thalora_tool(context, tool_name, input.clone()).await;
        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Browser tool '{}' failed: {}", tool_name, e),
            ),
        }
    }
}
