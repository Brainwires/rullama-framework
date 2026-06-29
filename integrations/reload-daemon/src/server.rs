use crate::config::DaemonConfig;
use crate::reload;
use rmcp::{
    ServerHandler, handler::server::tool::ToolRouter, model::*, tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct ReloadServer {
    config: Arc<DaemonConfig>,
    tool_router: ToolRouter<Self>,
}

impl ReloadServer {
    pub fn new(config: Arc<DaemonConfig>) -> Self {
        Self {
            config,
            tool_router: Self::tool_router(),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReloadAppRequest {
    /// Key into the `clients` config map (e.g. "claude-code").
    pub client_type: String,
    /// PID of the process to kill.
    pub pid: i32,
    /// The original argv of the process (used to build restart args).
    pub original_args: Vec<String>,
    /// Working directory for the restarted process.
    pub working_directory: String,
}

#[tool_router(router = tool_router)]
impl ReloadServer {
    #[tool(
        description = "Kill the calling process and restart it with transformed arguments. \
        The restart strategy (signals, timeouts, arg transforms) is driven by config."
    )]
    async fn reload_app(
        &self,
        rmcp::handler::server::wrapper::Parameters(req): rmcp::handler::server::wrapper::Parameters<
            ReloadAppRequest,
        >,
    ) -> Result<String, String> {
        let strategy = self
            .config
            .clients
            .get(&req.client_type)
            .ok_or_else(|| format!("unknown client_type: {}", req.client_type))?;

        // Validate that the caller's binary name matches expected process_name.
        // (Defence against accidental mis-targeting.)
        if let Some(arg0) = req.original_args.first() {
            let bin_name = std::path::Path::new(arg0)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(arg0);
            if bin_name != strategy.process_name {
                return Err(format!(
                    "binary name mismatch: expected '{}', got '{}'",
                    strategy.process_name, bin_name
                ));
            }
        }

        // Build the restart args before killing (so we have everything ready).
        let program = req
            .original_args
            .first()
            .cloned()
            .ok_or_else(|| "original_args is empty".to_string())?;

        let restart_args = match &strategy.restart_args_transform {
            Some(transform) => reload::transform_args(&req.original_args[1..], transform),
            None => req.original_args[1..].to_vec(),
        };

        // Kill the process.
        reload::kill_process(req.pid, strategy).await?;

        // Spawn the replacement.
        reload::spawn_process(&program, &restart_args, &req.working_directory)?;

        Ok(format!(
            "Reloaded {} (pid {}) with args {:?}",
            req.client_type, req.pid, restart_args
        ))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ReloadServer {
    fn get_info(&self) -> ServerInfo {
        {
            let mut info = ServerInfo::default();
            info.capabilities = ServerCapabilities::builder().enable_tools().build();
            info.server_info = Implementation::new("reload-daemon", env!("CARGO_PKG_VERSION"))
                .with_title("Reload Daemon — process restart MCP server");
            info.instructions = Some(
                "Exposes a single tool `reload_app` that kills a process and restarts it \
                 with transformed arguments. Restart strategies are config-driven."
                    .into(),
            );
            info
        }
    }
}
