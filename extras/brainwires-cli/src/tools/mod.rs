// Tools module - built-in tool implementations
//
// Re-exports from the brainwires-tools framework crate, plus CLI-specific tools.
#![allow(hidden_glob_reexports)]

pub use brainwires::tools::*;

// CLI-specific tool modules
mod agent_pool;
mod ask;
mod context_recall;
mod executor;
mod mcp_tool;
mod memory;
mod monitor;
mod plan;
mod session_task;
pub mod smart_router;
mod task_manager;

// Re-export wrappers preserving module paths used elsewhere in CLI
pub mod error;
pub mod validation_tools;

pub use agent_pool::*;
pub use ask::AskUserQuestionTool;
pub use context_recall::*;
pub use mcp_tool::*;
pub use memory::MemoryTool;
pub use monitor::MonitorTool;
pub use plan::*;
pub use session_task::*;
pub use smart_router::{analyze_messages, get_smart_tools, get_smart_tools_with_mcp};
pub use task_manager::*;
pub use validation_tools::*;

// Explicitly re-export the CLI's concrete ToolExecutor struct so it shadows
// the brainwires_tool_runtime::ToolExecutor trait that enters via the glob above.
pub use executor::ToolExecutor;

// ── CLI-level tool-selection flag ─────────────────────────────────────────
// Non-TUI chat paths default to the curated core set (14 tools including
// `search_tools`) so outbound request bodies stay small and get Anthropic
// prompt-cache hits. Users who want every registered tool enumerated up
// front can flip this once at startup via `--all-tools`.

use std::sync::atomic::{AtomicBool, Ordering};

static ALL_TOOLS_OVERRIDE: AtomicBool = AtomicBool::new(false);

/// Opt into eager enumeration of every registered tool (bypasses the curated
/// core set). Set once at startup from `--all-tools`.
pub fn set_all_tools_override(enabled: bool) {
    ALL_TOOLS_OVERRIDE.store(enabled, Ordering::Relaxed);
}

/// Was `--all-tools` requested?
pub fn all_tools_override() -> bool {
    ALL_TOOLS_OVERRIDE.load(Ordering::Relaxed)
}

/// Return the tool set to send to the provider for a non-TUI chat path.
/// Honors `--all-tools` when set; otherwise returns the curated core set
/// in canonical order (stable prefix for prompt caching).
pub fn select_non_tui_tools(registry: &brainwires_tool_runtime::ToolRegistry) -> Vec<Tool> {
    if all_tools_override() {
        registry.get_all().to_vec()
    } else {
        registry.get_core().into_iter().cloned().collect()
    }
}
