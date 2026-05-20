# brainwires-tools (DEPRECATED)

This crate has been **split in two** as of v0.11. There is no re-export
shim — depending on this crate gets you nothing.

| Old | New |
|---|---|
| `brainwires-tools` (executor / registry / sanitization / smart_router / tool_search / transaction / validation / orchestrator / oauth / openapi / sandbox_executor / sessions / tool_embedding) | [`brainwires-tool-runtime`](https://crates.io/crates/brainwires-tool-runtime) |
| `brainwires-tools` (bash / file_ops / git / web / search / code_exec / interpreters / browser / email / calendar / system / semantic_search / `BuiltinToolExecutor`) | [`brainwires-tool-builtins`](https://crates.io/crates/brainwires-tool-builtins) |

## Migration

### Cargo.toml

```toml
# Before
brainwires-tools = "0.10"

# After — pick whichever sub-crate you actually need.
# brainwires-tool-builtins already pulls brainwires-tool-runtime as a dep,
# so most consumers can take just the builtins crate.
brainwires-tool-runtime = "0.11"
brainwires-tool-builtins = "0.11"
```

### Imports

| Before | After |
|---|---|
| `brainwires_tools::ToolExecutor`            | `brainwires_tool_runtime::ToolExecutor` |
| `brainwires_tools::ToolRegistry`            | `brainwires_tool_runtime::ToolRegistry` |
| `brainwires_tools::ToolCategory`            | `brainwires_tool_runtime::ToolCategory` |
| `brainwires_tools::Tool*Error*`             | `brainwires_tool_runtime::*` |
| `brainwires_tools::sanitization::*`         | `brainwires_tool_runtime::sanitization::*` |
| `brainwires_tools::smart_router::*`         | `brainwires_tool_runtime::smart_router::*` |
| `brainwires_tools::ToolSearchTool`          | `brainwires_tool_runtime::ToolSearchTool` |
| `brainwires_tools::TransactionManager`      | `brainwires_tool_runtime::TransactionManager` |
| `brainwires_tools::ValidationTool`          | `brainwires_tool_runtime::ValidationTool` |
| `brainwires_tools::OrchestratorTool`        | `brainwires_tool_runtime::OrchestratorTool` |
| `brainwires_tools::SandboxedToolExecutor`   | `brainwires_tool_runtime::SandboxedToolExecutor` |
| `brainwires_tools::SessionsTool` (etc.)     | `brainwires_tool_runtime::SessionsTool` |
| `brainwires_tools::oauth::*`                | `brainwires_tool_runtime::oauth::*` |
| `brainwires_tools::openapi::*`              | `brainwires_tool_runtime::openapi::*` |
| `brainwires_tools::ToolEmbeddingIndex`      | `brainwires_tool_runtime::ToolEmbeddingIndex` |
| `brainwires_tools::BuiltinToolExecutor`     | `brainwires_tool_builtins::BuiltinToolExecutor` |
| `brainwires_tools::BashTool`                | `brainwires_tool_builtins::BashTool` |
| `brainwires_tools::FileOpsTool`             | `brainwires_tool_builtins::FileOpsTool` |
| `brainwires_tools::GitTool`                 | `brainwires_tool_builtins::GitTool` |
| `brainwires_tools::WebTool`                 | `brainwires_tool_builtins::WebTool` |
| `brainwires_tools::SearchTool`              | `brainwires_tool_builtins::SearchTool` |
| `brainwires_tools::CodeExecTool`            | `brainwires_tool_builtins::CodeExecTool` |
| `brainwires_tools::SemanticSearchTool`      | `brainwires_tool_builtins::SemanticSearchTool` |
| `brainwires_tools::BrowserTool`             | `brainwires_tool_builtins::BrowserTool` |
| `brainwires_tools::EmailTool`               | `brainwires_tool_builtins::EmailTool` |
| `brainwires_tools::CalendarTool`            | `brainwires_tool_builtins::CalendarTool` |
| `brainwires_tools::interpreters::*`         | `brainwires_tool_builtins::interpreters::*` |
| `brainwires_tools::system::*`               | `brainwires_tool_builtins::system::*` |
| `brainwires_tools::registry_with_builtins()` | `brainwires_tool_builtins::registry_with_builtins()` |

### Cargo features

The `brainwires-tools` features map 1:1 to the same-named features on
the new crates:

- runtime-side (`orchestrator`, `orchestrator-wasm`, `oauth`, `openapi`,
  `sandbox`, `sessions`) are on `brainwires-tool-runtime`.
- builtin-side (`rag`, `interpreters`, `interpreters-rhai`,
  `interpreters-lua`, `interpreters-js`, `interpreters-all`,
  `interpreters-wasm`, `email`, `calendar`, `browser`, `system`,
  `reactor`, `services`, `system-full`) are on `brainwires-tool-builtins`.
- `native` and `wasm` exist on both.

## Why the split

`brainwires-tools` had grown into 22 source files + 6 subdirs + 32
features mixing two unrelated concerns: a tool-execution **framework**
(executor / registry / dispatch) and 20+ concrete **builtin tools**
(bash / git / web / …). Every consumer that wanted the framework had to
compile every builtin's deps (lettre, async-imap, icalendar, mlua,
boa_engine, notify, rhai, …). Splitting lets you depend on the runtime
without dragging in any builtin tool.

See [ADR-0001](../../docs/adr/ADR-0001-crate-split-discipline.md) for
the project-wide rule on splits like this.
