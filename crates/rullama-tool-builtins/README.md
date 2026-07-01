# rullama-tool-builtins

[![Crates.io](https://img.shields.io/crates/v/rullama-tool-builtins.svg)](https://crates.io/crates/rullama-tool-builtins)
[![Documentation](https://docs.rs/rullama-tool-builtins/badge.svg)](https://docs.rs/rullama-tool-builtins)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/rullama-framework)

Concrete builtin tools for rullama agents. Built on
[`rullama-tool-runtime`](https://crates.io/crates/rullama-tool-runtime),
which provides the `ToolExecutor` / `ToolRegistry` framework these tools
plug into.

## What lives here

### Always available (`native` feature)

| Tool | Purpose |
|---|---|
| `BashTool` | Shell command execution with proactive output management |
| `FileOpsTool` | File read / write / edit / patch / list / search / delete / mkdir |
| `GitTool` | Git operations (status, diff, log, stage, commit, push, pull, …) |
| `WebTool` | URL fetching |
| `SearchTool` | Regex-based code search (respects `.gitignore`) |
| `BuiltinToolExecutor` | `ToolExecutor` impl that hardcodes dispatch to all builtins |

Plus `registry_with_builtins()` — convenience constructor that returns a
`ToolRegistry` pre-populated with every concrete builtin gated on by the
active feature set.

### Feature-gated builtins

| Feature | Tool | Notes |
|---|---|---|
| `interpreters` (`-rhai`, `-lua`, `-js`, `-all`, `-wasm`) | `CodeExecTool` + `interpreters::*` | Sandboxed multi-language code execution |
| `rag` | `SemanticSearchTool` | RAG-powered codebase search (pulls `rullama-rag`) |
| `email` | `EmailTool` + `gmail_push` | IMAP / SMTP / Gmail Push |
| `calendar` | `CalendarTool` | Google Calendar / CalDAV |
| `browser` | `BrowserTool` | Headless browser via the MCP Thalora subprocess |
| `system` | `system::*` | OS-level primitives — fs event watching, service management |

## Usage

```toml
[dependencies]
rullama-tool-builtins = "0.12"
```

```rust,ignore
use rullama_tool_builtins::{BashTool, BuiltinToolExecutor, registry_with_builtins};
use rullama_tool_runtime::{ToolExecutor, ToolRegistry};
use rullama_core::ToolContext;

let registry = registry_with_builtins();
let executor = BuiltinToolExecutor::new(registry, ToolContext::default());
```

## See also

- [`rullama-tool-runtime`](https://crates.io/crates/rullama-tool-runtime) — the executor / registry framework these tools plug into.
- [`rullama`](https://crates.io/crates/rullama) — umbrella facade with
  `tools` feature that re-exports both crates.

## License

MIT OR Apache-2.0
