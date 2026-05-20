# brainwires-tool-runtime

[![Crates.io](https://img.shields.io/crates/v/brainwires-tool-runtime.svg)](https://crates.io/crates/brainwires-tool-runtime)
[![Documentation](https://docs.rs/brainwires-tool-runtime/badge.svg)](https://docs.rs/brainwires-tool-runtime)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

The execution-runtime layer for Brainwires tool dispatch. Companion crate
to [`brainwires-tool-builtins`](https://crates.io/crates/brainwires-tool-builtins),
which provides the concrete `bash` / `file_ops` / `git` / `web` / etc. tools
that this runtime knows how to dispatch.

## What lives here

- `executor::ToolExecutor` — the trait every tool dispatcher implements.
- `registry::ToolRegistry` + `ToolCategory` — composable tool registry and
  category metadata.
- `error` — tool-error taxonomy with retry classification.
- `sanitization` — content-source tagging, injection detection,
  sensitive-data redaction.
- `tool_search::ToolSearchTool` — meta-tool for keyword / regex / semantic
  tool discovery.
- `smart_router` — query-driven category filtering.
- `transaction::TransactionManager` — idempotency + staging for
  file-mutating tools (native).
- `validation::ValidationTool` — duplicate / syntax / build checks (native).

### Feature-gated runtime modules

| Feature | Module | Notes |
|---|---|---|
| `orchestrator` (or `orchestrator-wasm`) | `orchestrator::OrchestratorTool` | Rhai script executor |
| `oauth` | `oauth` | OAuth 2.0 client, PKCE, pluggable token store |
| `openapi` | `openapi` | OpenAPI 3 spec → tool descriptor conversion |
| `sandbox` | `sandbox_executor::SandboxedToolExecutor` | Wrap any executor to route bash/code-exec through `brainwires-sandbox` |
| `sessions` | `sessions::SessionsTool` | `sessions_list` / `sessions_history` / `sessions_send` / `sessions_spawn` over a `brainwires-session::SessionBroker` |
| `rag` | `tool_embedding::ToolEmbeddingIndex` | RAG-backed semantic mode for `ToolSearchTool` |

## Usage

```toml
[dependencies]
brainwires-tool-runtime = "0.11"
# Or, for the standard built-in tools too:
brainwires-tool-builtins = "0.11"  # already pulls brainwires-tool-runtime
```

```rust,ignore
use brainwires_tool_runtime::{ToolExecutor, ToolRegistry};
use brainwires_tool_builtins::{registry_with_builtins, BuiltinToolExecutor};

let registry = registry_with_builtins();
let executor = BuiltinToolExecutor::new(registry, Default::default());
```

## See also

- [`brainwires-tool-builtins`](https://crates.io/crates/brainwires-tool-builtins) — concrete tools that plug into this runtime.
- [`brainwires`](https://crates.io/crates/brainwires) — the umbrella facade
  with `tools` feature that re-exports both crates.

## License

MIT OR Apache-2.0
