#![deprecated(
    since = "0.10.1",
    note = "split into `brainwires-tool-runtime` (executor + registry + framework) and `brainwires-tool-builtins` (concrete bash / file_ops / git / web / search / code_exec / browser / email / calendar / system tools). The 0.10.0 façade was removed in 0.11 to break the implicit coupling. Migrate per the README at https://github.com/Brainwires/brainwires-framework/tree/main/deprecated/brainwires-tools"
)]
//! `brainwires-tools` is **deprecated** as of 0.10.1.
//!
//! It has been split into two crates:
//!
//! - [`brainwires-tool-runtime`](https://docs.rs/brainwires-tool-runtime) —
//!   the execution-runtime layer (`ToolExecutor`, `ToolRegistry`,
//!   error taxonomy, sanitization, validation, transactions, smart router,
//!   plus optional orchestrator / OAuth / OpenAPI / sandbox / sessions /
//!   RAG-tool modules).
//! - [`brainwires-tool-builtins`](https://docs.rs/brainwires-tool-builtins) —
//!   the concrete builtin tools (`bash`, `file_ops`, `git`, `web`,
//!   `search`, `code_exec` + `interpreters`, `semantic_search`,
//!   `browser`, `email`, `calendar`, `system`) and the
//!   `BuiltinToolExecutor` that hardcodes dispatch to them.
//!
//! There is no re-export shim — depending on this crate gives you nothing.
//! Switch your `Cargo.toml` and your imports per the migration table in
//! the crate README.
