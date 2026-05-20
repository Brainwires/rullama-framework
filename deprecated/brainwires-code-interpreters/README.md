# brainwires-code-interpreters (DEPRECATED)

Sandboxed multi-language code interpreters (Python, JavaScript, Lua,
Rhai, remote, Docker) were **absorbed into `brainwires-tool-builtins`**
in v0.10 and now live under `brainwires_tool_builtins::interpreters`
behind the `interpreters` feature.

| Old | New |
|---|---|
| `brainwires-code-interpreters` | [`brainwires-tool-builtins`](https://crates.io/crates/brainwires-tool-builtins) with `features = ["interpreters"]` |

## Migration

### Cargo.toml

```toml
# Before
brainwires-code-interpreters = "0.8"

# After
brainwires-tool-builtins = { version = "0.11", features = ["interpreters"] }
```

### Imports

```rust
// Before
use brainwires_code_interpreters::{Executor, ExecutionRequest, Language};

// After
use brainwires_tool_builtins::interpreters::{Executor, ExecutionRequest, Language};
```

Fine-grained features (`interpreters-rhai`, `interpreters-lua`,
`interpreters-js`, `interpreters-all`, `interpreters-wasm`) remain
available on the new crate. See [CHANGELOG.md](https://github.com/Brainwires/brainwires-framework/blob/main/CHANGELOG.md).
