# brainwires-system (DEPRECATED)

System tools (process / fs / env helpers) were **absorbed into
`brainwires-tool-builtins`** in v0.10 and now live under
`brainwires_tool_builtins::system` behind the `system` feature.

| Old | New |
|---|---|
| `brainwires-system` | [`brainwires-tool-builtins`](https://crates.io/crates/brainwires-tool-builtins) with `features = ["system"]` |

## Migration

### Cargo.toml

```toml
# Before
brainwires-system = "0.8"

# After
brainwires-tool-builtins = { version = "0.11", features = ["system"] }
```

### Imports

```rust
// Before
use brainwires_system::*;

// After
use brainwires_tool_builtins::system::*;
```

The companion `system-full` feature is still available for the
extended toolset. See [CHANGELOG.md](https://github.com/Brainwires/rullama-framework/blob/main/CHANGELOG.md).
