# brainwires-tool-system (DEPRECATED)

This crate was an interim split from the system-tools work that was
**reabsorbed in v0.10** — first into `brainwires-tools`, then in v0.11
into the post-split `brainwires-tool-builtins::system` module behind
the `system` feature.

| Old | New |
|---|---|
| `brainwires-tool-system` | [`brainwires-tool-builtins`](https://crates.io/crates/brainwires-tool-builtins) with `features = ["system"]` |

## Migration

### Cargo.toml

```toml
# Before
brainwires-tool-system = "0.8"

# After
brainwires-tool-builtins = { version = "0.11", features = ["system"] }
```

### Imports

```rust
// Before
use brainwires_tool_system::*;

// After
use brainwires_tool_builtins::system::*;
```

See [CHANGELOG.md](https://github.com/Brainwires/rullama-framework/blob/main/CHANGELOG.md) and the [`brainwires-tools` tombstone](../brainwires-tools/README.md) for the wider split history.
