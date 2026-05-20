# brainwires-permissions (DEPRECATED)

This crate has been **renamed** to
[`brainwires-permission`](https://crates.io/crates/brainwires-permission)
(singular) for consistency with the framework's naming rule (singular
nouns for capability domains: `brainwires-agent`, `brainwires-permission`,
`brainwires-provider`, `brainwires-tool-runtime`, …).

There is no re-export shim — depending on this crate gets you nothing.

## Migration

```toml
# Before
brainwires-permissions = "0.10"

# After
brainwires-permission = "0.11"
```

```rust
// Before
use brainwires_permissions::{PolicyEngine, AuditLogger, TrustManager};

// After
use brainwires_permission::{PolicyEngine, AuditLogger, TrustManager};
```

The public API is otherwise unchanged.
