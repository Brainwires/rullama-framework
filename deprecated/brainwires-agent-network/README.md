# brainwires-agent-network (DEPRECATED)

This crate was **renamed** to `brainwires-network` in v0.10 — the
`agent-` prefix was dropped because the crate covers all agent and
non-agent networking (IPC, TCP, A2A, mesh, routing, discovery), not
just agent-to-agent transport.

| Old | New |
|---|---|
| `brainwires-agent-network` | [`brainwires-network`](https://crates.io/crates/brainwires-network) |

## Migration

### Cargo.toml

```toml
# Before
brainwires-agent-network = "0.8"

# After
brainwires-network = "0.11"
```

### Imports

All public types kept their names; only the crate-root path changed:

```rust
// Before
use brainwires_agent_network::{NetworkManager, Transport, MessageEnvelope};

// After
use brainwires_network::{NetworkManager, Transport, MessageEnvelope};
```

See [CHANGELOG.md](https://github.com/Brainwires/brainwires-framework/blob/main/CHANGELOG.md) for the full 0.10 → 0.11 history.
