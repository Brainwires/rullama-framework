# brainwires-mcp (DEPRECATED)

This crate has been **renamed** to
[`brainwires-mcp-client`](https://crates.io/crates/brainwires-mcp-client)
to disambiguate from
[`brainwires-mcp-server`](https://crates.io/crates/brainwires-mcp-server).

The old name was ambiguous — `brainwires-mcp` actually held the *client*
side (plus shared protocol types), but the asymmetry with
`brainwires-mcp-server` made that non-obvious. The pair is now
`mcp-client` / `mcp-server`.

There is no re-export shim — depending on this crate gets you nothing.

## Migration

```toml
# Before
brainwires-mcp = "0.10"

# After
brainwires-mcp-client = "0.11"
```

```rust
// Before
use brainwires_mcp::{McpClient, JsonRpcRequest, ...};

// After
use brainwires_mcp_client::{McpClient, JsonRpcRequest, ...};
```

The public API is otherwise unchanged.

## Note on shared types

`brainwires-mcp-server` continues to depend on `brainwires-mcp-client`
for the shared protocol types (`CallToolResult`, `JsonRpcRequest`,
etc.). That's a slightly odd dep edge (server depending on client),
acceptable until/unless a future refactor extracts a separate
`brainwires-mcp-types` crate.
