# brainwires-channels (DEPRECATED)

Universal messaging channels were **absorbed into `brainwires-network`**
in v0.10 (PR 3 of the framework restructuring). The channels code lives
under `brainwires_network::channels` and is gated behind the
`channels` feature.

| Old | New |
|---|---|
| `brainwires-channels` | [`brainwires-network`](https://crates.io/crates/brainwires-network) with `features = ["channels"]` |

## Migration

### Cargo.toml

```toml
# Before
brainwires-channels = "0.8"

# After
brainwires-network = { version = "0.11", features = ["channels"] }
```

### Imports

```rust
// Before
use brainwires_channels::*;

// After
use brainwires_network::channels::*;
```

See [CHANGELOG.md](https://github.com/Brainwires/brainwires-framework/blob/main/CHANGELOG.md) for the absorption rationale.
