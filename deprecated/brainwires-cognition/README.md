# brainwires-cognition (DEPRECATED)

This crate was **renamed/absorbed into `brainwires-knowledge`** in v0.10.
Per the Cargo.toml description: "use brainwires-knowledge instead".

| Old | New |
|---|---|
| `brainwires-cognition` | [`brainwires-knowledge`](https://crates.io/crates/brainwires-knowledge) |

## Migration

### Cargo.toml

```toml
# Before
brainwires-cognition = "0.8"

# After
brainwires-knowledge = "0.11"
```

### Imports

```rust
// Before
use brainwires_cognition::*;

// After
use brainwires_knowledge::*;
```

See [CHANGELOG.md](https://github.com/Brainwires/brainwires-framework/blob/main/CHANGELOG.md) for the full migration story.
