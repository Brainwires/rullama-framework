# brainwires-finetune-local (DEPRECATED)

This crate has been **moved** to
[`rullama-finetune`](https://github.com/Brainwires/rullama) in the sibling
`rullama` workspace.

In v0.11.0 the framework shed its low-level training layer. The Burn-backed
LoRA / QLoRA / DoRA pipeline that lived here now lives in `rullama-finetune`,
behind a `training` feature. The framework keeps its high-level surface
(agents, providers, MCP, RAG) and stops carrying a model runtime of its own.

There is no re-export shim — depending on this crate gets you nothing.

## Migration

```toml
# Before
brainwires-finetune-local = "0.10"

# After (depend on rullama directly)
rullama-finetune = { git = "https://github.com/Brainwires/rullama", features = ["training"] }
```

```rust
// Before
use brainwires_finetune_local::{LoraTrainer, QloraConfig};

// After
use rullama_finetune::{LoraTrainer, QloraConfig};
```

The high-level cloud fine-tune APIs remain in
[`brainwires-finetune`](https://crates.io/crates/brainwires-finetune) — they
were never part of `brainwires-finetune-local`.
