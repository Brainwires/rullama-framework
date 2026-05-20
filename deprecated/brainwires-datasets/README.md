# brainwires-datasets (DEPRECATED)

Dataset pipelines were **absorbed into `brainwires-finetune`** in v0.10
and now live under `brainwires_finetune::datasets`. The local-PEFT
side of training moved to the sibling `rullama` workspace in v0.11;
cloud fine-tune APIs + datasets stay on `brainwires-finetune`.

| Old | New |
|---|---|
| `brainwires-datasets` | [`brainwires-finetune`](https://crates.io/crates/brainwires-finetune) |

## Migration

### Cargo.toml

```toml
# Before
brainwires-datasets = "0.8"

# After
brainwires-finetune = "0.11"
```

### Imports

```rust
// Before
use brainwires_datasets::*;

// After
use brainwires_finetune::datasets::*;
```

See [CHANGELOG.md](https://github.com/Brainwires/brainwires-framework/blob/main/CHANGELOG.md) for the dataset + training split.
