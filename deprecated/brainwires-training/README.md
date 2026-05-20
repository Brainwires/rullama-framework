# brainwires-training (DEPRECATED)

This crate has been **moved** to
[`rullama-training`](https://github.com/Brainwires/rullama) in the sibling
`rullama` workspace.

In v0.10 this crate was renamed *from* a fine-tune-pipeline crate *into* a
placeholder for from-scratch training work; the fine-tune content moved to
[`brainwires-finetune`](https://crates.io/crates/brainwires-finetune).
In v0.11.0 the placeholder itself moved out to `rullama-training`, alongside
the broader low-level training stack rullama owns.

There is no re-export shim — depending on this crate gets you nothing.

## Migration

```toml
# Before
brainwires-training = "0.10"

# After (depend on rullama directly)
rullama-training = { git = "https://github.com/Brainwires/rullama" }
```

For cloud fine-tune APIs (OpenAI, Anthropic, Vertex, Bedrock), use
[`brainwires-finetune`](https://crates.io/crates/brainwires-finetune) instead.
