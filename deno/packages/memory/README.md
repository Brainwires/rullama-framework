# @rullama/memory

Tiered memory orchestration for the Brainwires Agent Framework.

In v0.11.0 this split out of `@rullama/storage` to mirror the Rust
`rullama-memory` crate. Tier substrate (StorageBackend trait, embeddings,
domain-store schemas) stays in `@rullama/storage` and `@rullama/stores`;
this package layers retention, multi-factor scoring, and hot/warm/cold tier flow
on top.
