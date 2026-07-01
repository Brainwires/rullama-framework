# @rullama/stores

Domain stores built on top of `@rullama/storage`'s `StorageBackend` trait.

In v0.11.0 these were extracted out of `@rullama/storage` to mirror the Rust
restructure (`rullama-stores`). The schemas live here; the underlying backend
traits (Postgres / Qdrant / SurrealDB / Pinecone / etc.) remain in
`@rullama/storage`.

## Stores

- **Message store** — chat history with metadata
- **Conversation store** — multi-turn conversation aggregates
- **Task store** — task graph + agent state
- **Plan store** — saved Plan-Work-Judge plan instances
- **Template store** — reusable plan templates with variable substitution

Tiered memory orchestration lives in `@rullama/memory`.
