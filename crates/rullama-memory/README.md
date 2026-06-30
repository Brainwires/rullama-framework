# rullama-memory

[![Crates.io](https://img.shields.io/crates/v/rullama-memory.svg)](https://crates.io/crates/rullama-memory)
[![Documentation](https://docs.rs/rullama-memory/badge.svg)](https://docs.rs/rullama-memory)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/rullama-framework)

Tiered hot/warm/cold agent memory **orchestration** for the
rullama agent framework.

## What this crate is

The schema layer — the five tier stores (`MessageStore`,
`SummaryStore`, `FactStore`, `MentalModelStore`, `TierMetadataStore`)
plus the shared `tier_types` (`MemoryTier`, `MemoryAuthority`,
`TierMetadata`, `MessageSummary`, `KeyFact`, `FactType`) — lives in
[`rullama-stores`](https://crates.io/crates/rullama-stores).
This crate adds:

- **`TieredMemory`** — multi-factor adaptive search across all four
  tiers (similarity × recency × importance), with promotion / demotion
  of entries when access patterns change.
- **`dream`** *(feature-gated)* — offline consolidation engine that
  summarises hot-tier messages into warm-tier summaries, extracts
  cold-tier facts, and demotes by retention score.

## Why a separate crate

`rullama-stores` is schema only — table definitions and CRUD. The
orchestration (`TieredMemory`) and consolidation (`dream`) live here
because they are **engines**, not schema: search policy, scoring
weights, write authority gating (`CanonicalWriteToken`), demotion
heuristics, summarisation pipelines.

A consumer that only needs to write rows directly to one tier (e.g. a
test) depends on `rullama-stores` alone. A consumer that wants
multi-tier adaptive search or offline consolidation pulls
`rullama-memory`, which re-exports the schema types it operates
over.

## Feature flags

- `dream` — offline consolidation engine. Pulls `futures` for the
  async cycle plumbing.
- `telemetry` *(implies `dream`)* — wire dream's `with_analytics` hook
  to `rullama-telemetry`'s `AnalyticsCollector`.

## License

MIT OR Apache-2.0
