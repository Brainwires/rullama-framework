# brainwires-memory

[![Crates.io](https://img.shields.io/crates/v/brainwires-memory.svg)](https://crates.io/crates/brainwires-memory)
[![Documentation](https://docs.rs/brainwires-memory/badge.svg)](https://docs.rs/brainwires-memory)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

Tiered hot/warm/cold agent memory **orchestration** for the
Brainwires Agent Framework.

## What this crate is

The schema layer — the five tier stores (`MessageStore`,
`SummaryStore`, `FactStore`, `MentalModelStore`, `TierMetadataStore`)
plus the shared `tier_types` (`MemoryTier`, `MemoryAuthority`,
`TierMetadata`, `MessageSummary`, `KeyFact`, `FactType`) — lives in
[`brainwires-stores`](https://crates.io/crates/brainwires-stores).
This crate adds:

- **`TieredMemory`** — multi-factor adaptive search across all four
  tiers (similarity × recency × importance), with promotion / demotion
  of entries when access patterns change.
- **`dream`** *(feature-gated)* — offline consolidation engine that
  summarises hot-tier messages into warm-tier summaries, extracts
  cold-tier facts, and demotes by retention score.

## Why a separate crate

`brainwires-stores` is schema only — table definitions and CRUD. The
orchestration (`TieredMemory`) and consolidation (`dream`) live here
because they are **engines**, not schema: search policy, scoring
weights, write authority gating (`CanonicalWriteToken`), demotion
heuristics, summarisation pipelines.

A consumer that only needs to write rows directly to one tier (e.g. a
test) depends on `brainwires-stores` alone. A consumer that wants
multi-tier adaptive search or offline consolidation pulls
`brainwires-memory`, which re-exports the schema types it operates
over.

## Feature flags

- `dream` — offline consolidation engine. Pulls `futures` for the
  async cycle plumbing.
- `telemetry` *(implies `dream`)* — wire dream's `with_analytics` hook
  to `brainwires-telemetry`'s `AnalyticsCollector`.

## License

MIT OR Apache-2.0
