# ADR-0005 — Stores consolidation: framework ships an opinionated minimum store set

- **Status:** Accepted
- **Date:** 2026-05-03
- **Authors:** Brainwires

## Context

Before Phase 10, the framework's data-store organization was scattered:

- `crates/brainwires-session/` (648 LOC) — `SessionStore` + sqlite/in-memory impls.
- `crates/brainwires-memory/` (~2.7k LOC) — five tier schema stores
  (`MessageStore`, `SummaryStore`, `FactStore`, `MentalModelStore`,
  `TierMetadataStore`) plus the `TieredMemory` orchestrator and the
  `dream` offline consolidation engine.
- `extras/brainwires-cli/src/storage/` (~4.2k LOC, 10 files) —
  `ConversationStore`, `TaskStore` / `AgentStateStore`, `PlanStore`,
  `TemplateStore`, `LockStore`, `ImageStore`, `PatternStore`,
  `PlanModeStore`, `PersistentTaskManager`. Parked here in Phase 5
  under the framing **"the framework stays minimal — CLI-only stores
  belong in the CLI."**

Audit during Phase 10 planning showed 8 of those 10 CLI stores had no
CLI-specific imports — they were generic `<B: StorageBackend = LanceDatabase>`
patterns built on framework primitives, parked in the CLI only because
no framework-side consumer existed yet.

The Phase 5 framing turned out to be wrong-shaped. Anyone building an
agent system on the framework needs sessions, tasks, plans, locks,
conversations, images, and tier memory. Forcing each consumer to
reinvent or copy-paste those primitives — because they happened to live
in the CLI — produced a worse outcome than shipping them.

A second issue surfaced during execution: an early Phase 10a draft
folded `dream` (offline consolidation) into the new stores crate.
That mixed engines into what was intended as schema. The course
correction was to keep the **schema vs. orchestration** boundary
sharp.

## Decision

Adopt a three-layer storage architecture, replacing the previous flat
"two small + one CLI-internal" arrangement:

- **`brainwires-storage`** — substrate. `StorageBackend` trait,
  backends (LanceDB, BM25, embeddings, file context), unchanged from
  before.
- **`brainwires-stores`** — opinionated minimum **schema + CRUD** set.
  Sessions, tasks, plans, conversations, locks, images, templates +
  the five tier-schema stores + shared `tier_types`. Default features:
  `session`, `task`, `plan`, `conversation`. Opt-in: `memory`, `lock`,
  `image`, `sqlite`. Built on the substrate.
- **`brainwires-memory`** — orchestration. `TieredMemory` (multi-factor
  adaptive search across tiers + promotion / demotion), the
  `CanonicalWriteToken` capability gate, and the `dream` consolidation
  engine. Depends on `brainwires-stores` for the schema types.

CLI-domain stores that genuinely belong in the CLI stay there:
`PlanModeStore` (couples to CLI message / plan-mode types) and
`PersistentTaskManager` (CLI-local helper around
`brainwires-agent::task_manager::TaskManager`). The SEAL pattern store
+ `LanceDatabaseExt` extension trait moved to
`crates/brainwires-agent/src/seal/pattern_store.rs` — its types
(`QueryPattern`, `QuestionType`) are SEAL-internal, and putting it in
the schema crate would create a `stores → agent → stores` cycle.

`extras/brainwires-cli/src/storage/mod.rs` is kept as a thin re-export
aggregator so the 29 CLI files using `crate::storage::{...}` don't
need an immediate import rewrite. That shim is a candidate for later
deletion when somebody wants to do the mass rewrite.

## Consequences

- **Positive.**
  - Framework consumers now get a coherent, opinionated minimum
    store set without having to depend on a CLI binary or copy-paste
    primitives.
  - The **schema vs. orchestration** boundary is sharp: a consumer
    wanting only to write rows to one tier depends on
    `brainwires-stores` alone; a consumer wanting multi-tier search
    or offline consolidation pulls `brainwires-memory`, which
    re-exports the schema it operates over.
  - `brainwires-stores` is feature-gated per store, so consumers only
    pay compile cost for what they use. `default = ["session", "task",
    "plan", "conversation"]` covers the common case.
  - The PatternStore relocation puts a SEAL primitive next to its
    types, removing the cross-crate cycle that would otherwise
    appear.
- **Negative.**
  - This **reverses** the Phase 5 stance documented earlier in
    [`crates/brainwires/Cargo.toml`](../../crates/brainwires/Cargo.toml)
    comments and the workspace-layout README section. Anyone reading
    older commits should know the principle changed.
  - The CLI's `crate::storage` namespace is now a thin re-export shim;
    until the 29-file consumer rewrite happens, the shim is duplicated
    surface area.
  - More crates in the workspace member list (was 26 in the brief
    Phase-10a fold-in state, back to 27 after Phase 10b's memory
    revival).
- **Neutral / follow-up.**
  - 29-file CLI import rewrite is queued for later cleanup; not
    blocking.
  - `PersistentTaskManager` has zero in-tree consumers as of
    Phase 10b — could be deleted in a follow-up cleanup if no
    consumers materialize.
  - The default-on `lance-backend` for `brainwires-stores`'s memory
    feature pulls a heavy dep tree; consumers who only need the
    in-memory session store should set `default-features = false` and
    enable `session` alone.

## Alternatives considered

- **Keep the Phase-5 status quo (CLI owns the stores).** Rejected
  because no framework-side consumer of those primitives could exist
  without depending on the CLI binary; the framework was hiding
  generally-useful schema in an application crate.
- **Fold `TieredMemory` + `dream` into `brainwires-stores` as
  feature-gated modules.** This was the Phase 10a draft. Rejected
  on review — it conflated schema (rows + CRUD) with engines
  (search policy, summarisation, demotion logic). Splitting
  preserves the clarity that "stores = data shape; memory = how the
  framework uses that data."
- **Fold `TieredMemory` + `dream` into `brainwires-agent` under a
  `memory` feature.** Rejected because memory orchestration is not
  agent-specific — non-agent consumers (e.g. CLI session compaction)
  also pull `TieredMemory`. Tying it to the agent crate would force
  those consumers through a heavier dep.
- **Bulk-rewrite the 29 CLI consumer files now to drop the
  `crate::storage` re-export shim.** Deferred; the shim is
  zero-overhead and the rewrite has no functional benefit beyond
  cleanup. Doing it later, in its own PR, keeps Phase 10b focused.

## References

- Phase 10a commit: `bf02aaf` — folded session + memory into stores.
- Phase 10b commit: `0ae17d9` — pulled 8 CLI stores up + revived
  brainwires-memory as orchestration.
- Plan: `~/.claude/plans/write-a-full-plan-zesty-harbor.md` Phase 10
  section.
- ADR-0001 — crate split discipline (the meta-rule that requires this
  ADR).
- ADR-0004 — framework / extras boundary (the rule whose application
  changed when the CLI-stores reasoning was revisited).
