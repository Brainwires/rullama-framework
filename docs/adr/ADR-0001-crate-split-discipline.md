# ADR-0001 — Crate split discipline

- **Status:** Accepted
- **Date:** 2026-05-02
- **Authors:** Brainwires

## Context

The `deprecated/` directory at the workspace root holds 32 crates that
were previously split out of the framework and have since been re-merged
into a small number of "god-crates":

- `brainwires-prompting` and `brainwires-rag` were merged into
  `brainwires-knowledge`.
- `brainwires-mdap`, `brainwires-seal`, and `brainwires-skills` were
  merged into `brainwires-agent`.
- `brainwires-mesh`, `brainwires-channels`, and `brainwires-agent-network`
  were merged into `brainwires-network`.
- `brainwires-code-interpreters`, `brainwires-system`, and
  `brainwires-tool-system` were merged into `brainwires-tools`.

The merges happened without a written record of *what changed* between
the original split and the merge — so future contributors cannot tell
whether the merge fixed a real problem or simply consolidated for short-
term convenience.

This pattern is expensive: each round trip throws away the cleaner
import paths, narrower compile units, and clearer ownership boundaries
that the splits provided.

## Decision

A crate may not be merged into another crate without an Architecture
Decision Record (ADR) under `docs/adr/` documenting:

1. The original reason the crates were split.
2. What changed since the split (a new dep, a removed consumer, a
   simplified API surface, etc.) that justifies undoing it.
3. The expected consequences of the merge for downstream consumers
   (compile time, feature flags, public API surface).

The same ADR discipline applies in the reverse direction: a new crate
extracted from an existing one should write an ADR explaining the
extraction's motivation and the consumers that benefit.

The ADR template lives at `docs/adr/ADR-template.md`.

## Consequences

- **Positive**: future contributors have a written rationale for the
  current crate layout and can challenge or extend it from a position of
  understanding.
- **Positive**: the friction of writing an ADR raises the activation
  energy for purely cosmetic merges, which is the desired effect.
- **Negative**: small, obviously-correct refactors now carry a paperwork
  cost. Mitigation: keep ADRs short — one page is enough — and reuse
  the template.
- **Neutral**: existing god-crates predate this rule. Splits planned in
  the rest of this refactor (see `docs/adr/ADR-0002`, `ADR-0003`, etc.)
  will each carry their own ADR explaining why we are re-extracting now.

## Alternatives considered

- **No rule.** Status quo. Rejected because the `deprecated/` pattern
  shows that without a written brake, merges accumulate silently.
- **Require an RFC for every crate change.** Heavier process. Rejected
  because most crate changes are not architectural; the ADR rule is
  scoped narrowly to merges and extractions.

## References

- `deprecated/` directory at workspace root (the 17 prior splits).
- Workspace plan: `crates/` vs `extras/` boundary (see
  `docs/adr/ADR-0004-framework-extras-boundary.md`).
