# ADR-0004 — Framework / extras boundary

- **Status:** Accepted
- **Date:** 2026-05-02
- **Authors:** Brainwires

## Context

The workspace contains two top-level directories of Rust crates:

- `crates/` — the **framework**: cohesive, independently-publishable
  libraries that form the building blocks of `brainwires`.
- `extras/` — applications and libraries that **consume** the framework:
  binaries (CLIs, MCP servers, demos), integration helpers, and adapter
  crates for specific external systems.

The boundary between the two has been implicit. Two cases historically
caused friction:

1. A library that lives in `extras/` but is depended on by something in
   `crates/`. This creates a hidden coupling: the framework can no
   longer be reasoned about without also reasoning about an "extra".
2. An `extras/` crate that depends on another `extras/` crate. This
   creates an undeclared hierarchy among siblings — adapters depending
   on adapters — which makes both harder to remove or replace.

Audit run at planning time (May 2026) confirmed that the codebase
already respects the rule everywhere:

- Zero `crates/ → extras/` dependency arrows (the `brainwires` umbrella
  crate's `Cargo.toml` even has comments explaining why specific
  features were removed to preserve the rule).
- Zero `extras/ → extras/` dependency arrows.

But the rule was nowhere written down, so future contributors had no
reference for what to do when a tempting cross-arrow appeared.

## Decision

Allowed dependency arrows in this workspace:

- `crates/X → crates/Y`
- `extras/X → crates/Y`

Forbidden:

- `crates/X → extras/Y` — the framework cannot depend on its consumers.
- `extras/X → extras/Y` — extras are siblings of equal standing, not a
  hierarchy. If an `extras/` library needs to be reused by another
  `extras/` library, that library belongs in `crates/`.

The rule is enforced by `cargo xtask lint-deps`, which walks every
`Cargo.toml` in the workspace and rejects any forbidden arrow. CI runs
the lint as part of the standard checks.

The rule is documented for human readers in `README.md` under
"Workspace layout".

## Consequences

- **Positive**: the framework stays portable — a downstream user can
  consume `crates/brainwires-*` without reasoning about anything in
  `extras/`.
- **Positive**: an `extras/` adapter can be removed, renamed, or
  replaced without churning any other entry in `extras/` or any crate
  in `crates/`.
- **Positive**: the lint catches violations at PR time instead of
  letting them accumulate silently (which is how the `deprecated/`
  god-crates happened — see ADR-0001).
- **Negative**: a library that genuinely belongs in `extras/` but
  starts being reused by another `extras/` library forces a move into
  `crates/`. We accept this — it's exactly the signal we want.
- **Neutral**: workspace-only crates with `publish = false` (e.g.,
  `brainwires-autonomy`) are perfectly fine in `extras/` as long as no
  `crates/` member depends on them.

## Alternatives considered

- **Allow `crates/ → extras/` for "trusted" extras.** Rejected: there
  is no useful definition of "trusted" that doesn't degrade into
  permission-by-precedent. If something is trusted enough for the
  framework to depend on, it belongs *in* the framework.
- **Allow `extras/ → extras/` if explicitly declared.** Rejected:
  declared dependencies between siblings are still siblings depending
  on siblings. The forced refactor (move the shared library into
  `crates/`) is the correct outcome.
- **Skip the xtask lint, rely on convention.** Rejected: convention
  alone is exactly what produced the `deprecated/` god-crates. Mechanical
  enforcement at CI time is cheap and prevents drift.

## References

- `README.md` — "Workspace layout" section names the rule for
  human readers.
- `xtask/src/lint_deps.rs` — implementation of the enforcement.
- ADR-0001 — crate split discipline (the same theme: write the rule
  down so it survives future churn).
