<!-- fallow:agent-install v1 authored sha256=249c350d89476d9acf9b857e89ac45990294e4f8b42d2dba3abc0f2a110fe939 -->
# AGENTS.md

This file gives coding agents project-specific context. Keep it short and update it when workflows change.

## Project Overview

- Primary app or package: the `rullama` Rust workspace (`crates/`) and its Deno/TypeScript
  port under `deno/` — 27 JSR packages published as `@rullama/*` (`deno/packages/*`).
- Main entry points: each `deno/packages/<name>/mod.ts` (plus the subpath `exports` in that
  package's `deno.json`); Rust crates under `crates/<name>/src/lib.rs`.
- Important directories: `deno/packages` (the published packages), `deno/tests` (cross-package
  integration tests), `deno/examples` (`deno run` samples), `deno/docs`, `deno/fixtures`
  (Rust↔Deno wire goldens), `crates/` (Rust).

## Architecture Notes

- Module boundaries: `@rullama/core` is the zero-dependency type layer; every other package
  depends on it and never on a sibling except as declared in `deno/README.md`'s dependency
  diagram. Sibling imports use the bare `@rullama/<name>` specifier and resolve to the local
  workspace member.
- Generated or vendored code: none in `deno/`; `deno/fixtures/*.json` are goldens read as text.
- Sensitive areas: `deno/packages/tool-builtins` (bash/file/git/web execution),
  `deno/packages/permission` (policy enforcement), `deno/packages/provider` (API keys,
  request signing), `deno/packages/mcp-server` and `a2a` (inbound protocol handling).

## Commands (Deno, run from `deno/`)

- Install: nothing to install — `deno` resolves the import map; `fallow` must be on PATH
  (`npm i -g fallow` or `cargo install fallow-cli`).
- Build: no build step (JSR publishes TypeScript sources).
- Test: `deno task test` (packages/ + tests/); one package: `deno test -A packages/<name>/`.
- Typecheck or lint: `deno task check` = fmt --check + lint + type-check (packages, tests,
  examples) + `deno task doc-lint` (JSDoc on every export of the 74 published entrypoints)
  + tests — what CI runs; `deno task check:fix` auto-fixes formatting and fixable lint.
- Publishing: push a `deno-v<version>` tag (all 27 packages must carry that version) — the
  `publish-deno.yml` workflow publishes with OIDC provenance; `deno/scripts/publish.sh` is the
  manual fallback. `deno task jsr:scores` refreshes `deno/docs/jsr-scores.md`.
- Coverage for fallow's CRAP score: `deno task coverage:fallow` (writes
  `deno/coverage/coverage-final.json`, which the pre-commit hook passes to `fallow audit`).

## Fallow

- Use `fallow audit --format json --quiet` before committing AI-generated changes.
- Use `fallow dead-code --format json --quiet`, `fallow dupes --format json --quiet`, and `fallow health --format json --quiet` for targeted checks.
- Use `fallow list --entry-points --format json --quiet` and `fallow list --boundaries --format json --quiet` to inspect project shape.

<!-- generated:task-matrix:start -->
| When the agent is about to... | Run |
|---|---|
| delete an "unused" export or file | `fallow dead-code --trace <file>:<export>` |
| prove a TypeScript symbol's exact consumers before refactoring | `fallow dead-code --type-aware --symbol-impact <file>:<export-or-class.method>` |
| delete an "unused" dependency | `fallow dead-code --trace-dependency <name>` |
| commit or open a PR | `fallow audit --base <ref>` |
| prioritize refactoring | `fallow health --hotspots --targets` |
| ask who owns code | `fallow health --ownership` |
| check untested-but-reachable code | `fallow health --coverage-gaps` |
| consolidate duplication | `fallow dupes --trace dup:<fingerprint>` |
| find feature flags | `fallow flags` |
| check which architecture rules apply to a file before changing it | `fallow guard <files>` |
| surface security candidates | `fallow security` |
| understand a finding | `fallow explain <issue-type>` |
| scope a monorepo | `--workspace <glob> / --changed-workspaces <ref>` (global flags, prefix any command) |
<!-- generated:task-matrix:end -->

## Agent Rules

- Do not edit: `deno/fixtures/**` (wire goldens shared with the Rust crates), the
  `<!-- fallow:... -->` managed blocks in this file, `.claude/hooks/fallow-gate.sh`.
- Always ask before: publishing to JSR, changing a package's public exports, deleting a
  package, or adding an `npm:` dependency to a package (only `storage` carries any today).
- Preferred style: idiomatic TypeScript over Rust transliteration — no `static new()`, no
  newtype wrapper classes, no free-function accessors that should be methods; JSDoc on every
  export (JSR "has docs" gate); tests beside the source as `<file>_test.ts`.

---

<!-- fallow:setup-hooks:start -->
## Fallow local gate

Before any `git commit` or `git push`, run `fallow audit --format json --quiet --explain --gate-marker agent`. If the verdict is `fail`, fix the reported findings before retrying. Treat JSON runtime errors like `{ "error": true, ... }` as non-blocking.

Audit defaults to `gate=new-only`: only findings introduced by the current changeset affect the verdict. Inherited findings on touched files are reported under `attribution` and annotated with `introduced: false`, but do not block the commit. Set `[audit] gate = "all"` in `fallow.toml` to gate every finding in changed files.

For non-skill agents, treat the task map below as the local onboarding source: run the listed fallow command before destructive edits, before commits, and before pull request handoff.

## Fallow task map

| When the agent is about to... | Run |
|---|---|
| delete an "unused" export or file | `fallow dead-code --trace <file>:<export>` |
| prove a TypeScript symbol's exact consumers before refactoring | `fallow dead-code --type-aware --symbol-impact <file>:<export-or-class.method>` |
| delete an "unused" dependency | `fallow dead-code --trace-dependency <name>` |
| commit or open a PR | `fallow audit --base <ref>` |
| prioritize refactoring | `fallow health --hotspots --targets` |
| ask who owns code | `fallow health --ownership` |
| check untested-but-reachable code | `fallow health --coverage-gaps` |
| consolidate duplication | `fallow dupes --trace dup:<fingerprint>` |
| find feature flags | `fallow flags` |
| check which architecture rules apply to a file before changing it | `fallow guard <files>` |
| surface security candidates | `fallow security` |
| understand a finding | `fallow explain <issue-type>` |
| scope a monorepo | `--workspace <glob> / --changed-workspaces <ref>` (global flags, prefix any command) |
<!-- fallow:setup-hooks:end -->
