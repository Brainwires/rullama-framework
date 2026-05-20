# Publishing Checklist

Reusable checklist for releasing new versions of the Brainwires Framework to crates.io.

## 0. New Crate Checklist

Run this whenever a new crate is added to the workspace before the release that includes it.

- [ ] **`README.md` exists** in the crate directory — crates.io displays nothing without it
- [ ] **`publish = false` is NOT set** (or is intentionally set for internal-only crates that must not be published)
- [ ] **`readme = "README.md"`** is present in `[package]` (not inherited from workspace)
- [ ] **`documentation`, `keywords`, `categories`** are set in `[package]` (not inherited from workspace)
- [ ] **No git-only dependencies** — all deps must have a `version = "..."` for crates.io. Git deps without a version block publishing. If a git-only dep is required (e.g. a fork), either:
  - Publish the fork to crates.io first, or
  - Put the git-dep code in a separate `extras/` crate marked `publish = false`
- [ ] **Added to `scripts/publish.sh` CRATES array** in the correct dependency layer
- [ ] **Added to the publish order table** in this file (Section 3)

Quick audit command (run from workspace root):
```bash
for dir in crates/*/; do
  crate=$(basename "$dir")
  readme="$dir/README.md"
  publish=$(grep -m1 'publish' "$dir/Cargo.toml" 2>/dev/null || echo "")
  in_script=$(grep -c "$crate" scripts/publish.sh 2>/dev/null || echo 0)
  echo "$crate | readme=$(test -f $readme && echo YES || echo MISSING) | $publish | in_script=$in_script"
done
```

---

## 1. Pre-release Checks

- [ ] All changes committed, clean working tree (`git status`)
- [ ] `CHANGELOG.md` has release notes under `## [Unreleased]` (version stamp is automatic — see step 2)
- [ ] **New crates audited** — run the Section 0 checklist for every crate added since the last release
- [ ] **README.md files updated** — verify all crate READMEs reflect the release changes:
  - Root `README.md` — crate descriptions, feature tables, architecture diagrams
  - Each changed crate's `README.md` — API tables, code examples, feature flags
  - `crates/README.md` — dependency tree and crate descriptions
  - `crates/brainwires/README.md` (facade) — feature table, crate count, prelude types
  - `extras/` server READMEs — cross-references to library crates
- [ ] `cargo xtask` passes (fmt, check, clippy, test, doc)
- [ ] **No unfinished code** — run `cargo xtask check-stubs` to scan for runtime-panic stubs and unfinished markers:
  ```bash
  cargo xtask check-stubs            # Should use "--strict"! Only fails on todo!(), unimplemented!(); warn on FIXME, HACK, etc.
  cargo xtask check-stubs --strict   # also fail on comment markers (FIXME, HACK, XXX, STUB, STOPSHIP)
  cargo xtask check-stubs --verbose  # list every file scanned
  ```
  Hard blockers (`todo!()`, `unimplemented!()`) in trait impls or public API must be replaced with `Err(...)` or the module removed before release. Comment markers (FIXME, HACK, etc.) are warnings by default — use `--strict` to enforce zero markers.
- [ ] `cargo build --workspace` succeeds
- [ ] `cargo test --workspace` passes

## 2. Version Bump

The bump tool has two modes, selected automatically based on whether the version change is a **patch** (same major.minor) or **minor/major** bump.

### Minor / Major bump (all crates)

```bash
cargo xtask bump-version 0.5.0
```

Bumps **every** crate to the new version. This:
- Resets any per-crate version overrides back to `version.workspace = true` (cleans up after previous patch releases)
- Updates `[workspace.package].version` in root `Cargo.toml`
- Updates all `version = "..."` on internal crate deps in `[workspace.dependencies]`
- Updates member `Cargo.toml` files with direct path deps (e.g. brainwires-wasm)
- Updates hardcoded version strings in `*.rs` source files
- Updates `*.md` files (skips CHANGELOG)
- Stamps `CHANGELOG.md`: `## [Unreleased]` → `## [X.Y.Z] - YYYY-MM-DD`

### Patch bump (selective)

```bash
# Auto-detect changed crates from git (uses git diff against last version tag)
cargo xtask bump-version 0.4.1

# Or specify crates manually
cargo xtask bump-version 0.4.1 --crates brainwires-core,brainwires-storage
```

Only bumps **affected crates** + their transitive dependents. This:
- Detects which crates changed since the last version tag (`v0.4.0`), or uses `--crates` if specified
- **Cascades** to all crates that depend (directly or transitively) on any affected crate
- Prints the full list before making changes
- Sets affected crates to explicit `version = "0.4.1"` (overriding `version.workspace = true`)
- Updates `[workspace.dependencies]` version fields for affected crates only
- Updates `.rs` and `.md` files only within affected crate directories
- Stamps `CHANGELOG.md`
- Leaves the workspace root version unchanged (e.g., stays at `0.4.0`)

**Cascade example:**

```
$ cargo xtask bump-version 0.4.1 --crates brainwires-core

Patch bump to 0.4.1:
  Direct:  brainwires-core
  Cascade: brainwires-agent, brainwires-autonomy, brainwires-mcp-client, ...
  Total:   14 crate(s)
```

On the next minor release (`0.5.0`), all crates reset to `version.workspace = true` automatically.

### After bumping

```bash
git diff                    # Review changes
cargo check --workspace     # Verify it compiles
git add -A && git commit -m "chore: bump version to X.Y.Z"
```

**Note:** The bumper handles version numbers automatically, but you must still manually verify README *content* (descriptions, API tables, architecture diagrams) matches the release changes.

## 3. Publish to crates.io

### Dry run (default)

```bash
./scripts/publish.sh
```

Only leaf crates fully verify in dry-run mode (later layers can't resolve deps not yet on crates.io). This is expected.

### Live publish

```bash
./scripts/publish.sh --live
```

The script handles:
- **Dependency ordering** — 8 layers, 32 crates, leaves first, facade last
- **Rate limiting** — burst 30 at once, then 1/min (32 crates fits in burst)
- **Idempotency** — already-published versions are skipped automatically
- **Tagging** — creates and pushes `vX.Y.Z` git tag on success

### Publish order

Source of truth: `scripts/publish.sh`. Reproduced here for reference.

| Layer | Crates |
|-------|--------|
| 0 — Contracts | `brainwires-core` |
| 1a — Infrastructure (deps: core) | `brainwires-telemetry`, `brainwires-storage`, `brainwires-eval` |
| 1b — Infrastructure (deps on 1a) | `brainwires-provider`, `brainwires-provider-speech`, `brainwires-hardware`, `brainwires-stores`, `brainwires-memory`, `brainwires-sandbox`, `brainwires-sandbox-proxy`, `brainwires-call-policy` |
| 2 — Protocols (deps: core only) | `brainwires-mcp-client`, `brainwires-mcp-server`, `brainwires-a2a` |
| 3 — Intelligence (storage-backed) | `brainwires-knowledge`, `brainwires-rag`, `brainwires-prompting` |
| 4a — Tool runtime | `brainwires-tool-runtime`, `brainwires-permission` |
| 4b — Reasoning (deps: tool-runtime) | `brainwires-reasoning` |
| 4c — Tool builtins (deps: tool-runtime + optional rag) | `brainwires-tool-builtins` |
| 5 — Agency | `brainwires-agent`, `brainwires-network`, `brainwires-skills`, `brainwires-mdap`, `brainwires-seal`, `brainwires-inference` |
| 6 — Fine-tune | `brainwires-finetune` |
| 7 — Facade | `brainwires` |

**Excluded from publish** (`publish = false` in their `Cargo.toml`): `brainwires-sandbox-proxy`, plus all `extras/*` crates (`brainwires-autonomy`, `brainwires-wasm`, etc.). The 0.11 cycle removed `brainwires-llama` (orphan rullama snapshot, never on crates.io) and `brainwires-finetune-local` / `brainwires-training` (moved to the sibling `rullama` workspace).

## 4. Post-publish

- [ ] Verify on crates.io: `cargo search brainwires`
- [ ] Tag pushed automatically by publish script (`vX.Y.Z`)
- [ ] Update CLI workspace version refs if needed (`/home/nightness/dev/brainwires-cli/Cargo.toml`)

## 5. Troubleshooting

**Publish fails mid-way?** Re-run `./scripts/publish.sh --live` — already-published crates are skipped.

**Rate limited?** Wait a few minutes and re-run. The script handles burst vs sustained rate limits.

**Version conflict?** A crate version already exists on crates.io. Bump to a new patch version and re-publish.
