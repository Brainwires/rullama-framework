#!/usr/bin/env bash
set -euo pipefail

# Brainwires Framework — crates.io publish script
#
# Rate limits for NEW VERSIONS of existing crates (as of 2026):
#   - Burst: 30 new versions at once
#   - After burst: 1 crate per minute
#   - 16 workspace crates total = all within burst → ~5 minutes
#
# Strategy: publish all 16 within the burst window with short index-propagation
# delays between each. If we ever exceed 30, fall back to 1/min after burst.
# Crates are ordered by dependency DAG (leaves first, facade last).
# Deprecated stubs are published separately after all workspace crates.
# Non-published extras (brainwires-autonomy, brainwires-wasm) are excluded.
#
# Usage:
#   ./scripts/publish.sh                  # Dry run (preflight + cargo dry-run)
#   ./scripts/publish.sh --preflight-only  # Fast manifest checks only
#   ./scripts/publish.sh --live            # Preflight + actually publish

DRY_RUN=true
PREFLIGHT_ONLY=false
case "${1:-}" in
    --live)
        DRY_RUN=false
        echo "=== LIVE PUBLISH MODE ==="
        echo "This will publish all 21 workspace crates + any unpublished deprecated crates to crates.io."
        echo "Estimated time: ~5 minutes (burst 30, then 1/min)"
        echo "Press Ctrl+C within 5 seconds to abort..."
        sleep 5
        ;;
    --preflight-only)
        PREFLIGHT_ONLY=true
        ;;
esac

# 30 publishable workspace crates in strict dependency order (leaves → facade).
# Within each layer, crates have no mutual dependencies.
# Excluded (publish = false): brainwires-autonomy, brainwires-wasm, brainwires-sandbox-proxy
# Excluded (webrtc git-only dep): brainwires-channels (tombstone only)
# Retired (deprecated/, picked up by the auto-detect loop below):
#   brainwires-tools — split into brainwires-tool-runtime + brainwires-tool-builtins.
#   brainwires-permissions, brainwires-providers, brainwires-mcp,
#   brainwires-resilience, brainwires-agents — singularized.
#   brainwires-resilience also got a content-rename to brainwires-call-policy.
#   brainwires-finetune-local — moved to rullama-finetune in 0.11.
#   brainwires-training — moved to rullama-training in 0.11.
CRATES=(
    # Layer 0: Contracts
    brainwires-core

    # Layer 1a: Infrastructure — zero internal deps (except core)
    brainwires-telemetry
    brainwires-storage
    brainwires-eval               # evaluation harness — no brainwires-* deps at all

    # Layer 1b: Infrastructure — deps on 1a
    brainwires-provider           # optional dep: telemetry (LLM clients only)
    brainwires-provider-speech    # speech TTS / STT clients
    brainwires-hardware           # optional dep: providers + provider-speech
    brainwires-stores             # dep: storage — schema + CRUD for the opinionated minimum store set
    brainwires-memory             # dep: stores — TieredMemory orchestration + dream consolidation
    brainwires-sandbox            # container-backed sandbox executor
    brainwires-sandbox-proxy      # dep: sandbox — out-of-process proxy
    brainwires-call-policy        # retry / circuit / budget / cache / classify policies on outbound calls

    # Layer 2: Protocols (dep: core only)
    brainwires-mcp-client
    brainwires-mcp-server         # depends on mcp-client for shared types
    brainwires-a2a

    # Layer 3: Intelligence (storage-backed)
    brainwires-knowledge          # BKS/PKS, brain client, entity graph
    brainwires-rag                # codebase indexing + retrieval (with internal spectral + code_analysis)
    brainwires-prompting          # adaptive prompting (optional dep: knowledge)

    # Layer 4a: Tool runtime — split out of the old `brainwires-tools` in 0.11
    brainwires-tool-runtime       # ToolExecutor, ToolRegistry, validation, smart_router, +optional rag
    brainwires-permission

    # Layer 4b: Reasoning — depends on tool-runtime (ToolCategory in router.rs).
    # Prior releases had reasoning as a Layer 3 re-export facade with no tools
    # dep; the 0.10 restoration moved real scorer modules back in and this
    # order became necessary.
    brainwires-reasoning

    # Layer 4c: Tool builtins — concrete bash/git/web/code_exec/email/calendar
    # tools. Depends on tool-runtime + optional rag.
    brainwires-tool-builtins

    # Layer 4d: MDAP — extracted from brainwires-agent in 0.11. Zero internal
    # framework deps beyond core; safe to publish before agent.
    brainwires-mdap

    # Layer 4e: Skills — extracted from brainwires-agent in 0.11. Depends on
    # core + tool-runtime only.
    brainwires-skills

    # Layer 4f: SEAL — extracted from brainwires-agent in 0.11. Depends on
    # core + tool-runtime + storage (LanceDB pattern store). Optional deps
    # on knowledge / permission / mdap behind features.
    brainwires-seal

    # Layer 5: Agency
    brainwires-agent
    brainwires-network

    # Layer 6: Inference — extracted from brainwires-agent in 0.11. Depends on
    # agent for coordination types (CommunicationHub, FileLockManager, etc.).
    brainwires-inference

    # Layer 6: Fine-tuning
    brainwires-finetune           # cloud fine-tune APIs + dataset pipelines

    # Facade (must be last)
    brainwires
)

# ── Preflight checks ────────────────────────────────────────────────────────
# Fast manifest-only checks: missing READMEs, unversioned git deps,
# deps on publish=false crates. No cargo invocations, runs in <2s.
# Runs on every mode — catches blockers before slow cargo operations.
SCRIPT_DIR_PF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT_PF="$SCRIPT_DIR_PF/.."

echo "============================================"
echo "Preflight checks"
echo "============================================"

PREFLIGHT_ERRORS=0

# Collect publish=false crate names from direct crate/extra subdirs
UNPUBLISHABLE=()
for search_dir in "$WORKSPACE_ROOT_PF/crates" "$WORKSPACE_ROOT_PF/extras"; do
    [ -d "$search_dir" ] || continue
    for sub in "$search_dir"/*/; do
        toml_candidate="$sub/Cargo.toml"
        [ -f "$toml_candidate" ] || continue
        if grep -qE '^publish\s*=\s*false' "$toml_candidate"; then
            name=$(grep -m1 '^name' "$toml_candidate" | sed 's/.*"\(.*\)"/\1/' || true)
            [ -n "$name" ] && UNPUBLISHABLE+=("$name")
        fi
    done
done

for crate in "${CRATES[@]}"; do
    toml="$WORKSPACE_ROOT_PF/crates/$crate/Cargo.toml"
    [ -f "$toml" ] || continue

    # 1. Missing README file
    # Handle three cases: literal `readme = "X"`, `readme.workspace = true`
    # (inherits workspace.package.readme — usually "README.md"), or absent.
    readme_line=$(grep -m1 '^readme\b' "$toml" || true)
    readme_field=""
    if [[ "$readme_line" == *".workspace"* ]]; then
        # Workspace-inherited; default is README.md in the crate dir.
        readme_field="README.md"
    elif [ -n "$readme_line" ]; then
        readme_field=$(echo "$readme_line" | sed 's/.*"\(.*\)"/\1/')
    fi
    if [ -n "$readme_field" ] && [ ! -f "$WORKSPACE_ROOT_PF/crates/$crate/$readme_field" ]; then
        echo "  ERROR [$crate] readme = \"$readme_field\" does not exist"
        PREFLIGHT_ERRORS=$((PREFLIGHT_ERRORS + 1))
    fi

    # 2. Git deps missing a version (cargo requires version when publishing)
    while IFS= read -r line; do
        dep_name=$(echo "$line" | sed 's/^\s*\([a-zA-Z0-9_-]*\)\s*=.*/\1/')
        if ! echo "$line" | grep -qE 'version\s*='; then
            echo "  ERROR [$crate] git dep '$dep_name' has no version field (cargo publish requires one)"
            PREFLIGHT_ERRORS=$((PREFLIGHT_ERRORS + 1))
        fi
    done < <(grep -E '^\s*[a-zA-Z0-9_-]+\s*=.*git\s*=' "$toml" || true)

    # 3. Deps on publish=false crates (can't be resolved from crates.io)
    for unpub in "${UNPUBLISHABLE[@]}"; do
        [ -n "$unpub" ] || continue
        if grep -vE '^\s*#' "$toml" | grep -qE "^\s*${unpub}\s*=\s*\{|^\s*${unpub}\.workspace"; then
            echo "  ERROR [$crate] depends on '$unpub' which has publish = false"
            PREFLIGHT_ERRORS=$((PREFLIGHT_ERRORS + 1))
        fi
    done
done

if [ "$PREFLIGHT_ERRORS" -eq 0 ]; then
    echo "  All checks passed."
else
    echo ""
    echo "  $PREFLIGHT_ERRORS preflight error(s) found. Fix before running --live."
    exit 1
fi
echo ""
$PREFLIGHT_ONLY && exit 0
# ── End preflight ────────────────────────────────────────────────────────────

BURST_LIMIT=30
BURST_DELAY=15          # seconds between crates in the burst (index propagation)
POST_BURST_DELAY=70     # 1 min 10 sec between crates after burst exhausted

TOTAL=${#CRATES[@]}
PUBLISHED=0
FAILED=0

echo "============================================"
echo "Brainwires Framework — Publish to crates.io"
echo "Mode: $(if $DRY_RUN; then echo 'DRY RUN'; else echo 'LIVE'; fi)"
echo "Crates: $TOTAL"
echo "============================================"

for i in "${!CRATES[@]}"; do
    crate="${CRATES[$i]}"
    n=$((i + 1))

    echo ""
    echo "[$n/$TOTAL] Publishing $crate..."

    if $DRY_RUN; then
        # Dry run: only the leaf crates will fully verify (deps not on crates.io yet)
        if cargo publish --dry-run -p "$crate" 2>&1 | tail -3; then
            echo "OK: $crate (dry run)"
        else
            echo "SKIP: $crate (expected — deps not yet on crates.io)"
        fi
        PUBLISHED=$((PUBLISHED + 1))
        continue
    fi

    # Live publish
    publish_output=$(cargo publish -p "$crate" 2>&1) && publish_rc=0 || publish_rc=$?
    if [ "$publish_rc" -eq 0 ]; then
        echo "OK: $crate"
        PUBLISHED=$((PUBLISHED + 1))
    elif echo "$publish_output" | grep -q "already exists"; then
        echo "SKIP: $crate (already published)"
        PUBLISHED=$((PUBLISHED + 1))
        continue
    else
        echo "$publish_output"
        echo "FAILED: $crate"
        FAILED=$((FAILED + 1))
        echo ""
        echo "Publish failed. $PUBLISHED/$TOTAL published so far."
        echo "Fix the issue and re-run — already-published crates are skipped by crates.io."
        exit 1
    fi

    # Rate limiting: burst the first 30, then wait 1 min between each
    if [ "$n" -lt "$TOTAL" ]; then
        if [ "$n" -lt "$BURST_LIMIT" ]; then
            echo "  [burst $n/$BURST_LIMIT] Waiting ${BURST_DELAY}s..."
            sleep "$BURST_DELAY"
        elif [ "$n" -eq "$BURST_LIMIT" ]; then
            remaining=$((TOTAL - n))
            echo "  [burst exhausted] Switching to 1-minute intervals."
            echo "  $remaining crates remaining (~${remaining} minutes)."
            echo "  Waiting ${POST_BURST_DELAY}s..."
            sleep "$POST_BURST_DELAY"
        else
            remaining=$((TOTAL - n))
            echo "  Waiting 1 minute... ($remaining crates left, ~${remaining} min remaining)"
            sleep "$POST_BURST_DELAY"
        fi
    fi
done

echo ""
echo "============================================"
echo "Done! $PUBLISHED/$TOTAL crates published."
if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED crate(s) failed."
fi
echo "============================================"

# Auto-detect and publish deprecated crates that haven't been published yet.
# Scans deprecated/ for Cargo.toml files, checks crates.io for the version,
# and publishes if needed. These go AFTER workspace crates.
SCRIPT_DIR_DEP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPRECATED_DIR="$SCRIPT_DIR_DEP/../deprecated"

if [ -d "$DEPRECATED_DIR" ]; then
    for dep_toml in "$DEPRECATED_DIR"/*/Cargo.toml; do
        [ -f "$dep_toml" ] || continue
        dep_dir="$(dirname "$dep_toml")"
        dep_crate=$(grep -m1 '^name' "$dep_toml" | sed 's/.*"\(.*\)"/\1/')
        dep_version=$(grep -m1 '^version' "$dep_toml" | sed 's/.*"\(.*\)"/\1/')

        [ -z "$dep_crate" ] && continue
        [ -z "$dep_version" ] && continue

        # Check if this version is already on crates.io
        crate_info=$(curl -sf "https://crates.io/api/v1/crates/$dep_crate/$dep_version" 2>/dev/null || true)
        if echo "$crate_info" | grep -q '"version"'; then
            echo "[deprecated] SKIP: $dep_crate v$dep_version (already on crates.io)"
            continue
        fi

        echo ""
        echo "[deprecated] Publishing $dep_crate v$dep_version..."

        if $DRY_RUN; then
            if (cd "$dep_dir" && cargo publish --dry-run 2>&1 | tail -3); then
                echo "OK: $dep_crate (dry run)"
            else
                echo "SKIP: $dep_crate (dry run failed — may need workspace crates published first)"
            fi
            continue
        fi

        dep_output=$(cd "$dep_dir" && cargo publish 2>&1) && dep_rc=0 || dep_rc=$?
        if [ "$dep_rc" -eq 0 ]; then
            echo "OK: $dep_crate v$dep_version (deprecated crate published)"
        elif echo "$dep_output" | grep -q "already exists"; then
            echo "SKIP: $dep_crate (already published)"
        else
            echo "$dep_output"
            echo "WARNING: Failed to publish deprecated $dep_crate — non-fatal, continuing."
        fi
    done
fi

# Tag the release after successful publish
if ! $DRY_RUN && [ "$FAILED" -eq 0 ]; then
    # Determine the release version: use the highest version found across all
    # member crates (handles patch bumps where some crates have explicit versions
    # higher than the workspace base version).
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    WORKSPACE_ROOT="$SCRIPT_DIR/.."
    WORKSPACE_TOML="$WORKSPACE_ROOT/Cargo.toml"
    BASE_VERSION=$(grep -m1 '^version' "$WORKSPACE_TOML" | sed 's/.*"\(.*\)"/\1/')
    VERSION="$BASE_VERSION"
    for crate_dir in "$WORKSPACE_ROOT"/crates/*/; do
        crate_toml="$crate_dir/Cargo.toml"
        [ -f "$crate_toml" ] || continue
        v=$(grep -m1 '^version\s*=' "$crate_toml" 2>/dev/null | sed 's/.*"\(.*\)"/\1/' || true)
        if [ -n "$v" ] && [ "$v" != "$BASE_VERSION" ]; then
            # Simple semver comparison: pick the higher version
            if printf '%s\n%s\n' "$VERSION" "$v" | sort -V | tail -1 | grep -qx "$v"; then
                VERSION="$v"
            fi
        fi
    done

    TAG="v${VERSION}"
    echo ""
    if git rev-parse "$TAG" >/dev/null 2>&1; then
        echo "Tag $TAG already exists — skipping."
    else
        echo "Tagging release as $TAG..."
        git tag -a "$TAG" -m "Release $TAG"
        echo "Created tag $TAG"
        echo "Pushing tag to remote..."
        git push origin "$TAG"
    fi
fi
