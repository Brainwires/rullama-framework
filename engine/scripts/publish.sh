#!/usr/bin/env bash
# Publish rullama + rullama-finetune to crates.io.
#
# `rullama-finetune` cannot resolve its `rullama = { version = "X.Y" }`
# constraint until `rullama` is actually live on the registry, so this
# script publishes them in order and waits for the registry index to
# pick up the new rullama before publishing finetune.
#
# Usage:
#   ./scripts/publish.sh                                # real publish, both crates
#   ./scripts/publish.sh --dry-run                      # dry-run rullama (skips finetune)
#   ./scripts/publish.sh --bump 0.3.0                   # bump versions then publish
#   ./scripts/publish.sh --bump 0.3.0 --dry-run         # bump, then dry-run
#
# Flags:
#   --bump <version>   call `cargo bump <version>` first
#   --dry-run          pass `--dry-run` to `cargo publish` for rullama;
#                      skip finetune entirely (its dry-run requires
#                      rullama to be on crates.io already)
#
# Notes:
#   - The script does not git-commit the bump; review the diff and commit
#     yourself before re-running without --dry-run.
#   - Token: set $CARGO_REGISTRY_TOKEN or `cargo login` beforehand.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=
BUMP=

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --bump)
            BUMP="${2:-}"
            if [[ -z "$BUMP" ]]; then
                echo "publish.sh: --bump requires a version (e.g. --bump 0.3.0)" >&2
                exit 2
            fi
            shift 2
            ;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        *)
            echo "publish.sh: unknown arg \`$1\`" >&2
            exit 2
            ;;
    esac
done

if [[ -n "$BUMP" ]]; then
    echo "==> cargo bump $BUMP"
    cargo bump "$BUMP"
fi

# Read the rullama version that's about to be published.
VERSION=$(awk -F'"' '/^version[[:space:]]*=/ {print $2; exit}' crates/rullama/Cargo.toml)
if [[ -z "$VERSION" ]]; then
    echo "publish.sh: could not read version from crates/rullama/Cargo.toml" >&2
    exit 1
fi

if [[ -n "$DRY_RUN" ]]; then
    echo "==> cargo publish --dry-run -p rullama (v$VERSION)"
    cargo publish --dry-run -p rullama
    echo
    echo "==> skipping rullama-finetune dry-run"
    echo "    its \`rullama = { version = \"$VERSION\" }\` constraint can't"
    echo "    resolve from crates.io until rullama is actually published."
    echo "    Run \`cargo publish -p rullama-finetune --dry-run --no-verify\` if you"
    echo "    just want to package-check finetune without verifying the build."
    exit 0
fi

echo "==> cargo publish -p rullama (v$VERSION)"
cargo publish -p rullama

# Poll crates.io for the new rullama version to appear before publishing
# finetune. The registry index updates within seconds in normal operation;
# we give it up to 5 minutes.
echo "==> waiting for rullama $VERSION on crates.io"
URL="https://crates.io/api/v1/crates/rullama/$VERSION"
for i in $(seq 1 60); do
    if curl -fsS -o /dev/null "$URL"; then
        echo "    rullama $VERSION is live"
        break
    fi
    if [[ "$i" == "60" ]]; then
        echo "publish.sh: timed out waiting for $URL after 5 minutes" >&2
        echo "publish.sh: rullama-finetune was NOT published. Re-run without --bump:" >&2
        echo "    cargo publish -p rullama-finetune" >&2
        exit 1
    fi
    sleep 5
done

echo "==> cargo publish -p rullama-finetune (v$VERSION)"
cargo publish -p rullama-finetune

echo "==> done. Don't forget to push the v$VERSION tag if you haven't:"
echo "    git push origin v$VERSION"
