#!/usr/bin/env bash
# Publish all 27 @rullama/* packages to JSR from a workstation.
#
# The normal path is the GitHub workflow (.github/workflows/publish-deno.yml): push a
# `deno-v<version>` tag and Actions publishes with OIDC provenance. This script is the
# manual fallback for the same thing.
#
#   ./deno/scripts/publish.sh --dry-run          # rehearse (allows a dirty tree)
#   ./deno/scripts/publish.sh                    # interactive browser auth
#   JSR_TOKEN=jsrp_... ./deno/scripts/publish.sh # token auth
#
# `deno publish` at the workspace root publishes every workspace member in dependency
# order (core → permission → tool-runtime → tool-builtins → inference …), so no
# hand-maintained tier list is needed. All packages must carry the same version.

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=""
TOKEN_ARG=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN="--dry-run --allow-dirty" ;;
    --token=*) TOKEN_ARG="$arg" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [ -z "$TOKEN_ARG" ] && [ -n "${JSR_TOKEN:-}" ]; then
  TOKEN_ARG="--token=$JSR_TOKEN"
fi

version=$(deno eval 'console.log(JSON.parse(Deno.readTextFileSync("packages/core/deno.json")).version)')
for f in packages/*/deno.json; do
  v=$(deno eval "console.log(JSON.parse(Deno.readTextFileSync('$f')).version)")
  if [ "$v" != "$version" ]; then
    echo "error: $f is $v but packages/core is $version — bump all packages in lockstep" >&2
    exit 1
  fi
done

echo "=== publishing @rullama/* $version ${DRY_RUN:+(dry run)} ==="
deno task check
# shellcheck disable=SC2086
deno publish $DRY_RUN $TOKEN_ARG
echo "=== done. Tag it: git tag -a deno-v$version -m 'Deno port v$version' && git push origin deno-v$version ==="
