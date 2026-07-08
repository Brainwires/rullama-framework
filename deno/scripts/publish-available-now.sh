#!/usr/bin/env bash
# Publish the subset of @rullama/* packages whose slots exist AND whose deps
# are all satisfiable right now (17 of 27). The remaining 10 are blocked on the
# 7 un-created slots (skills, storage, stores, telemetry, tool-builtins,
# tool-runtime, inference) — publish those with publish-v0.12.0.sh once the
# JSR weekly new-package quota resets or a quota increase is granted.
#
# Usage:
#   ! JSR_TOKEN=jsrp_... PATH="$HOME/.deno/bin:$PATH" ./deno/scripts/publish-available-now.sh
#   ! PATH="$HOME/.deno/bin:$PATH" ./deno/scripts/publish-available-now.sh   # browser auth
#
# Continues past a failing package instead of aborting, and prints a summary.

set -uo pipefail
cd "$(dirname "$0")/.."

TOKEN_ARG=""
[ -n "${JSR_TOKEN:-}" ] && TOKEN_ARG="--token=$JSR_TOKEN"

# Dependency order: core -> core-only tier1 -> mcp-server -> network
PKGS=(
  core
  a2a agent call-policy eval finetune knowledge mcp-client mdap prompting
  provider provider-speech rag reasoning session
  mcp-server
  network
)

ok=(); fail=()
for pkg in "${PKGS[@]}"; do
  echo ""; echo "=== publishing @rullama/$pkg ==="
  if (cd "packages/$pkg" && deno publish --allow-dirty $TOKEN_ARG); then
    ok+=("$pkg")
  else
    fail+=("$pkg")
    echo "!!! FAILED @rullama/$pkg — continuing"
  fi
done

echo ""; echo "=== summary ==="
echo "published (${#ok[@]}): ${ok[*]}"
echo "failed    (${#fail[@]}): ${fail[*]:-none}"
echo ""
echo "still blocked on missing slots (need quota): skills storage stores telemetry tool-builtins tool-runtime inference"
echo "plus their dependents: permission memory seal"
