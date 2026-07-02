#!/usr/bin/env bash
# Publish all 27 @rullama/* packages to JSR in dependency order.
#
# Usage:
#   ! ./deno/scripts/publish-v0.12.0.sh           # interactive (browser auth)
#   ! ./deno/scripts/publish-v0.12.0.sh --dry-run # dry run all packages
#   ! JSR_TOKEN=jsrp_... ./deno/scripts/publish-v0.12.0.sh
#   ! ./deno/scripts/publish-v0.12.0.sh --token jsrp_...

set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=""
TOKEN_ARG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN="--dry-run --allow-dirty"; shift ;;
    --token=*) TOKEN_ARG="--token=${1#--token=}"; shift ;;
    --token)   shift; TOKEN_ARG="--token=$1"; shift ;;
    *) shift ;;
  esac
done

if [ -z "$TOKEN_ARG" ] && [ -n "${JSR_TOKEN:-}" ]; then
  TOKEN_ARG="--token=$JSR_TOKEN"
fi

# Real dep graph (no transitional shims):
#
#   Tier 0: core (no @rullama deps)
#   Tier 1: depend only on core         — telemetry, storage, mcp-client,
#                                          call-policy, provider-speech,
#                                          session, agent, eval, mdap,
#                                          prompting, rag, reasoning,
#                                          finetune, a2a, tool-runtime
#   Tier 2: depend on tier 0 + tier 1   — permission (→telemetry),
#                                          stores (→storage),
#                                          mcp-server (→mcp-client),
#                                          provider (→core only, listed here
#                                          for symmetry — it has zero
#                                          @rullama deps after shim removal),
#                                          tool-builtins (→tool-runtime),
#                                          knowledge (BrainClient core only)
#   Tier 3: depend on tier 2            — memory (→stores), skills
#                                          (→tool-builtins), network
#                                          (→mcp-server),
#                                          seal (→permission, knowledge,
#                                          tool-runtime), inference
#                                          (→agent, tool-runtime)
#
# (provider, knowledge: zero @rullama deps now — placed in tier 1.)

TIER_0=("core")
TIER_1=("a2a" "agent" "call-policy" "eval" "finetune" "knowledge" "mcp-client" \
        "mdap" "prompting" "provider" "provider-speech" "rag" "reasoning" \
        "session" "storage" "telemetry" "tool-runtime")
TIER_2=("mcp-server" "permission" "stores" "tool-builtins")
TIER_3=("inference" "memory" "network" "seal" "skills")

publish_pkg() {
  local pkg="$1"
  echo ""
  echo "=== publishing @rullama/$pkg ==="
  (cd "packages/$pkg" && deno publish $DRY_RUN $TOKEN_ARG)
}

publish_tier() {
  local name="$1"; shift
  echo ""
  echo "### Tier: $name ###"
  for pkg in "$@"; do
    publish_pkg "$pkg"
  done
}

publish_tier "0  (zero @rullama deps)"          "${TIER_0[@]}"
publish_tier "1  (depends only on core)"           "${TIER_1[@]}"
publish_tier "2  (depends on tier 1)"              "${TIER_2[@]}"
publish_tier "3  (depends on tier 2)"              "${TIER_3[@]}"

echo ""
echo "=== all 27 packages published. tag deno-v0.12.0 next. ==="
echo "  git tag -a deno-v0.12.0 -m 'Deno port v0.12.0' && git push origin deno-v0.12.0"
