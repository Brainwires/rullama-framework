#!/usr/bin/env bash
# Pre-create all 27 @rullama/* packages on JSR (empty package slots).
#
# Prereqs:
#   1. The `rullama` scope must already exist (create at https://jsr.io/new).
#   2. A JSR personal access token with package-manage permission
#      (create at https://jsr.io/account/tokens).
#
# Usage:
#   ! JSR_TOKEN=jsrp_... ./deno/scripts/precreate-jsr-packages.sh
#   ! ./deno/scripts/precreate-jsr-packages.sh --token jsrp_...
#
# Idempotent: a package that already exists reports "exists" and is skipped.

set -euo pipefail

SCOPE="rullama"
TOKEN="${JSR_TOKEN:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --token=*) TOKEN="${1#--token=}"; shift ;;
    --token)   shift; TOKEN="$1"; shift ;;
    *) shift ;;
  esac
done

if [ -z "$TOKEN" ]; then
  echo "error: no token. Set JSR_TOKEN or pass --token jsrp_..." >&2
  exit 1
fi

PKGS=(
  a2a agent call-policy core eval finetune knowledge mcp-client mcp-server
  mdap memory network permission prompting provider provider-speech rag
  reasoning seal session skills storage stores telemetry tool-builtins
  tool-runtime inference
)

echo "Pre-creating ${#PKGS[@]} packages under @${SCOPE} ..."

for pkg in "${PKGS[@]}"; do
  code=$(curl -s -o /tmp/jsr_precreate_body -w "%{http_code}" \
    -X POST "https://api.jsr.io/scopes/${SCOPE}/packages" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"package\":\"${pkg}\"}")

  case "$code" in
    200|201) echo "  created  @${SCOPE}/${pkg}" ;;
    409)     echo "  exists   @${SCOPE}/${pkg}" ;;
    *)       echo "  FAILED   @${SCOPE}/${pkg}  (http ${code}): $(cat /tmp/jsr_precreate_body)" ;;
  esac
done

echo "Done. Verify at https://jsr.io/@${SCOPE}"
