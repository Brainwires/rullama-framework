#!/usr/bin/env bash
# Publish all 27 @rullama/* packages to JSR from a workstation.
#
# The normal release path is `deno task publish:changes`: it tags changed packages
# `<package>-v<version>` and the publish-deno.yml workflow publishes each one with
# OIDC provenance. This script is the manual fallback for the same thing.
#
#   ./deno/scripts/publish.sh --dry-run                 # rehearse everything (dirty tree ok)
#   ./deno/scripts/publish.sh --package tool-runtime    # one package (like a <pkg>-v* tag)
#   ./deno/scripts/publish.sh                           # every package, interactive auth
#   JSR_TOKEN=jsrp_... ./deno/scripts/publish.sh        # token auth
#
# With --package it publishes that member only; without it, every member (all must
# carry the same version).

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=""
TOKEN_ARG=""
PACKAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN="--dry-run --allow-dirty" ;;
    --token=*) TOKEN_ARG="$1" ;;
    --package) shift; PACKAGE="$1" ;;
    --package=*) PACKAGE="${1#--package=}" ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
if [ -z "$TOKEN_ARG" ] && [ -n "${JSR_TOKEN:-}" ]; then
  TOKEN_ARG="--token=$JSR_TOKEN"
fi

if [ -n "$PACKAGE" ]; then
  cfg="packages/$PACKAGE/deno.json"
  [ -f "$cfg" ] || { echo "error: no such package: $PACKAGE" >&2; exit 1; }
  version=$(deno eval "console.log(JSON.parse(Deno.readTextFileSync('$cfg')).version)")
  echo "=== publishing @rullama/$PACKAGE $version ${DRY_RUN:+(dry run)} ==="
  deno task check
  # shellcheck disable=SC2086
  deno publish --config "$cfg" $DRY_RUN $TOKEN_ARG
  echo "=== done. Tag it: git tag -a $PACKAGE-v$version -m '@rullama/$PACKAGE $version' && git push origin $PACKAGE-v$version ==="
  exit 0
fi

version=$(deno eval 'console.log(JSON.parse(Deno.readTextFileSync("packages/core/deno.json")).version)')
for f in packages/*/deno.json; do
  v=$(deno eval "console.log(JSON.parse(Deno.readTextFileSync('$f')).version)")
  if [ "$v" != "$version" ]; then
    echo "error: $f is $v but packages/core is $version — bump all packages in lockstep, or use --package" >&2
    exit 1
  fi
done

echo "=== publishing every @rullama/* package at $version ${DRY_RUN:+(dry run)} ==="
deno task check
# shellcheck disable=SC2086
deno publish $DRY_RUN $TOKEN_ARG
echo "=== done ==="
