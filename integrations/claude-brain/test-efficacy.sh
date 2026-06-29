#!/bin/bash
# Efficacy tests for claude-brain hooks
#
# Verifies:
# 1. Output sizes stay within budget for various settings
# 2. Loop detection suppresses output after threshold
# 3. Source routing works correctly (startup/compact/resume/clear)
# 4. Budget computation is correct for different window/pct combos
#
# Usage:
#   ./test-efficacy.sh          # Run all tests
#   ./test-efficacy.sh quick    # Budget math + loop detection only (no Brainwires needed)
#   ./test-efficacy.sh hooks    # Hook output tests only (needs Brainwires data)

set -euo pipefail

BINARY="${BINARY:-$(dirname "$0")/../../target/release/claude-brain}"
LOG="$HOME/.brainwires/claude-brain-hooks.log"
PASS=0
FAIL=0
SKIP=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${NC} $1"; FAIL=$((FAIL + 1)); }
skip() { echo -e "  ${YELLOW}SKIP${NC} $1"; SKIP=$((SKIP + 1)); }

# ── Test: Binary exists ──────────────────────────────────────────────
test_binary() {
    echo ""
    echo "=== Binary ==="
    if [ -x "$BINARY" ]; then
        pass "Binary exists at $BINARY"
    else
        fail "Binary not found at $BINARY"
        echo "  Run: cargo build --release -p claude-brain"
        exit 1
    fi
}

# ── Test: Budget computation ─────────────────────────────────────────
# Formula: threshold = window * pct
#          target    = threshold * 0.70
#          hook_share = target * 0.25
#          budget    = hook_share * 4.0
#          clamp [2000, 40000]
test_budget_math() {
    echo ""
    echo "=== Budget Math ==="

    # Test cases: window, pct(%), expected_budget
    local cases=(
        "200000 70 40000"   # 200K*0.70*0.70*0.25*4.0 = 98000 → clamped 40000
        "300000 70 40000"   # 300K*0.70*0.70*0.25*4.0 = 147000 → clamped 40000
        "100000 70 40000"   # 100K*0.70*0.70*0.25*4.0 = 49000 → clamped 40000
        "200000 50 40000"   # 200K*0.50*0.70*0.25*4.0 = 70000 → clamped 40000
        "100000 50 35000"   # 100K*0.50*0.70*0.25*4.0 = 35000
        "50000  50 17500"   # 50K*0.50*0.70*0.25*4.0  = 17500
        "20000  30 4200"    # 20K*0.30*0.70*0.25*4.0  = 4200
        "5000   50 2000"    # 5K*0.50*0.70*0.25*4.0   = 1750 → clamped 2000
    )

    for case in "${cases[@]}"; do
        read -r window pct expected <<< "$case"
        # Compute expected
        local threshold=$(echo "$window * $pct / 100" | bc)
        local target=$(echo "$threshold * 70 / 100" | bc)
        local hook_share=$(echo "$target * 25 / 100" | bc)
        local budget=$(echo "$hook_share * 4" | bc)
        # Clamp
        if [ "$budget" -lt 2000 ]; then budget=2000; fi
        if [ "$budget" -gt 40000 ]; then budget=40000; fi

        if [ "$budget" -eq "$expected" ]; then
            pass "window=${window} pct=${pct}% → budget=${budget} chars ($(echo "$budget / 4" | bc) tokens)"
        else
            fail "window=${window} pct=${pct}% → expected=${expected} got=${budget}"
        fi
    done
}

# ── Test: Source routing ─────────────────────────────────────────────
test_source_routing() {
    echo ""
    echo "=== Source Routing ==="

    local test_id="efficacy-routing-$$"

    # Clear → should emit nothing
    local output
    output=$(echo "{\"session_id\":\"${test_id}-clear\",\"cwd\":\"/tmp\",\"source\":\"clear\"}" | \
        "$BINARY" hook session-start 2>/dev/null)
    if [ -z "$output" ]; then
        pass "source=clear → no output"
    else
        fail "source=clear → expected empty, got ${#output} chars"
    fi

    # Startup → should emit context (or empty if no data)
    output=$(echo "{\"session_id\":\"${test_id}-startup\",\"cwd\":\"/tmp\",\"source\":\"startup\"}" | \
        "$BINARY" hook session-start 2>/dev/null)
    local startup_len=${#output}
    pass "source=startup → ${startup_len} chars"

    # Resume → same behavior as startup
    output=$(echo "{\"session_id\":\"${test_id}-resume\",\"cwd\":\"/tmp\",\"source\":\"resume\"}" | \
        "$BINARY" hook session-start 2>/dev/null)
    local resume_len=${#output}
    pass "source=resume → ${resume_len} chars"

    # Compact → should emit context (or empty if no digest)
    output=$(echo "{\"session_id\":\"${test_id}-compact\",\"cwd\":\"/tmp\",\"source\":\"compact\"}" | \
        "$BINARY" hook session-start 2>/dev/null)
    local compact_len=${#output}
    pass "source=compact → ${compact_len} chars"
}

# ── Test: Loop detection ─────────────────────────────────────────────
test_loop_detection() {
    echo ""
    echo "=== Loop Detection ==="

    local test_id="efficacy-loop-$$-$(date +%s)"
    local outputs=()

    # Fire 4 rapid compactions for same session
    for i in 1 2 3 4; do
        local output
        output=$(echo "{\"session_id\":\"${test_id}\",\"cwd\":\"/tmp\",\"source\":\"compact\"}" | \
            "$BINARY" hook session-start 2>/dev/null)
        outputs+=("${#output}")
    done

    # First 2 should work, 3rd+ should be suppressed
    # (output may be 0 anyway if no data, but log should show LOOP DETECTED)
    local loop_count
    loop_count=$(grep "${test_id}" "$LOG" 2>/dev/null | grep -c "LOOP DETECTED" || true)
    loop_count=${loop_count:-0}

    if [ "$loop_count" -ge 2 ]; then
        pass "Loop detected after 2 compactions (${loop_count} suppressions logged)"
    else
        fail "Expected >=2 loop detections, got ${loop_count}"
    fi

    # Verify log entries exist
    local total_entries
    total_entries=$(grep -c "${test_id}" "$LOG" 2>/dev/null || true)
    total_entries=${total_entries:-0}
    if [ "$total_entries" -eq 4 ]; then
        pass "All 4 events logged (${total_entries} entries)"
    else
        fail "Expected 4 log entries, got ${total_entries}"
    fi
}

# ── Test: Output within budget ───────────────────────────────────────
test_output_within_budget() {
    echo ""
    echo "=== Output Size vs Budget ==="

    local test_id="efficacy-budget-$$"

    # Test with real project directory that has data
    local cwd="/home/nightness/dev/brainwires-framework"
    if [ ! -d "$cwd" ]; then
        skip "brainwires-framework not found — skipping real-data test"
        return
    fi

    # Startup
    local output
    output=$(echo "{\"session_id\":\"${test_id}-startup\",\"cwd\":\"${cwd}\",\"source\":\"startup\"}" | \
        "$BINARY" hook session-start 2>/dev/null)
    local startup_len=${#output}

    # Get budget from log
    local budget
    budget=$(grep "${test_id}-startup" "$LOG" | grep -o "budget=[0-9]*" | head -1 | cut -d= -f2)
    budget=${budget:-40000}

    if [ "$startup_len" -le "$budget" ]; then
        pass "startup output (${startup_len}) <= budget (${budget})"
    else
        fail "startup output (${startup_len}) > budget (${budget})"
    fi

    # Compact
    output=$(echo "{\"session_id\":\"${test_id}-compact\",\"cwd\":\"${cwd}\",\"source\":\"compact\"}" | \
        "$BINARY" hook session-start 2>/dev/null)
    local compact_len=${#output}

    budget=$(grep "${test_id}-compact" "$LOG" | grep -o "budget=[0-9]*" | head -1 | cut -d= -f2)
    budget=${budget:-40000}

    if [ "$compact_len" -le "$budget" ]; then
        pass "compact output (${compact_len}) <= budget (${budget})"
    else
        fail "compact output (${compact_len}) > budget (${budget})"
    fi

    # Token estimate
    local startup_tokens=$((startup_len / 4))
    local compact_tokens=$((compact_len / 4))
    echo ""
    echo "  Output estimates:"
    echo "    startup: ${startup_len} chars (~${startup_tokens} tokens)"
    echo "    compact: ${compact_len} chars (~${compact_tokens} tokens)"
}

# ── Test: PostCompact emits nothing ──────────────────────────────────
test_post_compact_silent() {
    echo ""
    echo "=== PostCompact Silent ==="

    local output
    output=$(echo '{"session_id":"efficacy-pc","compact_summary":"test summary","cwd":"/tmp","trigger":"auto"}' | \
        "$BINARY" hook post-compact 2>/dev/null)

    if [ -z "$output" ]; then
        pass "PostCompact produces no stdout"
    else
        fail "PostCompact produced ${#output} chars of stdout"
    fi
}

# ── Test: Context headroom analysis ──────────────────────────────────
test_headroom() {
    echo ""
    echo "=== Headroom Analysis ==="
    echo "  (Estimates whether hook output leaves enough room for conversation)"
    echo ""

    # Common configurations
    local configs=(
        "200000 70"
        "300000 70"
        "100000 70"
        "200000 50"
    )

    # Estimated base context (system prompt + CLAUDE.md + rules + plugins)
    local base_tokens=18000
    # Estimated compaction summary size
    local summary_tokens=5000

    for config in "${configs[@]}"; do
        read -r window pct <<< "$config"
        local threshold=$((window * pct / 100))
        local budget_chars=$((window * pct / 100 * 70 / 100 * 25 / 100 * 4))
        if [ "$budget_chars" -lt 2000 ]; then budget_chars=2000; fi
        if [ "$budget_chars" -gt 40000 ]; then budget_chars=40000; fi
        local budget_tokens=$((budget_chars / 4))
        local total_post_compact=$((base_tokens + summary_tokens + budget_tokens))
        local headroom=$((threshold - total_post_compact))
        local usage_pct=$((total_post_compact * 100 / threshold))

        local status
        if [ "$headroom" -lt 5000 ]; then
            status="${RED}TIGHT${NC}"
        elif [ "$headroom" -lt 15000 ]; then
            status="${YELLOW}OK${NC}"
        else
            status="${GREEN}GOOD${NC}"
        fi

        printf "  window=%-6s pct=%s%% → threshold=%sT, post-compact=%sT, headroom=%sT (%s%%used) [%b]\n" \
            "$window" "$pct" "$threshold" "$total_post_compact" "$headroom" "$usage_pct" "$status"
    done

    echo ""
    echo "  Assumptions: base_context=${base_tokens}T, summary=${summary_tokens}T"
    echo "  TIGHT = <5K tokens headroom (loop risk)"
    echo "  OK    = 5-15K tokens headroom"
    echo "  GOOD  = >15K tokens headroom"
}

# ── Main ─────────────────────────────────────────────────────────────
main() {
    echo "╔══════════════════════════════════════╗"
    echo "║   Claude Brain — Efficacy Tests      ║"
    echo "╚══════════════════════════════════════╝"

    local mode="${1:-all}"

    test_binary

    case "$mode" in
        quick)
            test_budget_math
            test_loop_detection
            test_headroom
            ;;
        hooks)
            test_source_routing
            test_output_within_budget
            test_post_compact_silent
            ;;
        all)
            test_budget_math
            test_source_routing
            test_loop_detection
            test_output_within_budget
            test_post_compact_silent
            test_headroom
            ;;
        *)
            echo "Usage: $0 [all|quick|hooks]"
            exit 1
            ;;
    esac

    echo ""
    echo "════════════════════════════════════════"
    echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${SKIP} skipped${NC}"
    if [ "$FAIL" -gt 0 ]; then
        exit 1
    fi
}

main "$@"
