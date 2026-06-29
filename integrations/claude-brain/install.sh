#!/usr/bin/env bash
# install.sh — Install or uninstall claude-brain for Claude Code.
#
# Usage:
#   ./install.sh [install|uninstall|status] [--global] [--project-dir /path/to/project]
#
# Modes:
#   --global             Install to ~/.claude/ (all sessions, all projects)
#   --project-dir PATH   Install to a specific project (default: framework root)
#   (neither)            Install to framework root project
#
# Actions:
#   install  (default) — build binary, wire hooks + MCP + rules
#   uninstall          — remove hooks, MCP entry, and rules (keeps binary + data)
#   status             — show what's installed

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY_PATH="${CARGO_HOME:-$HOME/.cargo}/bin/claude-brain"
INTEGRATION_DIR="$SCRIPT_DIR/integration"
BRAINWIRES_DIR="$HOME/.brainwires"
CONFIG_FILE="$BRAINWIRES_DIR/claude-brain.toml"
LOG_FILE="$BRAINWIRES_DIR/claude-brain-hooks.log"

GLOBAL=false
PROJECT_DIR=""

ACTION="${1:-install}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --global)
            GLOBAL=true
            shift
            ;;
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Resolve target paths based on mode
if $GLOBAL; then
    CLAUDE_DIR="$HOME/.claude"
    SETTINGS_FILE="$CLAUDE_DIR/settings.json"
    MCP_FILE="$CLAUDE_DIR/mcp.json"
    RULES_DIR="$CLAUDE_DIR/rules"
    RULES_FILE="$RULES_DIR/claude-brain.md"
    SKILLS_DIR="$CLAUDE_DIR/skills/claude-brain"
    SKILL_FILE="$SKILLS_DIR/SKILL.md"
    INSTALL_LABEL="global (~/.claude/)"
else
    PROJECT_DIR="${PROJECT_DIR:-$FRAMEWORK_DIR}"
    CLAUDE_DIR="$PROJECT_DIR/.claude"
    SETTINGS_FILE="$CLAUDE_DIR/settings.local.json"
    MCP_FILE="$PROJECT_DIR/.mcp.json"
    RULES_DIR="$CLAUDE_DIR/rules"
    RULES_FILE="$RULES_DIR/claude-brain.md"
    SKILLS_DIR="$CLAUDE_DIR/skills/claude-brain"
    SKILL_FILE="$SKILLS_DIR/SKILL.md"
    INSTALL_LABEL="project ($PROJECT_DIR)"
fi

# ── Helpers ────────────────────────────────────────────────────────────

green()  { printf "\033[32m%s\033[0m\n" "$1"; }
yellow() { printf "\033[33m%s\033[0m\n" "$1"; }
red()    { printf "\033[31m%s\033[0m\n" "$1"; }

need_python() {
    command -v python3 >/dev/null 2>&1 || { red "python3 required for JSON merging"; exit 1; }
}

# Merge hooks + env into settings JSON using python3
merge_settings() {
    need_python
    python3 - "$SETTINGS_FILE" "$BINARY_PATH" <<'PYEOF'
import json, sys, os

settings_path = sys.argv[1]
binary = sys.argv[2]

# Load existing or start fresh
if os.path.exists(settings_path):
    with open(settings_path) as f:
        data = json.load(f)
else:
    data = {}

# Ensure env section
data.setdefault("env", {})
data["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "200000"
data["env"]["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "70"

# Ensure MCP tool permissions are allowed
data.setdefault("permissions", {})
allowed = data["permissions"].setdefault("allow", [])
mcp_perms = [
    "mcp__claude-brain__memory_stats",
    "mcp__claude-brain__recall_context",
    "mcp__claude-brain__search_memory",
    "mcp__claude-brain__search_knowledge",
    "mcp__claude-brain__capture_thought",
    "mcp__claude-brain__consolidate_now",
    "mcp__claude-brain__learn",
]
for perm in mcp_perms:
    if perm not in allowed:
        allowed.append(perm)

# Build hook entries
hook_defs = {
    "SessionStart": {"command": f"{binary} hook session-start", "timeout": 5},
    "Stop":         {"command": f"{binary} hook stop",          "timeout": 30},
    "PreCompact":   {"command": f"{binary} hook pre-compact",   "timeout": 10},
    "PostCompact":  {"command": f"{binary} hook post-compact",  "timeout": 10},
}

data.setdefault("hooks", {})
for event, hook_cfg in hook_defs.items():
    entry = {"hooks": [{"type": "command", **hook_cfg}]}
    existing = data["hooks"].get(event, [])

    # Check if claude-brain hook already exists in the list
    already = False
    for group in existing:
        for h in group.get("hooks", []):
            if "claude-brain" in h.get("command", ""):
                # Update in place
                h.update({"type": "command", **hook_cfg})
                already = True
                break

    if not already:
        existing.append(entry)
    data["hooks"][event] = existing

with open(settings_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"  Updated {settings_path}")
PYEOF
}

# Remove claude-brain hooks from settings JSON
remove_settings() {
    need_python
    [ -f "$SETTINGS_FILE" ] || return 0
    python3 - "$SETTINGS_FILE" <<'PYEOF'
import json, sys, os

settings_path = sys.argv[1]
if not os.path.exists(settings_path):
    sys.exit(0)

with open(settings_path) as f:
    data = json.load(f)

# Remove claude-brain hooks (keep other hooks intact)
hooks = data.get("hooks", {})
for event in list(hooks.keys()):
    groups = hooks[event]
    cleaned = []
    for group in groups:
        filtered = [h for h in group.get("hooks", []) if "claude-brain" not in h.get("command", "")]
        if filtered:
            group["hooks"] = filtered
            cleaned.append(group)
    if cleaned:
        hooks[event] = cleaned
    else:
        del hooks[event]

# Remove compaction env vars we set
env = data.get("env", {})
env.pop("CLAUDE_CODE_AUTO_COMPACT_WINDOW", None)
env.pop("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", None)
if not env:
    data.pop("env", None)
if not hooks:
    data.pop("hooks", None)

# Remove MCP tool permissions we added
perms = data.get("permissions", {})
allowed = perms.get("allow", [])
allowed = [p for p in allowed if not p.startswith("mcp__claude-brain__")]
if allowed:
    perms["allow"] = allowed
elif "allow" in perms:
    del perms["allow"]

with open(settings_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"  Cleaned {settings_path}")
PYEOF
}

# Merge MCP server entry into mcp.json
merge_mcp() {
    need_python
    python3 - "$MCP_FILE" "$BINARY_PATH" <<'PYEOF'
import json, sys, os

mcp_path = sys.argv[1]
binary = sys.argv[2]

if os.path.exists(mcp_path):
    with open(mcp_path) as f:
        data = json.load(f)
else:
    data = {}

data.setdefault("mcpServers", {})
data["mcpServers"]["claude-brain"] = {
    "command": binary,
    "args": ["serve"]
}

with open(mcp_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"  Updated {mcp_path}")
PYEOF
}

# Remove MCP server entry
remove_mcp() {
    need_python
    [ -f "$MCP_FILE" ] || return 0
    python3 - "$MCP_FILE" <<'PYEOF'
import json, sys, os

mcp_path = sys.argv[1]
if not os.path.exists(mcp_path):
    sys.exit(0)

with open(mcp_path) as f:
    data = json.load(f)

data.get("mcpServers", {}).pop("claude-brain", None)

with open(mcp_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"  Cleaned {mcp_path}")
PYEOF
}

# Install rules file from source
install_rules() {
    mkdir -p "$RULES_DIR"
    cp "$INTEGRATION_DIR/rules.md" "$RULES_FILE"
    echo "  Installed $RULES_FILE"
}

# Install skill file from source
install_skill() {
    mkdir -p "$SKILLS_DIR"
    cp "$INTEGRATION_DIR/SKILL.md" "$SKILL_FILE"
    echo "  Installed $SKILL_FILE"
}

# Write default config
install_config() {
    mkdir -p "$BRAINWIRES_DIR"
    if [ -f "$CONFIG_FILE" ]; then
        yellow "  Config exists, skipping: $CONFIG_FILE"
        return
    fi
    cat > "$CONFIG_FILE" <<'TOML'
[storage]
# Paths default to ~/.brainwires/ — uncomment to override
# brain_path = "/home/you/.brainwires/claude-brain"
# pks_path = "/home/you/.brainwires/pks.db"
# bks_path = "/home/you/.brainwires/bks.db"

[policy]
hot_max_age_hours = 24
warm_max_age_days = 7
hot_token_budget = 50000
keep_recent = 4
min_importance = 0.3

[session_start]
max_facts = 20
max_summaries = 5
max_context_tokens = 4000

[capture]
extract_facts = true
consolidation_threshold = 20
TOML
    echo "  Wrote default config: $CONFIG_FILE"
}

# ── Actions ────────────────────────────────────────────────────────────

do_install() {
    echo ""
    green "═══ Installing Claude Brain ($INSTALL_LABEL) ═══"
    echo ""
    echo "  Framework: $FRAMEWORK_DIR"
    echo "  Target:    $INSTALL_LABEL"
    echo "  Settings:  $SETTINGS_FILE"
    echo "  MCP:       $MCP_FILE"
    echo ""

    # 1. Build + install to ~/.cargo/bin/
    echo "Installing binary via cargo install..."
    (cd "$FRAMEWORK_DIR" && cargo install --path extras/claude-brain --force 2>&1 | tail -5)
    if [ ! -f "$BINARY_PATH" ]; then
        red "Install failed — binary not found at $BINARY_PATH"
        exit 1
    fi
    green "  Binary: $BINARY_PATH (stable path, survives cargo rebuilds)"
    echo ""

    # 2. Config
    echo "Setting up config..."
    install_config
    echo ""

    # 3. Hooks + env
    echo "Wiring hooks..."
    mkdir -p "$(dirname "$SETTINGS_FILE")"
    merge_settings
    echo ""

    # 4. MCP server
    echo "Wiring MCP server..."
    mkdir -p "$(dirname "$MCP_FILE")"
    merge_mcp
    echo ""

    # 5. Rules file
    echo "Installing rules..."
    install_rules
    echo ""

    # 6. Skill file
    echo "Installing skill..."
    install_skill
    echo ""

    green "═══ Installation Complete ═══"
    echo ""
    echo "  Start a new Claude Code session to activate."
    echo "  Hook log: $LOG_FILE"
    echo "  Config:   $CONFIG_FILE"
    if $GLOBAL; then
        echo ""
        yellow "  Global install — brain active in ALL Claude Code sessions."
        echo "  Per-project .claude/settings.local.json hooks will stack on top."
    fi
    echo ""
}

do_uninstall() {
    echo ""
    yellow "═══ Uninstalling Claude Brain ($INSTALL_LABEL) ═══"
    echo ""

    echo "Removing hooks from $SETTINGS_FILE..."
    remove_settings

    echo "Removing MCP entry from $MCP_FILE..."
    remove_mcp

    if [ -f "$RULES_FILE" ]; then
        rm "$RULES_FILE"
        echo "  Removed $RULES_FILE"
    fi

    if [ -f "$SKILL_FILE" ]; then
        rm "$SKILL_FILE"
        rmdir "$SKILLS_DIR" 2>/dev/null || true
        echo "  Removed $SKILL_FILE"
    fi

    echo ""
    yellow "═══ Uninstall Complete ═══"
    echo ""
    echo "  Binary kept at: $BINARY_PATH (run 'cargo uninstall claude-brain' to remove)"
    echo "  Data kept at:   $BRAINWIRES_DIR"
    echo "  To fully purge data: rm -rf $BRAINWIRES_DIR"
    echo ""
}

do_status() {
    echo ""
    echo "═══ Claude Brain Status ═══"
    echo ""

    # Binary
    if [ -f "$BINARY_PATH" ]; then
        green "  Binary:     $BINARY_PATH ($(du -h "$BINARY_PATH" | cut -f1))"
    else
        red   "  Binary:     NOT INSTALLED (run ./install.sh install)"
    fi

    # Config
    if [ -f "$CONFIG_FILE" ]; then
        green "  Config:     $CONFIG_FILE"
    else
        yellow "  Config:     not found (defaults will be used)"
    fi

    # Check global
    local global_settings="$HOME/.claude/settings.json"
    local global_mcp="$HOME/.claude/mcp.json"
    local global_rules="$HOME/.claude/rules/claude-brain.md"
    local global_skill="$HOME/.claude/skills/claude-brain/SKILL.md"

    echo ""
    echo "  Global (~/.claude/):"
    if [ -f "$global_settings" ] && grep -q "claude-brain" "$global_settings" 2>/dev/null; then
        green "    Hooks:    wired in $global_settings"
    else
        yellow "    Hooks:    not configured"
    fi
    if [ -f "$global_mcp" ] && grep -q "claude-brain" "$global_mcp" 2>/dev/null; then
        green "    MCP:      registered in $global_mcp"
    else
        yellow "    MCP:      not configured"
    fi
    if [ -f "$global_rules" ]; then
        green "    Rules:    $global_rules"
    else
        yellow "    Rules:    not installed"
    fi
    if [ -f "$global_skill" ]; then
        green "    Skill:    $global_skill"
    else
        yellow "    Skill:    not installed"
    fi

    # Check project (if not global mode or always show framework)
    if ! $GLOBAL; then
        local proj_settings="$CLAUDE_DIR/settings.local.json"
        local proj_mcp="$MCP_FILE"
        local proj_rules="$RULES_DIR/claude-brain.md"

        echo ""
        echo "  Project ($PROJECT_DIR):"
        if [ -f "$proj_settings" ] && grep -q "claude-brain" "$proj_settings" 2>/dev/null; then
            green "    Hooks:    wired in $proj_settings"
        else
            yellow "    Hooks:    not configured"
        fi
        if [ -f "$proj_mcp" ] && grep -q "claude-brain" "$proj_mcp" 2>/dev/null; then
            green "    MCP:      registered in $proj_mcp"
        else
            yellow "    MCP:      not configured"
        fi
        if [ -f "$proj_rules" ]; then
            green "    Rules:    $proj_rules"
        else
            yellow "    Rules:    not installed"
        fi
        if [ -f "$SKILL_FILE" ]; then
            green "    Skill:    $SKILL_FILE"
        else
            yellow "    Skill:    not installed"
        fi
    fi

    # Data
    echo ""
    if [ -d "$BRAINWIRES_DIR" ]; then
        local size
        size=$(du -sh "$BRAINWIRES_DIR" 2>/dev/null | cut -f1)
        green "  Data:       $BRAINWIRES_DIR ($size)"
    else
        yellow "  Data:       no data yet"
    fi

    # Recent log
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "  Last 5 log entries:"
        tail -5 "$LOG_FILE" | sed 's/^/    /'
    fi

    echo ""
}

# ── Main ───────────────────────────────────────────────────────────────

case "$ACTION" in
    install)   do_install   ;;
    uninstall) do_uninstall ;;
    status)    do_status    ;;
    *)
        echo "Usage: $0 [install|uninstall|status] [--global] [--project-dir /path/to/project]"
        exit 1
        ;;
esac
