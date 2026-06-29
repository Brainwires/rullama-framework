#!/bin/bash
# Test script for claude-brain compaction hooks
#
# HOW TO USE:
# 1. Run this script first to set up test mode
# 2. Open a NEW claude code session in the brainwires-framework dir
# 3. Ask Claude to read several large files (fills context fast)
# 4. Watch ~/.brainwires/claude-brain-hooks.log for hook events
# 5. When done, run: ./test-compaction.sh restore
#
# The script sets a tiny 20K window at 30% trigger — compaction fires FAST.

SETTINGS="/home/nightness/dev/brainwires-framework/.claude/settings.local.json"
BACKUP="$SETTINGS.backup"
LOG="$HOME/.brainwires/claude-brain-hooks.log"

case "${1:-setup}" in
  setup)
    echo "=== Claude Brain Compaction Test Setup ==="
    echo ""

    # Backup current settings
    cp "$SETTINGS" "$BACKUP"
    echo "Backed up settings to $BACKUP"

    # Clear old log
    mkdir -p "$HOME/.brainwires"
    > "$LOG"
    echo "Cleared hook log at $LOG"

    # Patch env vars to tiny window
    python3 -c "
import json
with open('$SETTINGS') as f:
    d = json.load(f)
d['env']['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] = '20000'
d['env']['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] = '30'
with open('$SETTINGS', 'w') as f:
    json.dump(d, f, indent=2)
print('Set window=20000, trigger=30%')
"

    echo ""
    echo "TEST MODE ACTIVE — tiny context window!"
    echo ""
    echo "Now:"
    echo "  1. Open a NEW claude code session in brainwires-framework/"
    echo "  2. Ask Claude to read 2-3 large files"
    echo "  3. Watch the log:  tail -f $LOG"
    echo "  4. When done:  ./test-compaction.sh restore"
    ;;

  restore)
    echo "=== Restoring Production Settings ==="
    if [ -f "$BACKUP" ]; then
      cp "$BACKUP" "$SETTINGS"
      rm "$BACKUP"
      echo "Restored from backup"
    else
      echo "No backup found — manually set window/trigger in settings"
    fi
    echo ""
    echo "Hook log preserved at: $LOG"
    echo "Contents:"
    cat "$LOG" 2>/dev/null || echo "(empty)"
    ;;

  log)
    echo "=== Hook Log ==="
    cat "$LOG" 2>/dev/null || echo "(empty)"
    ;;

  watch)
    echo "=== Watching Hook Log (Ctrl+C to stop) ==="
    tail -f "$LOG"
    ;;

  *)
    echo "Usage: $0 [setup|restore|log|watch]"
    ;;
esac
