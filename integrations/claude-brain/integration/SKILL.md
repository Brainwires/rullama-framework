---
name: claude-brain
description: Check Claude Brain status — shows whether hooks, MCP server, and memory are working
user-invocable: true
allowed-tools: Bash, Read
---

# Claude Brain Status Check

Run the following checks and report results concisely:

1. Binary exists? Check `$CLAUDE_BRAIN_BINARY` (set by install.sh) or default path
2. Hook log — last 5 entries from `~/.brainwires/claude-brain-hooks.log`
3. Memory stats — run `memory_stats` MCP tool if available
4. Config — show active settings from `~/.brainwires/claude-brain.toml`

Report a concise status summary showing what's working and what's not.
