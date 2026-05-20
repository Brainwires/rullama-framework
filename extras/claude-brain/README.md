# claude-brain

Claude Code context manager that replaces compaction's lossy LLM summary with Brainwires tiered memory, vector search, and knowledge extraction.

## Overview

Claude Code's built-in compaction flattens your full conversation into an LLM-generated paragraph. Decisions, code context, and architectural reasoning get lost. `claude-brain` augments the compaction lifecycle via Claude Code hooks so that context survives compaction, spans sessions, and stays semantically searchable.

It ships as a single Rust binary with two modes:

- `claude-brain serve` — long-lived MCP server (stdio) exposing memory tools during a session.
- `claude-brain hook <event>` — short-lived hook handler invoked by Claude Code on lifecycle events.

## Architecture

```text
  Claude Code ──► hooks (stdin JSON) ──► claude-brain hook <event>
             └──► MCP (stdio)        ──► claude-brain serve
                                           │
                                           ▼
                                    ContextManager
                                  (wraps BrainClient)
                                           │
                         ┌─────────────────┼──────────────────┐
                         ▼                 ▼                  ▼
                    LanceDB            SQLite (PKS)       SQLite (BKS)
                    thoughts +         personal facts     behavioral
                    384d vectors       (preferences,      truths
                                        identity)         (cross-session)
```

Embeddings use `all-MiniLM-L6-v2` via FastEmbed (ONNX, fully local, no API calls).

## MCP Tools

Exposed when `claude-brain serve` is running:

| Tool | Description |
|------|-------------|
| `recall_context` | Semantic search over conversation history outside the current window |
| `capture_thought` | Persist a decision, insight, or fact (auto-categorized, embedded, fact-extracted) |
| `search_memory` | Semantic search across all memory tiers (thoughts + PKS + BKS) |
| `search_knowledge` | Keyword search over PKS facts and BKS truths |
| `memory_stats` | Dashboard of memory counts, categories, tags, and recent activity |

## Hook Handlers

Invoked as `claude-brain hook <event>`, with Claude Code passing a JSON payload on stdin:

| Event | Timeout | Purpose |
|-------|---------|---------|
| `session-start` | 5s | Load prior context on new/resume/compact starts; stdout is injected into Claude's context |
| `stop` | 30s | Capture every turn (user + assistant messages) to vector storage |
| `pre-compact` | 10s | Export full JSONL transcript to LanceDB and create a session digest before compaction runs |
| `post-compact` | 10s | Logging only — Claude Code ignores PostCompact stdout |

Only `SessionStart` and `UserPromptSubmit` stdout reaches Claude's context, so all post-compaction restoration is routed through the `session-start` handler (invoked with `source="compact"`).

## Quick Start

```bash
# Build + wire into Claude Code (hooks, env vars, MCP, rules file)
./extras/claude-brain/install.sh --global

# Status / uninstall
./extras/claude-brain/install.sh status
./extras/claude-brain/install.sh uninstall
```

Use `--project-dir PATH` to scope the install to a single project instead of the global `~/.claude/` config.

Minimal config at `~/.brainwires/claude-brain.toml`:

```toml
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
```

Storage paths default to `~/.brainwires/` and can be overridden under `[storage]`.

## Testing

```bash
cd extras/claude-brain
./test-efficacy.sh            # budget math, routing, loop detection, output sizes
./test-compaction.sh setup    # shrink context window to force compaction for manual testing
```

Hook events are logged to `~/.brainwires/claude-brain-hooks.log`.

## Deep Dive

See [`TECH_BRIEFING.md`](TECH_BRIEFING.md) for the full architecture: storage schemas, evidence system, PKS fact extraction patterns, dream consolidation, and the complete compaction sequence.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
