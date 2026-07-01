# rullama-brain-server

Standalone MCP server binary for the Open Brain knowledge system.

This is the executable wrapper for `rullama-knowledge::knowledge` (the subsystem absorbed from the deprecated `rullama-brain` crate in the 0.10 consolidation) that exposes all thought capture, memory search, and knowledge-system functionality as MCP tools and prompts.

## Quick Start

```bash
cargo run -p rullama-brain-server -- serve
```

Or install and run:

```bash
cargo install --path servers/rullama-brain-server
rullama-brain serve
```

## Claude Desktop Configuration

Add to `~/.claude/mcp_servers.json`:

```json
{
  "rullama-brain": {
    "command": "/path/to/rullama-brain",
    "args": ["serve"]
  }
}
```

## MCP Tools & Prompts

| Tool | Description |
|------|-------------|
| `capture_thought` | Capture a thought with auto-detection, embedding, and PKS extraction |
| `search_memory` | Semantic search across thoughts and PKS facts |
| `list_recent` | Browse recent thoughts with category and time filters |
| `get_thought` | Retrieve a specific thought by UUID |
| `search_knowledge` | Query PKS personal facts and BKS behavioral truths |
| `memory_stats` | Dashboard of counts, categories, recency, and top tags |
| `delete_thought` | Delete a thought by UUID |

| Prompt | Description |
|--------|-------------|
| `capture` | Capture a new thought into persistent memory |
| `search` | Semantic search across all memory |
| `recent` | List recently captured thoughts |
| `stats` | Show memory statistics dashboard |
| `knowledge` | Search personal and behavioral knowledge |

## Library vs Server

- **Library**: Use `rullama-knowledge` crate (feature `knowledge`) in your Rust code for programmatic access
- **Server**: Use this binary to expose Open Brain as an MCP server for any AI assistant

See [rullama-knowledge README](../../crates/rullama-knowledge/README.md) for full API documentation.

## License

Licensed under either MIT or Apache-2.0 at your option. See [LICENSE-MIT](https://github.com/Brainwires/rullama-framework/blob/main/LICENSE-MIT) and [LICENSE-APACHE](https://github.com/Brainwires/rullama-framework/blob/main/LICENSE-APACHE).
