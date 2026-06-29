# brainwires-rag-server

Standalone MCP server binary for codebase RAG (Retrieval-Augmented Generation).

This is the executable wrapper for `brainwires-knowledge::rag` (formerly the standalone `brainwires-rag` crate, absorbed in the 0.10 consolidation) that exposes all indexing, search, and code navigation functionality as MCP tools and slash commands.

## Quick Start

```bash
cargo run -p brainwires-rag-server -- serve
```

Or install and run:

```bash
cargo install --path extras/brainwires-rag-server
brainwires-rag serve
```

## Claude Desktop Configuration

Add to `~/.claude/mcp_servers.json`:

```json
{
  "brainwires-rag": {
    "command": "/path/to/brainwires-rag",
    "args": ["serve"]
  }
}
```

## MCP Tools & Slash Commands

| Tool | Slash Command | Description |
|------|---------------|-------------|
| `index_codebase` | `/project:index` | Smart indexing (auto full or incremental) |
| `query_codebase` | `/project:query` | Semantic search with adaptive thresholds |
| `get_statistics` | `/project:stats` | Index statistics and language breakdown |
| `clear_index` | `/project:clear` | Clear all indexed data |
| `search_by_filters` | `/project:search` | Advanced filtered search |
| `search_git_history` | `/project:git-search` | Semantic search over commit history |
| `find_definition` | `/project:definition` | Find where a symbol is defined |
| `find_references` | `/project:references` | Find all references to a symbol |
| `get_call_graph` | `/project:callgraph` | Analyze function caller/callee relationships |

## Library vs Server

- **Library**: Use `brainwires-knowledge` crate (feature `rag`) in your Rust code for programmatic access
- **Server**: Use this binary to expose RAG as an MCP server for any AI assistant

See [brainwires-knowledge README](../../crates/brainwires-knowledge/README.md) for full API documentation.

## License

Licensed under either MIT or Apache-2.0 at your option. See [LICENSE-MIT](https://github.com/Brainwires/brainwires-framework/blob/main/LICENSE-MIT) and [LICENSE-APACHE](https://github.com/Brainwires/brainwires-framework/blob/main/LICENSE-APACHE).
