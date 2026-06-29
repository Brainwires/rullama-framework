# agent-chat

> **This is the minimal reference implementation** — a small, readable example of building a chat client on the Brainwires Framework. For a full-featured CLI tool with multi-agent orchestration, MCP server mode, infinite context, and more, see [`extras/brainwires-cli/`](../brainwires-cli/).

A simplified, open-source AI chat client built on the [Brainwires Framework](../../). Supports all cloud providers, built-in tool execution, and both plain (readline) and fullscreen TUI modes.

## Quick Start

```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-..."

# Plain mode (readline-style)
cargo run -p agent-chat -- --provider anthropic --model claude-sonnet-4-20250514

# TUI mode (fullscreen)
cargo run -p agent-chat -- --tui
```

Or install it:

```bash
cargo install --path extras/agent-chat
agent-chat --provider anthropic
```

## Usage

```
agent-chat [OPTIONS] [COMMAND]

Options:
  -p, --provider <PROVIDER>    AI provider (anthropic, openai, google, groq, ollama, together, fireworks)
  -m, --model <MODEL>          Model name (e.g. claude-sonnet-4-20250514, gpt-4o)
  -s, --system <SYSTEM>        System prompt
      --tui                    Use fullscreen TUI mode
      --max-tokens <N>         Maximum tokens to generate
      --temperature <F>        Temperature (0.0 - 1.0)
      --api-key <KEY>          API key (overrides env/config)

Commands:
  config   Manage configuration
  models   List available models
  auth     Manage API keys
```

## Modes

### Plain Mode (default)

Readline-style chat in the terminal. Tokens stream directly to stdout, tool approvals are prompted on stderr.

```
> What files are in the current directory?
[calling list_directory: {"path":"."}]
[list_directory ok: src/ Cargo.toml ...]

The current directory contains...
```

Slash commands: `/help`, `/clear`, `/exit`

### TUI Mode (`--tui`)

Fullscreen terminal UI with scrollable chat, input area, and status bar.

| Key | Action |
|-----|--------|
| Enter | Send message |
| Ctrl+C | Exit (confirm) |
| F1 | Help overlay |
| F2 | Console/debug log |
| F3 | Fullscreen chat |
| F4 | Fullscreen input |
| Up/Down | Scroll chat history |

Tool approval shows a popup with `[Y]es / [N]o / [A]lways` options.

## Configuration

Config lives in `~/.brainwires/chat/`.

```bash
# List all settings
agent-chat config list

# Set defaults
agent-chat config set default_provider anthropic
agent-chat config set default_model claude-sonnet-4-20250514
agent-chat config set temperature 0.5
agent-chat config set max_tokens 8192
agent-chat config set permission_mode auto   # auto | ask | reject
```

## API Keys

Keys are resolved in order: `--api-key` flag > environment variable > `~/.brainwires/chat/api_keys.toml`.

```bash
# Save a key (prompted securely)
agent-chat auth set anthropic

# List configured providers
agent-chat auth show

# Remove a key
agent-chat auth remove anthropic
```

Environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`.

## Available Models

```bash
# List all known models
agent-chat models

# Filter by provider
agent-chat models --provider openai
```

## Built-in Tools

The AI assistant has access to these tools during chat:

| Tool | Description |
|------|-------------|
| `bash` / `execute_command` | Shell command execution |
| `read_file`, `write_file`, `edit_file`, ... | File operations |
| `git_status`, `git_diff`, `git_log`, ... | Git operations |
| `search_code`, `search_files` | Regex-based code search |
| `check_duplicates`, `verify_build`, `check_syntax` | Code validation |
| `fetch_url` | Web fetching |

Permission modes control tool approval:
- **auto** - All tools run without prompting
- **ask** (default) - Prompts before each tool call (with "Always" option)
- **reject** - Only read-only tools are allowed

## Features

```toml
[features]
default = []
bedrock = ["brainwires-provider/bedrock"]    # AWS Bedrock
vertex-ai = ["brainwires-provider/vertex-ai"] # Google Vertex AI
```

## License

MIT OR Apache-2.0
