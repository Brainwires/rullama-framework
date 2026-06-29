# reload-daemon

A minimal MCP server that enables AI coding clients to kill and restart themselves with transformed arguments.

## Overview

AI coding assistants (Claude Code, Cursor, etc.) sometimes need to restart themselves — to switch modes, update permissions, or continue a session with different flags. They can't do this directly because killing your own process means you can't spawn the replacement.

`reload-daemon` solves this by running as a sidecar MCP server. The client calls the `reload_app` tool with its own PID and arguments, and the daemon handles the kill-and-respawn cycle on its behalf.

## How It Works

```text
  ┌──────────────┐          ┌──────────────┐
  │  AI Client   │  reload  │   Reload     │
  │ (Claude Code)├─────────►│   Daemon     │
  │  pid: 1234   │ HTTP/MCP │  :3100/mcp   │
  └──────┬───────┘          └──────┬───────┘
         │                         │
         │  1. Compute new args    │
         │  2. Send SIGINT ────────┤
         │  3. Wait / escalate     │
         │  4. Send SIGTERM ───────┤  (if still alive)
         │  5. Send SIGKILL ───────┤  (if still alive)
         │                         │
         │  6. Spawn replacement   │
         │                         │
  ┌──────▼───────┐                 │
  │  New Client  │◄────────────────┘
  │  pid: 5678   │
  └──────────────┘
```

1. Client calls `reload_app` with its PID, original argv, and working directory
2. Daemon validates the binary name against the config (prevents mis-targeting)
3. New arguments are computed **before** killing (safety: plan before acting)
4. Daemon sends escalating signals (SIGINT → SIGTERM → SIGKILL) with configurable timeouts
5. Once the process is dead, the daemon spawns a replacement with transformed args
6. The new process is detached (survives daemon exit, adopted by init/PID 1)

## Files

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI entry point, Axum HTTP server setup, graceful shutdown |
| `src/config.rs` | `DaemonConfig`, `ClientStrategy`, and `ArgsTransform` structs |
| `config.json` | Example configuration for Claude Code |
| `src/server.rs` | `ReloadServer` MCP handler with `reload_app` tool definition |
| `src/reload.rs` | Process killing (escalating signals), arg transformation, spawning |

## Configuration

The daemon is driven by a JSON config file:

```json
{
  "listen": "127.0.0.1:3100",
  "clients": {
    "claude-code": {
      "process_name": "claude",
      "kill_signals": ["SIGINT", "SIGTERM", "SIGKILL"],
      "kill_timeouts_ms": [2000, 3000, 0],
      "restart_args_transform": {
        "preserve_flags": ["--allow-dangerously-skip-permissions"],
        "replace_trailing": ["--continue", "continue"]
      }
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `listen` | `String` | Address and port to bind the HTTP server |
| `clients` | `Map<String, ClientStrategy>` | Named client configurations, keyed by `client_type` |

### ClientStrategy

| Field | Type | Description |
|-------|------|-------------|
| `process_name` | `String` | Expected binary name (validated against argv[0]) |
| `kill_signals` | `Vec<String>` | Signals to send in order: `SIGINT`, `SIGTERM`, `SIGKILL`, `SIGHUP`, `SIGUSR1`, `SIGUSR2` |
| `kill_timeouts_ms` | `Vec<u64>` | Timeout per signal before escalating. `0` = fire-and-forget (used for SIGKILL) |
| `restart_args_transform` | `Option<ArgsTransform>` | How to transform args for the replacement process |

### ArgsTransform

| Field | Type | Description |
|-------|------|-------------|
| `preserve_flags` | `Vec<String>` | Flags from the original argv to keep |
| `replace_trailing` | `Vec<String>` | Arguments appended after preserved flags |

In the example config, this means: keep `--allow-dangerously-skip-permissions` if present, drop everything else, then append `--continue continue`.

## Usage

### 1. Start the daemon

```sh
cargo run -p reload-daemon -- \
  --config extras/reload-daemon/config.json
```

Or with debug logging:

```sh
RUST_LOG=debug cargo run -p reload-daemon -- \
  --config extras/reload-daemon/config.json
```

### 2. Register with Claude Code

```sh
claude mcp add --transport http reload-daemon http://127.0.0.1:3100/mcp
```

### 3. The client calls the tool

The AI client invokes the `reload_app` MCP tool when it needs to restart itself.

## Tool: `reload_app`

| Parameter | Type | Description |
|-----------|------|-------------|
| `client_type` | `String` | Key into the `clients` config map (e.g. `"claude-code"`) |
| `pid` | `i32` | PID of the calling process |
| `original_args` | `Vec<String>` | Full original argv (program path + arguments) |
| `working_directory` | `String` | CWD for the replacement process |

**Behavior:**

1. Looks up the `ClientStrategy` for `client_type`
2. Validates that the binary name in `original_args[0]` matches `process_name`
3. Builds replacement args using `ArgsTransform` (if configured) or passes original args through
4. Kills the process with escalating signals, polling every 100ms between attempts
5. Spawns the replacement detached with stdin/stdout/stderr redirected to null
6. Returns a success message with the new args

**Error cases:**
- Unknown `client_type` → error
- Binary name mismatch → error
- Empty `original_args` → error
- Process survives all signals → error
- Spawn failure → error

## Platform Support

Unix only. The daemon uses POSIX signals (`libc::kill`) for process control. On non-Unix platforms, `kill_process` returns an error.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
