# brainwires-scheduler

A local-machine MCP server for cron-based job scheduling with optional Docker sandboxing.

## Overview

`brainwires-scheduler` is the on-premise equivalent of Claude Code's cloud `/schedule` feature. It runs as a persistent background daemon on your machine and exposes a full scheduling interface as MCP tools. Claude (or any MCP client) can create, manage, and trigger cron jobs — with full access to local files, environment variables, and tools — without your machine needing to stay connected to any cloud service.

Jobs optionally run inside Docker containers for isolation: memory limits, CPU caps, network lockdown, and read-only mounts, all configurable per job.

## How It Works

```text
  ┌────────────────┐  MCP (stdio)   ┌──────────────────────────────┐
  │  Claude Code   ├───────────────►│     brainwires-scheduler      │
  │  (or any MCP   │                │                               │
  │   client)      │◄───────────────┤  ┌──────────────────────────┐ │
  └────────────────┘  tool results  │  │   Scheduler Daemon Loop  │ │
                                    │  │  (tokio, cron expressions)│ │
                                    │  └────────────┬─────────────┘ │
                                    │               │ fires jobs     │
                                    │  ┌────────────▼─────────────┐ │
                                    │  │      Job Executor        │ │
                                    │  │  native  │  docker run   │ │
                                    │  └────────────┬─────────────┘ │
                                    │               │               │
                                    │  ┌────────────▼─────────────┐ │
                                    │  │  ~/.brainwires/scheduler/ │ │
                                    │  │  jobs.json  logs/<id>/   │ │
                                    │  └──────────────────────────┘ │
                                    └──────────────────────────────┘
```

1. Claude calls `add_job` with a cron expression, command, and optional Docker config
2. The daemon persists the job to `jobs.json` and begins scheduling it
3. At each cron tick, the executor launches the command natively or via `docker run`
4. Stdout, stderr, exit code, and duration are logged per-run
5. Claude can query status, fetch logs, or trigger jobs on-demand at any time

## Quick Start

```sh
# Build
cargo build -p brainwires-scheduler --release

# Register with Claude Code (stdio transport)
claude mcp add --transport stdio brainwires-scheduler \
  ./target/release/brainwires-scheduler

# Or run directly for development
cargo run -p brainwires-scheduler
```

## Usage

```sh
brainwires-scheduler [OPTIONS]

Options:
  --jobs-dir <PATH>        Storage directory [default: ~/.brainwires/scheduler/]
  --max-concurrent <N>     Max parallel jobs [default: 4]
  --http <ADDR>            Also serve HTTP MCP at this address (e.g. 127.0.0.1:3200)
  -h, --help               Print help
  -V, --version            Print version
```

### Stdio mode (default — for Claude Code)

```sh
brainwires-scheduler
```

Logs are written to stderr; stdout is reserved for the MCP wire protocol.

### HTTP mode (remote access or multiple clients)

```sh
brainwires-scheduler --http 127.0.0.1:3200
```

Register the HTTP endpoint separately:

```sh
claude mcp add --transport http brainwires-scheduler-http http://127.0.0.1:3200/mcp
```

Both transports can run simultaneously — the stdio server is always started, and `--http` adds a second listener.

## Claude Code Integration

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "brainwires-scheduler": {
      "command": "brainwires-scheduler",
      "args": []
    }
  }
}
```

Or with a custom storage directory:

```json
{
  "mcpServers": {
    "brainwires-scheduler": {
      "command": "brainwires-scheduler",
      "args": ["--jobs-dir", "/var/brainwires/scheduler"]
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `add_job` | Add a new scheduled job. Returns the generated job ID. |
| `remove_job` | Permanently delete a job by ID. |
| `list_jobs` | List all jobs with enabled status, next run time, and last result. |
| `get_job` | Full job details: config, sandbox, failure policy, next/last run. |
| `enable_job` | Re-enable a disabled job. |
| `disable_job` | Pause a job without deleting it. |
| `run_job` | Trigger a job immediately (outside its schedule). Awaits completion. |
| `get_logs` | Fetch stdout/stderr from recent executions (newest first). |
| `status` | Overall daemon status: uptime, job counts, failure summary. |

### `add_job` parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `string` | ✓ | Human-readable job name |
| `cron` | `string` | ✓ | Cron expression (see below) |
| `command` | `string` | ✓ | Executable (on PATH or absolute path) |
| `args` | `string[]` | | Arguments passed to the command |
| `working_dir` | `string` | | Working directory (defaults to daemon CWD) |
| `timeout_secs` | `integer` | | Kill timeout in seconds (default: 3600) |
| `failure_policy` | `object` | | What to do on non-zero exit (see below) |
| `sandbox` | `object` | | Docker sandbox config (see below) |
| `env` | `object` | | Extra environment variables `{KEY: VALUE}` |

### `get_logs` parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `string` | ✓ | Job ID |
| `limit` | `integer` | | Number of recent runs to return (default: 5, max: 20) |

## Cron Expressions

Standard 5-field expressions are supported (`min hour dom month dow`):

| Expression | Meaning |
|------------|---------|
| `* * * * *` | Every minute |
| `0 * * * *` | Every hour at :00 |
| `0 9 * * 1-5` | Weekdays at 09:00 |
| `30 2 * * *` | Daily at 02:30 |
| `0 0 1 * *` | First of the month at midnight |
| `*/15 * * * *` | Every 15 minutes |

7-field expressions with a leading seconds field are also accepted (e.g. `0 */5 * * * *` = every 5 minutes on the dot).

## Failure Policies

Configure what happens when a job exits non-zero:

```json
{ "type": "ignore" }
```
Log the failure and continue scheduling. **Default.**

```json
{ "type": "retry", "max_retries": 3, "backoff_secs": 60 }
```
Retry up to `max_retries` times with `backoff_secs` between attempts.

```json
{ "type": "disable" }
```
Disable the job permanently after the first failure. Re-enable with `enable_job`.

## Docker Sandboxing

Set `sandbox` on any job to run it inside a Docker container:

```json
{
  "image": "ubuntu:24.04",
  "memory_mb": 512,
  "cpu_limit": 1.0,
  "network": false,
  "mounts": ["/data/input:/data/input:ro"],
  "extra_args": ["--cap-drop=ALL"]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `string` | — | Docker image to run (e.g. `"alpine"`, `"rust:latest"`) |
| `memory_mb` | `integer` | `512` | Memory limit in megabytes |
| `cpu_limit` | `float` | `1.0` | CPU shares (e.g. `0.5` = half a core) |
| `network` | `bool` | `false` | Allow outbound network access |
| `mounts` | `string[]` | `[]` | Volume mounts: `"host:container"` or `"host:container:ro"` |
| `extra_args` | `string[]` | `[]` | Flags forwarded verbatim to `docker run` (escape hatch) |

The working directory is automatically mounted at the same path inside the container and set as the container's working directory, so relative paths in commands work as expected.

**Docker availability:** If Docker is configured for a job but the `docker` binary is unavailable or the daemon isn't running, the job fails with a clear error — it will never silently fall back to native execution.

### Example: isolated Python script

```json
{
  "name": "nightly data export",
  "cron": "0 2 * * *",
  "command": "python3",
  "args": ["/data/export.py"],
  "sandbox": {
    "image": "python:3.12-slim",
    "memory_mb": 256,
    "network": false,
    "mounts": ["/data:/data:ro", "/data/out:/data/out"]
  }
}
```

### Example: Rust CI check

```json
{
  "name": "cargo clippy",
  "cron": "0 */6 * * *",
  "command": "cargo",
  "args": ["clippy", "--all-targets", "--", "-D", "warnings"],
  "working_dir": "/home/user/projects/my-crate",
  "sandbox": {
    "image": "rust:latest",
    "memory_mb": 2048,
    "cpu_limit": 2.0,
    "network": false,
    "mounts": ["/home/user/.cargo/registry:/usr/local/cargo/registry:ro"]
  },
  "failure_policy": { "type": "disable" }
}
```

## Storage Layout

```
~/.brainwires/scheduler/
├── jobs.json                  ← all job definitions (rewritten on every change)
└── logs/
    └── <job-id>/
        ├── 20260401T020000Z.json
        ├── 20260402T020000Z.json
        └── ...                ← newest 20 runs kept, older pruned automatically
```

Each log file is a JSON `JobResult`:

```json
{
  "success": true,
  "exit_code": 0,
  "stdout": "exported 1423 rows\n",
  "stderr": "",
  "started_at": "2026-04-01T02:00:01Z",
  "duration_secs": 4.72,
  "error": null
}
```

Output is capped at 4 KB per stream (tail-truncated with a `[...truncated]` marker).

## Files

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI entry point, daemon spawn, stdio + optional HTTP transport |
| `src/config.rs` | Startup config struct and `jobs_dir` resolution |
| `src/job.rs` | `Job`, `DockerSandbox`, `FailurePolicy`, `JobResult` types |
| `src/store.rs` | JSON-backed job registry and per-run log file management |
| `src/executor.rs` | Native `tokio::process` and `docker run` execution paths |
| `src/daemon.rs` | Scheduler loop, `DaemonHandle` command channel, cron helpers |
| `src/server.rs` | `SchedulerServer` — rmcp `#[tool_router]` with 9 MCP tools |
| `src/lib.rs` | Re-exports for use as a library crate |

## Scheduling Semantics

- Jobs are scheduled by comparing the current time to `next_cron_tick_after(last_fired_at)`.
- Newly added jobs wait for the **next** natural cron tick — they do not fire immediately on creation.
- On daemon restart, `last_fired_at` is restored from `jobs.json`. Missed ticks while the daemon was down are **not** replayed.
- The scheduler wakes at the earliest upcoming tick across all enabled jobs, sleeping no longer than 60 seconds between checks (ensuring commands from the MCP server are serviced promptly).

## Logging

Set `RUST_LOG` to control log verbosity. All logs go to **stderr** so they don't interfere with the MCP wire protocol on stdout.

```sh
RUST_LOG=debug brainwires-scheduler
RUST_LOG=brainwires_scheduler=trace brainwires-scheduler
```

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or [MIT License](../../LICENSE-MIT) at your option.
