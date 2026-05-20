# IPC and Remote Control Architecture

This document describes the Inter-Process Communication (IPC) system used by Brainwires CLI for local agent-viewer communication, and the Remote Control system for web-based agent management.

## Table of Contents

1. [Overview](#overview)
2. [Local IPC Architecture](#local-ipc-architecture)
3. [Multi-Agent System](#multi-agent-system)
4. [Remote Control System](#remote-control-system)
5. [Security Considerations](#security-considerations)
6. [Configuration](#configuration)

---

## Overview

Brainwires CLI uses a multi-layered communication architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Web Browser (brainwires-studio)                    │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ RemoteAgentsPanel│    │RemoteAgentViewer │    │   useRemoteAgent │       │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘       │
│           │                       │                       │                  │
│           └───────────────────────┼───────────────────────┘                  │
│                                   │                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │            GlobalRemoteCLIService (Singleton)                        │    │
│  │  - Manages Supabase Realtime channel subscription                    │    │
│  │  - Persists across page navigations                                  │    │
│  │  - Handles heartbeat, stream, event messages                         │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│                          Supabase Realtime WebSocket                         │
│                          Channel: cli:{userId}                               │
│                                    │                                         │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │  Supabase Realtime  │
                          │   (Cloud Service)   │
                          └──────────┬──────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────┐
│                          CLI Host Machine                                    │
│                                    │                                         │
│  ┌─────────────────────────────────┼───────────────────────────────────┐    │
│  │                    RemoteBridge (Rust)                               │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │              RealtimeClient (src/remote/realtime.rs)         │    │    │
│  │  │  - WebSocket connection to Supabase Realtime                 │    │    │
│  │  │  - Phoenix protocol (phx_join, heartbeat, broadcast)         │    │    │
│  │  │  - JWT token exchange for authentication                     │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │  - Collects agent status via HeartbeatCollector                      │    │
│  │  - Broadcasts heartbeats via Realtime channel                        │    │
│  │  - Receives and relays commands from web clients                     │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│           ┌────────────────────────┼────────────────────────────────┐       │
│           │                        │                                │       │
│           ▼                        ▼                                ▼       │
│  ┌─────────────────┐     ┌─────────────────┐              ┌─────────────────┐│
│  │  Agent Process  │     │  Agent Process  │     ...      │  Agent Process  ││
│  │  (session-001)  │     │  (session-002)  │              │  (session-00N)  ││
│  │  ┌───────────┐  │     │  ┌───────────┐  │              │  ┌───────────┐  ││
│  │  │ PTY/Term  │  │     │  │ PTY/Term  │  │              │  │ PTY/Term  │  ││
│  │  └───────────┘  │     │  └───────────┘  │              │  └───────────┘  ││
│  │  ┌───────────┐  │     │  ┌───────────┐  │              │  ┌───────────┐  ││
│  │  │Unix Socket│  │     │  │Unix Socket│  │              │  │Unix Socket│  ││
│  │  │  (IPC)    │  │     │  │  (IPC)    │  │              │  │  (IPC)    │  ││
│  │  └─────┬─────┘  │     │  └─────┬─────┘  │              │  └─────┬─────┘  ││
│  └────────┼────────┘     └────────┼────────┘              └────────┼────────┘│
│           │                       │                                │         │
│           ▼                       ▼                                ▼         │
│  ┌─────────────────┐     ┌─────────────────┐              ┌─────────────────┐│
│  │  TUI Viewer     │     │  TUI Viewer     │              │  TUI Viewer     ││
│  │  (attached)     │     │  (detached)     │              │  (attached)     ││
│  └─────────────────┘     └─────────────────┘              └─────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Local IPC Architecture

### Agent-Viewer Separation

Each agent runs as a separate process with its own:
- **PTY** (pseudo-terminal) for terminal I/O
- **Tokio runtime** for async operations
- **Unix domain socket** for IPC with the TUI viewer
- **Session state** (conversation history, MCP connections, etc.)

The TUI is a thin viewer that can attach/detach without losing state.

### TUI Session Management

The TUI uses an **event-driven architecture** for IPC communication. Events from the Agent (stream chunks, tool results, status updates) flow through a unified event channel alongside keyboard and terminal events.

**Starting a Session:**
```bash
brainwires chat
```
This spawns an Agent process and launches the TUI in viewer mode connected via IPC.

**Backgrounding a Session:**
- Press `Ctrl+Z` in the TUI to open the background dialog
- Choose "Background" to detach the TUI while keeping the Agent running
- The Agent continues processing any ongoing work

**Listing Sessions:**
```bash
brainwires sessions
```
Shows all backgrounded sessions with their status (running/stale).

**Attaching to a Session:**
```bash
# Attach to most recent session
brainwires attach

# Attach to specific session
brainwires attach <session-id>
```
Launches a new TUI that connects to the existing Agent via IPC. The Agent sends a `ConversationSync` message with full state on connect.

**Terminating a Session:**
```bash
brainwires exit <session-id>
```
Sends an `Exit` message to the Agent, causing it to shut down.

**Normal TUI Exit:**
When the TUI exits normally (via `Ctrl+C` or quit command), it sends `ViewerMessage::Exit` to shut down the Agent. Only `Ctrl+Z` backgrounding keeps the Agent running.

### Socket Location

Agent sockets are stored at:
```
~/.local/share/brainwires/sessions/<session-id>.sock
```

Metadata files are stored alongside:
```
~/.local/share/brainwires/sessions/<session-id>.meta.json
```

Log files for debugging:
```
~/.local/share/brainwires/sessions/<session-id>.stdout.log
~/.local/share/brainwires/sessions/<session-id>.stderr.log
```

### Protocol Format

Messages are newline-delimited JSON with a `type` field for discrimination:

```json
{"type":"user_input","content":"Hello world","context_files":[]}
```

### Viewer → Agent Messages (`ViewerMessage`)

| Type | Description | Fields |
|------|-------------|--------|
| `user_input` | User submitted text | `content`, `context_files` |
| `cancel` | Cancel current operation | - |
| `sync_request` | Request full state sync | - |
| `detach` | Viewer going to background | `exit_when_done` |
| `exit` | Request agent exit | - |
| `slash_command` | Execute slash command | `command`, `args` |
| `set_tool_mode` | Change tool mode | `mode` |
| `queue_message` | Queue message for injection | `content` |
| `acquire_lock` | Request resource lock | `resource_type`, `scope`, `description` |
| `release_lock` | Release resource lock | `resource_type`, `scope` |
| `query_locks` | Query lock status | `scope` (optional) |
| `list_agents` | List all active agents | - |
| `spawn_agent` | Spawn child agent | `model`, `reason`, `working_directory` |
| `disconnect` | Graceful viewer disconnect | - |

### Agent → Viewer Messages (`AgentMessage`)

| Type | Description | Fields |
|------|-------------|--------|
| `stream_chunk` | Streaming text delta | `text` |
| `stream_end` | Stream completed | `finish_reason` |
| `tool_call_start` | Tool execution started | `id`, `name`, `server`, `input` |
| `tool_progress` | Tool progress update | `name`, `message`, `progress` |
| `tool_result` | Tool execution completed | `id`, `name`, `output`, `error` |
| `conversation_sync` | Full state sync | `session_id`, `model`, `messages`, `status`, `is_busy`, `tool_mode`, `mcp_servers` |
| `message_added` | New message added | `message` |
| `status_update` | Status change | `status` |
| `task_update` | Task list update | `task_tree`, `task_count`, `completed_count` |
| `error` | Error occurred | `message`, `fatal` |
| `exiting` | Agent exiting | `reason` |
| `agent_spawned` | Child agent spawned | `new_session_id`, `parent_session_id`, `spawn_reason`, `model` |
| `agent_list` | Response to list_agents | `agents` |

### Handshake Protocol

1. **New Session:**
   ```json
   {"version":1,"is_reattach":false,"session_id":null}
   ```

2. **Reattach:**
   ```json
   {"version":1,"is_reattach":true,"session_id":"session-abc123"}
   ```

3. **Response:**
   ```json
   {"accepted":true,"session_id":"session-abc123","error":null}
   ```

---

## Multi-Agent System

### Agent Hierarchy

Agents can spawn child agents, forming a tree structure:

```
Root Agent (session-001)
├── Child Agent (session-002) - "Research task"
│   └── Grandchild Agent (session-003) - "Deep dive subtask"
└── Child Agent (session-004) - "Implementation task"
```

### Agent Metadata

Each agent writes metadata to `<session-id>.meta.json`:

```json
{
  "session_id": "session-001",
  "parent_agent_id": null,
  "spawn_reason": null,
  "model": "claude-3-5-sonnet",
  "created_at": 1702800000,
  "last_activity": 1702800100,
  "working_directory": "/home/user/project",
  "is_busy": false,
  "pid": 12345
}
```

### Resource Locking

Multi-agent coordination uses resource locks to prevent conflicts:

| Lock Type | Description |
|-----------|-------------|
| `build` | Build operations |
| `test` | Test execution |
| `build_test` | Combined build+test |
| `git_index` | Git staging area |
| `git_commit` | Git commits |
| `git_remote_write` | Git push |
| `git_remote_merge` | Git pull |
| `git_branch` | Branch operations |
| `git_destructive` | Destructive git ops |

### Agent Depth Limit

To prevent runaway recursion, agents have a maximum depth of **5 levels**.

### Parent-Child Communication

When a parent agent exits:
1. Children receive `ParentSignal` message
2. Based on signal type (`ParentExiting`, `Shutdown`, `Detached`), children:
   - Shut down if idle
   - Set `exit_when_done` if busy
   - Become orphaned (detach)

---

## Remote Control System

### Architecture

The remote control system allows web-based management of CLI agents using **Supabase Realtime** for bidirectional WebSocket communication:

1. **CLI ↔ Supabase Realtime (Bidirectional)**
   - CLI connects via WebSocket to Supabase Realtime
   - Uses Phoenix protocol for channel subscription
   - Authenticated via JWT token exchange (API key → Supabase JWT)
   - Channel: `cli:{userId}`

2. **Web ↔ Supabase Realtime (Bidirectional)**
   - Web clients subscribe to same channel via Supabase client
   - Receive heartbeats, streams, events in real-time
   - Send commands via channel broadcast

3. **No HTTP Polling or SSE**
   - All communication uses persistent WebSocket connections
   - Sub-second latency for real-time updates
   - Efficient resource usage (no polling overhead)

### CLI Components

#### RealtimeClient (`src/remote/realtime.rs`)

Supabase Realtime WebSocket client that:
- Establishes WebSocket connection to Supabase Realtime
- Implements Phoenix protocol (phx_join, heartbeat, broadcast)
- Exchanges API key for Supabase-compatible JWT via `/api/remote/connect`
- Handles automatic reconnection on disconnect

#### RemoteBridge (`crates/brainwires-framework/crates/brainwires-network/src/remote/bridge.rs`)

Bridge coordinator that:
- Manages RealtimeClient lifecycle
- Collects agent status via HeartbeatCollector
- Broadcasts heartbeats on interval via Realtime channel
- Receives and relays commands to appropriate agents
- Streams agent output back via Realtime

#### HeartbeatCollector (`src/remote/heartbeat.rs`)

Collects agent information from IPC metadata:
- Reads all `.meta.json` files in sessions directory
- Converts `AgentMetadata` to `RemoteAgentInfo`
- Calculates system load

#### RemoteBridgeManager (`src/remote/manager.rs`)

High-level lifecycle management:
- Start/stop bridge
- Configuration management
- Auto-start on first agent spawn (if enabled)

### Protocol Messages

All messages are broadcast on the Supabase Realtime channel with event type `remote`.

#### Realtime Message Wrapper (`RemoteRealtimeMessage`)

```typescript
{
  type: "remote.heartbeat" | "remote.stream" | "remote.event" | "remote.command" | ...,
  id: string,           // Message ID
  payload: object,      // Type-specific payload
  timestamp: number,    // Unix timestamp
  userId: string        // User ID for routing
}
```

#### CLI → Web (via Realtime)

| Type | Description | Payload Fields |
|------|-------------|----------------|
| `remote.register` | Initial registration | `hostname`, `os`, `version`, `cliId` |
| `remote.heartbeat` | Status update | `agents`, `hostname`, `os`, `version` |
| `remote.stream` | Agent output | `agentId`, `chunkType`, `content` |
| `remote.event` | Agent lifecycle | `agentId`, `eventType`, `data` |
| `remote.command_result` | Command response | `commandId`, `success`, `result`, `error` |
| `remote.pong` | Keepalive response | `timestamp` |

#### Web → CLI (via Realtime)

| Type | Description | Payload Fields |
|------|-------------|----------------|
| `remote.command` | Command to CLI | `type`, `agent_id`, `content`, `command`, `args` |

Command types within `remote.command`:
- `send_input`: Send text to agent (`content`)
- `slash_command`: Execute command (`command`, `args`)
- `cancel_operation`: Cancel current operation
- `subscribe`: Subscribe to agent stream (`agent_id`)
- `unsubscribe`: Unsubscribe from stream (`agent_id`)

### Backend Components (Next.js)

#### API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/api/remote/connect` | POST | JWT token exchange (API key → Supabase JWT) |

The `/api/remote/connect` endpoint:
1. Validates the API key
2. Looks up the associated user
3. Generates a Supabase-compatible JWT with user claims
4. Returns token with Realtime URL for WebSocket connection

#### Remote CLI Listener (`lib/realtime/remote-cli-listener.ts`)

Server-side Realtime listener (runs in Next.js instrumentation):
- Subscribes to `cli:*` channels for all users
- Processes incoming CLI messages
- Can trigger server-side actions on events

#### Remote CLI Channel (`lib/realtime/remote-cli-channel.ts`)

Client-side Realtime channel class:
- `RemoteCLIChannel`: Per-user channel subscription
- `GlobalRemoteCLIService`: Singleton that persists across page navigations
- Handles heartbeat timeout detection (30s without heartbeat = disconnected)

### Database Schema

```sql
-- CLI bridge connections
CREATE TABLE cli_connections (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  session_token TEXT NOT NULL,
  hostname TEXT,
  os TEXT,
  cli_version TEXT,
  connected_at TIMESTAMPTZ,
  last_heartbeat TIMESTAMPTZ,
  is_active BOOLEAN,
  system_load REAL,
  metadata JSONB
);

-- Agents reported by CLI
CREATE TABLE remote_agents (
  id UUID PRIMARY KEY,
  connection_id UUID REFERENCES cli_connections(id),
  session_id TEXT NOT NULL,
  model TEXT,
  is_busy BOOLEAN,
  parent_id TEXT,
  working_directory TEXT,
  message_count INTEGER,
  status TEXT,
  name TEXT,
  last_activity TIMESTAMPTZ,
  created_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ,
  metadata JSONB
);

-- Audit log for commands
CREATE TABLE remote_command_log (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  connection_id UUID REFERENCES cli_connections(id),
  agent_id TEXT NOT NULL,
  command_id TEXT NOT NULL,
  command_type TEXT NOT NULL,
  command_data JSONB,
  result_success BOOLEAN,
  result_data JSONB,
  error_message TEXT,
  sent_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ
);
```

### Web UI Components

| Component | Description |
|-----------|-------------|
| `RemoteAgentsPanel` | Sidebar showing all connected agents |
| `RemoteAgentCard` | Individual agent status card |
| `RemoteAgentViewer` | Full viewer for interacting with agent |

### React Hooks

| Hook | Description |
|------|-------------|
| `useRemoteAgents` | Subscribes to Realtime channel, manages global agent state via Jotai atoms |
| `useRemoteAgent` | Single agent interaction, sends commands via Realtime |

Both hooks use **Supabase Realtime exclusively** - no HTTP polling or SSE fallback. This ensures efficient resource usage and real-time updates with sub-second latency.

### Real-Time Streaming Pipeline

Communication flows through Supabase Realtime WebSocket:

```
Web Client                  Supabase Realtime           CLI Bridge              Agent
    │                              │                         │                      │
    │ Subscribe cli:{userId}       │                         │                      │
    │ ─────────────────────────────>                         │                      │
    │                              │                         │                      │
    │                              │ <─ Subscribe cli:{userId}                      │
    │                              │                         │                      │
    │                              │                         │ Connect to IPC socket│
    │                              │                         │ ──────────────────────>
    │                              │                         │                      │
    │                              │                         │ <─── AgentMessage    │
    │                              │                         │      (StreamChunk,   │
    │                              │                         │       ToolResult,    │
    │                              │ remote.stream           │       etc.)          │
    │                              │ <────────────────────────                      │
    │ Broadcast: remote.stream     │                         │                      │
    │ <─────────────────────────────                         │                      │
    │                              │                         │                      │
    │ Broadcast: remote.command    │                         │                      │
    │ ─────────────────────────────>                         │                      │
    │                              │ remote.command          │                      │
    │                              │ ─────────────────────────>                     │
    │                              │                         │ ViewerMessage        │
    │                              │                         │ ──────────────────────>
    │                              │                         │                      │
```

The streaming pipeline converts `AgentMessage` types to `StreamChunkType`:

| AgentMessage Type | StreamChunkType | Description |
|-------------------|-----------------|-------------|
| `StreamChunk` | `text` | AI response text |
| `StreamEnd` | `complete` | Response completed |
| `ToolCallStart` | `tool_call` | Tool execution started |
| `ToolResult` | `tool_result` | Tool completed |
| `ToolProgress` | `system` | Tool progress update |
| `Error` | `error` | Error occurred |
| `StatusUpdate` | `system` | Status change |
| `TaskUpdate` | `system` | Task list change |
| `Toast` | `system` | Toast notification |
| `ConversationSync` | `history` | Full conversation history on connect |

---

## Security Considerations

### Local IPC Security

1. **Unix Socket Permissions**: Sockets are created with user-only permissions
2. **File-based Discovery**: Only sockets in user's sessions directory are accessible
3. **Process Isolation**: Each agent runs in its own process

### Remote Control Security

1. **Outbound-Only**: CLI initiates all connections (no open ports)
2. **API Key Auth**: Initial authentication via `bw_*` API keys
3. **JWT Token Exchange**: API key exchanged for Supabase-compatible JWT with user claims
4. **Channel Isolation**: Each user has dedicated channel (`cli:{userId}`)
5. **TLS Required**: All WebSocket connections over WSS (TLS)
6. **Heartbeat Timeout**: Connections marked disconnected after 30s without heartbeat
7. **Audit Logging**: All commands logged to `remote_command_log`
8. **Row Level Security**: Database policies enforce user isolation
9. **No Polling Overhead**: WebSocket connections are persistent and efficient

---

## Configuration

### CLI Configuration (`~/.brainwires/config.json`)

```json
{
  "remote": {
    "enabled": false,
    "backend_url": "https://brainwires.studio/api/remote/heartbeat",
    "api_key": "bw_prod_xxxxx",
    "heartbeat_interval_secs": 30,
    "reconnect_delay_secs": 5,
    "max_reconnect_attempts": 10,
    "auto_start": true
  }
}
```

### CLI Commands

```bash
# View remote config
brainwires remote config

# Enable remote control
brainwires remote config --enabled true

# Set API key
brainwires remote config --api-key bw_prod_xxxxx

# Start bridge manually
brainwires remote start

# Check status
brainwires remote status

# Stop bridge
brainwires remote stop
```

---

## Testing

### Rust Tests (26 tests)

```bash
cargo test --lib remote::
```

Tests cover:
- Protocol serialization/deserialization
- Bridge configuration defaults
- URL parsing and WebSocket key generation
- Heartbeat collection
- Manager lifecycle
- Agent metadata conversion
- Stream chunk conversion (text, tool calls, errors, etc.)
- Non-streamable message filtering

### TypeScript Tests (16 tests)

```bash
npm test -- --testPathPattern="remote"
```

Tests cover:
- Protocol type definitions
- Audit log data extraction
- Command sanitization
- Stream chunk types

---

## Files Reference

### CLI (brainwires-cli)

| File | Description |
|------|-------------|
| `src/ipc/mod.rs` | IPC module root |
| `src/ipc/protocol.rs` | Local IPC message types |
| `src/ipc/socket.rs` | Unix socket utilities |
| `src/remote/mod.rs` | Remote module root |
| `src/remote/protocol.rs` | Remote protocol types |
| `crates/brainwires-framework/crates/brainwires-network/src/remote/bridge.rs` | Bridge coordinator |
| `src/remote/realtime.rs` | Supabase Realtime WebSocket client |
| `src/remote/heartbeat.rs` | Agent status collection |
| `src/remote/manager.rs` | Bridge lifecycle management |
| `src/cli/remote.rs` | CLI commands |

### Backend (brainwires-studio)

| File | Description |
|------|-------------|
| `lib/remote/protocol.ts` | TypeScript protocol types (Realtime messages) |
| `lib/remote/audit-log.ts` | Command audit logging |
| `lib/realtime/remote-cli-channel.ts` | Client-side Realtime channel class |
| `lib/realtime/remote-cli-listener.ts` | Server-side Realtime listener |
| `app/api/remote/connect/route.ts` | JWT token exchange endpoint |
| `hooks/use-remote-agents.ts` | Agent list hook (Realtime-only) |
| `hooks/use-remote-agent.ts` | Single agent hook (Realtime-only) |
| `components/remote/*.tsx` | UI components |
| `instrumentation.ts` | Next.js instrumentation (starts Realtime listener) |
| `supabase/migrations/20251217000001_*.sql` | Database schema |
