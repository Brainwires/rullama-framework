# A2A Protocol

The `@rullama/a2a` package implements Google's Agent-to-Agent (A2A) protocol
v1.0 for inter-agent communication using JSON-RPC and REST transports (no gRPC).

## Overview

A2A enables agents to discover each other, exchange messages, manage tasks, and
stream results. The protocol uses:

- **Agent Cards** for capability advertisement and discovery
- **JSON-RPC 2.0** for structured method calls
- **REST endpoints** for task management
- **SSE (Server-Sent Events)** for streaming responses

## Agent Card

An `AgentCard` describes what an agent can do, what protocols it supports, and
how to authenticate:

```ts
import type { AgentCapabilities, AgentCard, AgentSkill } from "@rullama/a2a";

const card: AgentCard = {
  name: "code-reviewer",
  description: "Reviews code for quality and security issues",
  url: "https://agent.example.com",
  version: "1.0.0",
  capabilities: { streaming: true, pushNotifications: false },
  skills: [
    { id: "review", name: "Code Review", description: "Reviews pull requests" },
  ],
  securitySchemes: {},
  security: [],
};
```

See: `../examples/a2a/agent_card.ts`.

## A2A Client

`A2aClient` connects to remote A2A agents:

```ts
import { A2aClient } from "@rullama/a2a";
import { createUserMessage } from "@rullama/a2a";

const client = new A2aClient({ baseUrl: "https://agent.example.com" });

// Send a message
const response = await client.sendMessage({
  message: createUserMessage([{ type: "text", text: "Review this PR" }]),
});

// Get task status
const task = await client.getTask({ id: response.id });
```

## Task Lifecycle

Tasks progress through states: `submitted` -> `working` -> `completed` (or
`failed`, `canceled`).

```ts
import type { Task, TaskState, TaskStatus } from "@rullama/a2a";
```

The client supports task operations: `sendMessage`, `getTask`, `listTasks`,
`cancelTask`, `resubscribe`.

## SSE Streaming

Stream responses in real-time using Server-Sent Events:

```ts
import {
  isArtifactUpdate,
  isStatusUpdate,
  parseSseStream,
} from "@rullama/a2a";

// Stream a message send
const stream = await client.sendMessageStream({
  message: createUserMessage([{ type: "text", text: "Analyze this code" }]),
});

for await (const event of stream) {
  if (isStatusUpdate(event)) {
    console.log("Status:", event.status.state);
  } else if (isArtifactUpdate(event)) {
    console.log("Artifact:", event.artifact);
  }
}
```

Types: `StreamResponse`, `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`.

See: `../examples/a2a/a2a_streaming.ts`.

## Push Notifications

Configure webhook-based push notifications for task updates:

```ts
import type {
  AuthenticationInfo,
  TaskPushNotificationConfig,
} from "@rullama/a2a";
```

Methods: `setPushNotificationConfig`, `getPushNotificationConfig`,
`listPushNotificationConfigs`, `deletePushNotificationConfig`.

## JSON-RPC Methods

All methods are available as constants: `METHOD_MESSAGE_SEND`,
`METHOD_MESSAGE_STREAM`, `METHOD_TASKS_GET`, `METHOD_TASKS_LIST`,
`METHOD_TASKS_CANCEL`, `METHOD_TASKS_RESUBSCRIBE`, `METHOD_PUSH_CONFIG_SET`,
`METHOD_PUSH_CONFIG_GET`, `METHOD_PUSH_CONFIG_LIST`,
`METHOD_PUSH_CONFIG_DELETE`, `METHOD_EXTENDED_CARD`.

## Handler Interface

Implement `A2aHandler` to build your own A2A-compliant agent server:

```ts
import type { A2aHandler } from "@rullama/a2a";
```

See: `../examples/a2a/a2a_client_server.ts`.

## Further Reading

- [Networking](./networking.md) for the underlying MCP server and routing layers
- [Agents](./agents.md) for the agent runtime that drives A2A servers
