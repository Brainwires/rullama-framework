# A2A Protocol

The `@rullama/a2a` package implements the **client side** of Google's
Agent-to-Agent (A2A) protocol v1.0 over JSON-RPC and REST transports (no gRPC).
It ships the full A2A type system, the JSON-RPC method-name constants,
`A2aError` with the spec error codes and an SSE parser. `A2aHandler` is the
interface an agent server would implement, but no server transport or router
ships in this package -- build one on `@rullama/mcp-server`-style transports or
an HTTP server of your choice.

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
import type { AgentCard } from "@rullama/a2a";

const card: AgentCard = {
  name: "code-reviewer",
  description: "Reviews code for quality and security issues",
  version: "1.0.0",
  supportedInterfaces: [],
  capabilities: { streaming: true, pushNotifications: false },
  skills: [
    {
      id: "review",
      name: "Code Review",
      description: "Reviews pull requests",
      tags: ["code", "review"],
    },
  ],
  defaultInputModes: ["text/plain"],
  defaultOutputModes: ["text/plain"],
};
```

See: `../examples/a2a/agent_card.ts`.

## A2A Client

`A2aClient` connects to remote A2A agents. `A2aClientOptions` takes `baseUrl`,
an optional `transport` (`"jsonrpc"` default, or `"rest"`) and an optional
`bearerToken` (or call `withBearerToken`).

```ts
import { A2aClient, createUserMessage } from "@rullama/a2a";

const client = new A2aClient({ baseUrl: "https://agent.example.com" })
  .withBearerToken(Deno.env.get("A2A_TOKEN")!);

// Send a message
const response = await client.sendMessage({
  message: createUserMessage("Review this PR"),
});

// Get task status (the response carries a Task snapshot or a Message)
if (response.task) {
  const task = await client.getTask({ id: response.task.id });
  console.log(task.status);
}
```

## Task Lifecycle

Tasks progress through states: `submitted` -> `working` -> `completed` (or
`failed`, `canceled`).

```ts
import type { Task, TaskState, TaskStatus } from "@rullama/a2a";
```

Client task operations: `sendMessage`, `streamMessage`, `getTask`, `listTasks`,
`cancelTask`, `subscribeToTask`.

## SSE Streaming

`streamMessage` and `subscribeToTask` are async iterables of `StreamResponse`
(over JSON-RPC or the REST `:stream` endpoints); `parseSseStream` is exported
for raw streams.

```ts
import {
  createUserMessage,
  isArtifactUpdate,
  isStatusUpdate,
} from "@rullama/a2a";

for await (
  const event of client.streamMessage({
    message: createUserMessage("Analyze this code"),
  })
) {
  if (isStatusUpdate(event)) {
    console.log("status update", event);
  } else if (isArtifactUpdate(event)) {
    console.log("artifact update", event);
  }
}
```

Types: `StreamResponse`, `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`;
guards: `isStatusUpdate`, `isArtifactUpdate`, `isTaskResponse`,
`isMessageResponse`.

See: `../examples/a2a/a2a_streaming.ts`.

## Push Notifications

Configure webhook-based push notifications for task updates:

```ts
import type {
  AuthenticationInfo,
  TaskPushNotificationConfig,
} from "@rullama/a2a";
```

Client methods: `setPushConfig`, `getPushConfig`, `listPushConfigs`,
`deletePushConfig`; `getAuthenticatedExtendedCard` fetches the extended card.

## JSON-RPC Methods

All methods are available as constants: `METHOD_MESSAGE_SEND`,
`METHOD_MESSAGE_STREAM`, `METHOD_TASKS_GET`, `METHOD_TASKS_LIST`,
`METHOD_TASKS_CANCEL`, `METHOD_TASKS_RESUBSCRIBE`, `METHOD_PUSH_CONFIG_SET`,
`METHOD_PUSH_CONFIG_GET`, `METHOD_PUSH_CONFIG_LIST`,
`METHOD_PUSH_CONFIG_DELETE`, `METHOD_EXTENDED_CARD`. Error codes:
`TASK_NOT_FOUND`, `TASK_NOT_CANCELABLE`, `PUSH_NOT_SUPPORTED`, … via `A2aError`.

## Handler Interface

Implement `A2aHandler` to build your own A2A-compliant agent server on top of
any HTTP framework:

```ts
import type { A2aHandler } from "@rullama/a2a";
```

See: `../examples/a2a/a2a_client_server.ts`.

## Further Reading

- [Networking](./networking.md) for the MCP server and the relay / routing layer
- [Agents](./agents.md) for the agent runtime behind an A2A server
