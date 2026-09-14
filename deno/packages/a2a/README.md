# @rullama/a2a

Agent-to-Agent (A2A) protocol **client** for the rullama — a Deno-native
implementation of [Google's A2A protocol](https://github.com/google/A2A) for
inter-agent communication, speaking either the JSON-RPC or the REST binding over
`fetch()`.

This package is client-only. It ships `A2aClient`, the full A2A v1.0 type system
(messages, tasks, agent cards, security schemes, streaming events), the error
codes and an SSE parser. The `A2aHandler` interface describes what an agent
server would implement, but **no server implementation** (HTTP router, JSON-RPC
dispatcher, task store) is included — bring your own or wait for a later
release. gRPC is not supported.

Equivalent to the Rust `rullama-a2a` crate.

> **Spec alignment in 0.13.** The JSON-RPC method names currently sent by the
> client are the v1.0 RPC-style names (`SendMessage`, `SendStreamingMessage`,
> `GetTask`, `CancelTask`, `SubscribeToTask`, `ListTasks`, …, see the `METHOD_*`
> constants) and `TaskState` uses the SCREAMING_SNAKE_CASE enum values
> (`TASK_STATE_WORKING`, `TASK_STATE_COMPLETED`, …). Both are being aligned to
> the published A2A specification in 0.13; expect them to change.

## Install

```sh
deno add @rullama/a2a
```

## Quick Example

```ts
import {
  A2aClient,
  createUserMessage,
  isStatusUpdate,
  isTaskResponse,
} from "@rullama/a2a";
import type { AgentCard } from "@rullama/a2a";

// Discover an agent via GET {baseUrl}/.well-known/agent-card.json
const card: AgentCard = await A2aClient.discover("http://localhost:8080");
console.log(`Agent: ${card.name} — ${card.description}`);

// Talk to it over JSON-RPC (default) or `transport: "rest"`
const client = new A2aClient({
  baseUrl: "http://localhost:8080",
  transport: "jsonrpc",
});

// Send a message and wait for the terminal task state
const message = createUserMessage("Summarize this document for me.");
const response = await client.sendMessage({
  message,
  configuration: { returnImmediately: false },
});
if (response.task) {
  console.log("task", response.task.id, response.task.status.state);
} else if (response.message) {
  console.log("reply", response.message.parts);
}

// Stream status/artifact updates over Server-Sent Events
for await (const event of client.streamMessage({ message })) {
  if (isTaskResponse(event)) console.log("task:", event.task?.id);
  if (isStatusUpdate(event)) console.log("status:", event.statusUpdate?.status);
}
```

## Creating an Agent Card

`supportedInterfaces` is required: it lists where the agent is reachable and
which protocol binding each URL speaks.

```ts
import type { AgentCard } from "@rullama/a2a";

const card: AgentCard = {
  name: "SummaryAgent",
  description: "Summarizes documents and web pages.",
  version: "1.0.0",
  supportedInterfaces: [
    {
      url: "http://localhost:8080",
      protocolBinding: "JSONRPC",
      protocolVersion: "1.0",
    },
  ],
  capabilities: { streaming: true, pushNotifications: false },
  skills: [
    {
      id: "summarize",
      name: "Summarize",
      description: "Produce a concise summary of text input.",
      tags: ["nlp", "summarization"],
      examples: ["Summarize this article."],
    },
  ],
  defaultInputModes: ["text/plain"],
  defaultOutputModes: ["text/plain"],
};
```

## Key Exports

| Export                                     | Description                                                                                                                                                                 |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `A2aClient`                                | Client for one agent: `discover`, `sendMessage`, `streamMessage`, `getTask`, `listTasks`, `cancelTask`, `subscribeToTask`, push-config CRUD, `getAuthenticatedExtendedCard` |
| `A2aClientOptions` / `Transport`           | `baseUrl`, `transport: "jsonrpc" \| "rest"`, optional `bearerToken`                                                                                                         |
| `AgentCard` + security-scheme types        | Self-describing agent manifest and its `SecurityScheme` / `OAuthFlows` variants                                                                                             |
| `A2aError` + `*_ERROR` / `TASK_*` codes    | Typed error carrying the JSON-RPC and A2A-specific error codes                                                                                                              |
| `createUserMessage` / `createAgentMessage` | Message factory helpers (one text part, random `messageId`)                                                                                                                 |
| `METHOD_*`                                 | JSON-RPC method-name constants                                                                                                                                              |
| `parseSseStream`                           | Server-Sent Events parser yielding `StreamResponse` values                                                                                                                  |
| `A2aHandler`                               | Interface an agent server would implement (no server ships in this package)                                                                                                 |
| Streaming types                            | `StreamResponse`, `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent` + type guards                                                                                          |
