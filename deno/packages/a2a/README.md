# @rullama/a2a

Agent-to-Agent (A2A) protocol implementation for the rullama.
This is one of the first Deno-native implementations of
[Google's A2A protocol](https://github.com/google/A2A), enabling standardized
inter-agent communication with JSON-RPC and REST transports.

Equivalent to the Rust `rullama-a2a` crate.

## Install

```sh
deno add @rullama/a2a
```

## Quick Example

```ts
import { A2aClient, createUserMessage } from "@rullama/a2a";
import type { AgentCard } from "@rullama/a2a";

// Discover an agent
const client = new A2aClient({ baseUrl: "http://localhost:8080" });
const card: AgentCard = await client.getAgentCard();
console.log(`Agent: ${card.name} — ${card.description}`);

// Send a message
const message = createUserMessage("Summarize this document for me.");
const response = await client.sendMessage({
  message,
  configuration: { blocking: true },
});
console.log(response);

// Stream responses
const stream = client.streamMessage({ message });
for await (const event of stream) {
  console.log("Event:", event);
}
```

## Creating an Agent Card

```ts
import type { AgentCard } from "@rullama/a2a";

const card: AgentCard = {
  name: "SummaryAgent",
  description: "Summarizes documents and web pages.",
  version: "1.0.0",
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

| Export                                     | Description                                                       |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `A2aClient`                                | Unified client supporting JSON-RPC and REST transports            |
| `AgentCard`                                | Self-describing agent manifest type                               |
| `A2aError`                                 | Typed error with standard A2A error codes                         |
| `createUserMessage` / `createAgentMessage` | Message factory helpers                                           |
| `parseSseStream`                           | Server-Sent Events parser for streaming                           |
| `A2aHandler`                               | Interface for implementing A2A agent servers                      |
| Streaming types                            | `StreamEvent`, `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent` |
