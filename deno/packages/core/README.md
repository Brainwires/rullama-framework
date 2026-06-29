# @rullama/core

Foundation types, traits, and error handling for the Brainwires Agent Framework.
This is the zero-dependency base that every other `@rullama/*` package builds
on.

Equivalent to the Rust `rullama-core` crate.

## Install

```sh
deno add @rullama/core
```

## Quick Example

```ts
import { ChatOptions, FrameworkError, Message } from "@rullama/core";

// Create messages
const userMsg = Message.user("Explain Deno in one sentence.");
const systemMsg = Message.system("You are a concise assistant.");

// Configure chat options with the builder API
const options = ChatOptions.new()
  .setTemperature(0.3)
  .setMaxTokens(256)
  .setSystem("You are a concise assistant.");

// Use preset options for common patterns
const routing = ChatOptions.deterministic(16);
const factual = ChatOptions.factual(2048);
```

## Key Exports

| Export              | Kind      | Description                                           |
| ------------------- | --------- | ----------------------------------------------------- |
| `Message`           | class     | Conversation message with role, content, and metadata |
| `ChatOptions`       | class     | Chat completion options with builder API and presets  |
| `Provider`          | interface | Base trait for AI providers (`chat`, `streamChat`)    |
| `Tool`              | interface | Tool definition (name, description, input schema)     |
| `ToolResult`        | class     | Result returned from tool execution                   |
| `ToolContext`       | class     | Execution context passed to tool handlers             |
| `Task`              | class     | Task with priority, status, and metadata              |
| `FrameworkError`    | class     | Typed error with `FrameworkErrorKind` discriminant    |
| `HookRegistry`      | class     | Registry for lifecycle event hooks                    |
| `WorkingSet`        | class     | Token-budgeted file context window                    |
| `SerializablePlan`  | class     | Plan with steps, budget, and metadata                 |
| `ContentSource`     | type      | Content origin tracking for sanitization              |
| `EmbeddingProvider` | interface | Vector embedding generation                           |
| `VectorStore`       | interface | Vector similarity search                              |
