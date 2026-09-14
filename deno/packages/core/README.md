# @rullama/core

Foundation types, traits, and error handling for the rullama framework. This is
the zero-dependency base that every other `@rullama/*` package builds on.

Equivalent to the Rust `rullama-core` crate.

## Install

```sh
deno add jsr:@rullama/core
```

## Quick Example

```ts
import {
  ChatOptions,
  FrameworkError,
  Message,
  Task,
  ToolContext,
  ToolResult,
} from "@rullama/core";

// Conversation messages
const userMsg = Message.user("Explain Deno in one sentence.");
const systemMsg = Message.system("You are a concise assistant.");
console.log(userMsg.role, systemMsg.text());

// Chat options: builder API, presets, or a plain object
const options = ChatOptions.create()
  .setTemperature(0.3)
  .setMaxTokens(256)
  .setModel("claude-sonnet-4-20250514");
const routing = ChatOptions.deterministic(16);
const factual = new ChatOptions({ temperature: 0.1, max_tokens: 2048 });
console.log(options.model, routing.temperature, factual.max_tokens);

// Tasks and tool plumbing shared by the agent packages
const task = new Task("task-1", "Add error handling to the parser");
const ctx = new ToolContext({ working_directory: Deno.cwd() });
const result = ToolResult.success("call-1", "done");
console.log(task.description, ctx.working_directory, result.is_error);

// Typed errors
const err = FrameworkError.providerAuth("anthropic", "invalid API key");
console.log(err.kind.type); // "provider_auth"
```

## Key Exports

| Export                           | Kind              | Description                                                                                 |
| -------------------------------- | ----------------- | ------------------------------------------------------------------------------------------- |
| `Message`                        | class             | Conversation message with role and content blocks (`text()`)                                |
| `ChatOptions`                    | class             | Chat options (`temperature`, `max_tokens`, `system`, `model`…) with builder API and presets |
| `ChatResponse`                   | interface         | `{ message, usage, finish_reason? }` returned by a provider                                 |
| `Provider`                       | interface         | Base trait for AI providers (`name`, `chat`, `streamChat`)                                  |
| `Tool`                           | interface         | Tool definition (name, description, input schema)                                           |
| `ToolUse` / `ToolResult`         | interface / class | Tool call request and response                                                              |
| `ToolContext`                    | class             | Working directory + metadata passed to tool handlers                                        |
| `PermissionMode`                 | type              | `"read-only"` / `"auto"` / `"full"`                                                         |
| `Task`                           | class             | Task with priority, status, and metadata                                                    |
| `FrameworkError`                 | class             | Typed error with `FrameworkErrorKind` discriminant                                          |
| `LifecycleHook` / `HookRegistry` | interface / class | Lifecycle event interception                                                                |
| `WorkingSet`                     | class             | Token-budgeted file context window                                                          |
| `FileContextManager`             | class             | Chunked file injection with de-duplication                                                  |
| `SerializablePlan`               | class             | Plan with steps, budget, and metadata                                                       |
| `ContentSource`                  | type              | Content origin tracking for sanitization                                                    |
| `EmbeddingProvider`              | interface         | Vector embedding generation                                                                 |
| `VectorStore`                    | interface         | Vector similarity search                                                                    |
| `RateLimiter`                    | class             | Token-bucket limiter (requests per minute)                                                  |
| `extractConfidence`              | function          | CISC-style response confidence scoring                                                      |
| `PlatformPaths`                  | namespace         | XDG / platform data, cache and config directories                                           |
