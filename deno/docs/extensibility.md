# Extensibility

This guide covers how to extend the Brainwires framework by implementing key
interfaces. The framework is interface-driven: implement an interface, pass it
to the component, done.

## Key Interfaces

| Interface           | Package                 | Purpose                        |
| ------------------- | ----------------------- | ------------------------------ |
| `Provider`          | `@rullama/core`      | AI chat completion backend     |
| `EmbeddingProvider` | `@rullama/core`      | Text embedding generation      |
| `VectorStore`       | `@rullama/core`      | Embedding storage and search   |
| `StorageBackend`    | `@rullama/storage`   | Record persistence backend     |
| `VectorDatabase`    | `@rullama/storage`   | Storage + vector search        |
| `ToolExecutor`      | `@rullama/tools`     | Custom tool execution backend  |
| `ToolPreHook`       | `@rullama/tools`     | Pre-execution tool gate        |
| `AgentRuntime`      | `@rullama/agents`    | Custom agent execution loop    |
| `LifecycleHook`     | `@rullama/core`      | Framework event interception   |
| `OutputParser`      | `@rullama/core`      | Structured LLM output parsing  |
| `BrainClient`       | `@rullama/knowledge` | Knowledge storage interface    |
| `RagClient`         | `@rullama/knowledge` | Semantic code search interface |
| `Middleware`        | `@rullama/network`   | MCP server request processing  |
| `Discovery`         | `@rullama/network`   | Peer discovery protocol        |
| `A2aHandler`        | `@rullama/a2a`       | A2A agent server handler       |

## Custom Provider

Implement `Provider` from `@rullama/core`:

```ts
import type {
  ChatResponse,
  Provider,
  StreamChunk,
  Tool,
} from "@rullama/core";
import { ChatOptions, createUsage, Message } from "@rullama/core";

class MyProvider implements Provider {
  name(): string {
    return "my-provider";
  }

  async chat(
    messages: Message[],
    tools?: Tool[],
    options?: ChatOptions,
  ): Promise<ChatResponse> {
    const last = messages.findLast((m) => m.role === "user");
    const text = last?.text() ?? "";
    return {
      message: Message.assistant(`Response to: ${text}`),
      usage: createUsage(10, 20),
      finish_reason: "stop",
    };
  }

  async *streamChat(
    messages: Message[],
    tools?: Tool[],
    options?: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const resp = await this.chat(messages, tools, options);
    yield { type: "text", text: resp.message.text() ?? "" };
    yield { type: "done" };
  }
}
```

Use it anywhere a `Provider` is expected -- `spawnTaskAgent`, `runAgentLoop`,
etc.

## Custom Storage Backend

Implement `StorageBackend` from `@rullama/storage`:

```ts
import type {
  FieldDef,
  Filter,
  Record,
  StorageBackend,
} from "@rullama/storage";

class RedisBackend implements StorageBackend {
  async createTable(name: string, fields: FieldDef[]): Promise<void> {/* ... */}
  async insert(table: string, record: Record): Promise<void> {/* ... */}
  async get(table: string, id: string): Promise<Record | null> {/* ... */}
  async update(table: string, id: string, record: Record): Promise<void> {
    /* ... */
  }
  async delete(table: string, id: string): Promise<void> {/* ... */}
  async query(table: string, filter: Filter): Promise<Record[]> {/* ... */}
  async list(
    table: string,
    limit?: number,
    offset?: number,
  ): Promise<Record[]> {/* ... */}
}
```

Pass it to any domain store: `new MessageStore(new RedisBackend())`.

## Custom Tools

Implement `ToolExecutor` from `@rullama/tools`:

```ts
import type { ToolExecutor } from "@rullama/tools";
import {
  objectSchema,
  type Tool,
  ToolResult,
  type ToolUse,
} from "@rullama/core";

const databaseTool: Tool = {
  name: "query_db",
  description: "Run a SQL query",
  input_schema: objectSchema({ sql: { type: "string" } }, ["sql"]),
};

class DatabaseExecutor implements ToolExecutor {
  availableTools(): Tool[] {
    return [databaseTool];
  }

  async execute(toolUse: ToolUse): Promise<ToolResult> {
    const result = await runQuery(toolUse.input.sql);
    return ToolResult.success(toolUse.id, JSON.stringify(result));
  }
}
```

## Custom Agent Runtime

Implement `AgentRuntime` for full control over the agent loop:

```ts
import type { AgentExecutionResult, AgentRuntime } from "@rullama/agent";
import { runAgentLoop } from "@rullama/agent";

class MyRuntime implements AgentRuntime {
  agentId(): string {
    return "custom-agent";
  }
  maxIterations(): number {
    return 20;
  }
  async callProvider(): Promise<Message> {/* ... */}
  extractToolUses(msg: Message): ToolUse[] {/* ... */}
  isCompletion(msg: Message): boolean {/* ... */}
  async executeTool(toolUse: ToolUse): Promise<ToolResult> {/* ... */}
  // ... remaining lifecycle methods
}

const result = await runAgentLoop(new MyRuntime(), hub, lockManager);
```

## Custom Middleware

Implement `Middleware` for the MCP server pipeline:

```ts
import {
  type Middleware,
  middlewareContinue,
  middlewareReject,
  type MiddlewareResult,
  RequestContext,
} from "@rullama/network";

class MetricsMiddleware implements Middleware {
  async process(ctx: RequestContext): Promise<MiddlewareResult> {
    const start = performance.now();
    // Middleware runs before the handler; return continue to proceed
    console.log(`Request: ${ctx.method} (${performance.now() - start}ms)`);
    return middlewareContinue();
  }
}
```

## Custom Lifecycle Hooks

Intercept framework events with `LifecycleHook`:

```ts
import type {
  HookResult,
  LifecycleEvent,
  LifecycleHook,
} from "@rullama/core";

const loggingHook: LifecycleHook = {
  name: () => "logging",
  priority: () => 10,
  filter: () => ({ eventTypes: ["tool_start", "tool_end"] }),
  onEvent: async (event: LifecycleEvent): Promise<HookResult> => {
    console.log(`[${event.type}] ${event.agentId}: ${event.toolName}`);
    return { proceed: true };
  },
};
```

## Error Handling

Use `FrameworkError` for domain-specific errors:

```ts
import { FrameworkError } from "@rullama/core";

throw FrameworkError.providerAuth("my-provider", "Invalid API key");
throw FrameworkError.storageSchema("my-store", "Missing table");
```

## Where to Define Extensions

- **Types and interfaces** -- `@rullama/core`
- **Tool implementations** -- `@rullama/tools`
- **Agent coordination** -- `@rullama/agents`
- **Storage backends** -- `@rullama/storage`
- **Network components** -- `@rullama/network`

## Further Reading

- [Architecture](./architecture.md) for the package dependency graph
- [Providers](./providers.md) for the built-in provider implementations
- [Tools](./tools.md) for built-in tool examples
- [Storage](./storage.md) for built-in storage backends
