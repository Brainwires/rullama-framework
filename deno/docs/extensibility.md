# Extensibility

This guide covers how to extend the rullama framework by implementing key
interfaces. The framework is interface-driven: implement an interface, pass it
to the component, done.

## Key Interfaces

| Interface             | Package                  | Purpose                                              |
| --------------------- | ------------------------ | ---------------------------------------------------- |
| `Provider`            | `@rullama/core`          | AI chat completion backend                           |
| `EmbeddingProvider`   | `@rullama/core`          | Text embedding generation                            |
| `VectorStore`         | `@rullama/core`          | Embedding storage and search                         |
| `LifecycleHook`       | `@rullama/core`          | Framework event interception                         |
| `OutputParser`        | `@rullama/core`          | Structured LLM output parsing                        |
| `StorageBackend`      | `@rullama/storage`       | Typed-table persistence backend                      |
| `VectorDatabase`      | `@rullama/storage`       | RAG embedding store with hybrid search               |
| `ToolExecutor`        | `@rullama/tool-runtime`  | Custom tool execution backend                        |
| `ToolPreHook`         | `@rullama/tool-runtime`  | Pre-execution tool gate                              |
| `ApprovalHandler`     | `@rullama/tool-runtime`  | Human-in-the-loop approval for tool calls            |
| `ToolProvider`        | `@rullama/tool-builtins` | A tool class the built-in executor can dispatch to   |
| `AgentRuntime`        | `@rullama/inference`     | Custom agent execution loop                          |
| `AgentLifecycleHooks` | `@rullama/inference`     | Per-iteration / per-tool loop control                |
| `BrainClient`         | `@rullama/knowledge`     | Knowledge storage interface                          |
| `RagClient`           | `@rullama/rag`           | Semantic code search interface                       |
| `Middleware`          | `@rullama/mcp-server`    | MCP server request processing                        |
| `ServerTransport`     | `@rullama/mcp-server`    | MCP server I/O (stdio ships)                         |
| `Discovery`           | `@rullama/network`       | Peer discovery protocol                              |
| `A2aHandler`          | `@rullama/a2a`           | A2A agent server handler (no server transport ships) |
| `AnalyticsSink`       | `@rullama/telemetry`     | Analytics event destination                          |
| `CacheBackend`        | `@rullama/call-policy`   | Response cache for `CachedProvider`                  |

## Custom Provider

Implement `Provider` from `@rullama/core`. `name` is a property, and the three
`chat` / `streamChat` arguments are positional (tools may be `undefined`):

```ts
import type {
  ChatOptions,
  ChatResponse,
  Provider,
  StreamChunk,
  Tool,
} from "@rullama/core";
import { createUsage, Message } from "@rullama/core";

class MyProvider implements Provider {
  readonly name = "my-provider";

  chat(
    messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): Promise<ChatResponse> {
    const last = messages.findLast((m) => m.role === "user");
    const text = last?.text() ?? "";
    return Promise.resolve({
      message: Message.assistant(`Response to: ${text}`),
      usage: createUsage(10, 20),
      finish_reason: "stop",
    });
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const resp = await this.chat(messages, tools, options);
    yield { type: "text", text: resp.message.text() ?? "" };
    yield { type: "done" };
  }
}
```

Use it anywhere a `Provider` is expected -- `TaskAgent`, `AgentPool`, the
`@rullama/call-policy` decorators, etc.

## Custom Storage Backend

Implement `StorageBackend` from `@rullama/storage` (typed tables, structured
`Filter`s, vector search):

```ts
import type {
  FieldDef,
  Filter,
  Record,
  ScoredRecord,
  StorageBackend,
} from "@rullama/storage";

class RedisBackend implements StorageBackend {
  ensureTable(_table: string, _schema: FieldDef[]): Promise<void> {
    return Promise.resolve();
  }
  insert(_table: string, _records: Record[]): Promise<void> {
    return Promise.resolve();
  }
  query(_table: string, _filter?: Filter, _limit?: number): Promise<Record[]> {
    return Promise.resolve([]);
  }
  delete(_table: string, _filter: Filter): Promise<void> {
    return Promise.resolve();
  }
  count(_table: string, _filter?: Filter): Promise<number> {
    return Promise.resolve(0);
  }
  vectorSearch(
    _table: string,
    _vectorColumn: string,
    _vector: number[],
    _limit: number,
    _filter?: Filter,
  ): Promise<ScoredRecord[]> {
    return Promise.resolve([]);
  }
}
```

Pass it to any domain store from `@rullama/stores`, e.g.
`new MessageStore(new RedisBackend(), embeddings)`. Validate table and column
names before interpolating them into a query language (the shipped SQL backends
accept only `^[A-Za-z_][A-Za-z0-9_]{0,62}$`) and refuse `Raw` filters unless the
caller opted in.

## Custom Tools

Implement `ToolExecutor` from `@rullama/tool-runtime`, then wrap it with
`enforce()` (or hand it to `AgentContext`, which does so by default):

```ts
import type { ToolContext } from "@rullama/core";
import { enforce, type ToolExecutor } from "@rullama/tool-runtime";
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

  async execute(toolUse: ToolUse, _context: ToolContext): Promise<ToolResult> {
    const result = await runQuery(toolUse.input.sql);
    return ToolResult.success(toolUse.id, JSON.stringify(result));
  }
}

const executor = enforce(new DatabaseExecutor(), { mode: "auto" });
```

To add a tool class to the built-in executor instead, implement `ToolProvider`
(`getTools()` + `execute(toolUseId, toolName, input, context)`) and pass it in
`createBuiltinExecutor({ providers: [...DEFAULT_TOOL_PROVIDERS, MyTools] })`.

## Custom Agent Runtime

Implement `AgentRuntime` from `@rullama/inference` for full control over the
agent loop:

```ts
import type { ChatResponse, ToolResult, ToolUse } from "@rullama/core";
import type { LockType } from "@rullama/agent";
import { type AgentRuntime, runAgentLoop } from "@rullama/inference";

class MyRuntime implements AgentRuntime {
  agentId(): string {
    return "custom-agent";
  }
  maxIterations(): number {
    return 20;
  }
  callProvider(): Promise<ChatResponse> {/* ... */}
  extractToolUses(response: ChatResponse): ToolUse[] {/* ... */}
  isCompletion(response: ChatResponse): boolean {/* ... */}
  executeTool(toolUse: ToolUse): Promise<ToolResult> {/* ... */}
  getLockRequirement(toolUse: ToolUse): [string, LockType] | undefined {
    /* ... */
  }
  onProviderResponse(response: ChatResponse): void {/* ... */}
  onToolResult(toolUse: ToolUse, result: ToolResult): void {/* ... */}
  onCompletion(response: ChatResponse): Promise<string | undefined> {/* ... */}
  onIterationLimit(iterations: number): string {/* ... */}
}

const result = await runAgentLoop(new MyRuntime(), hub, lockManager);
```

## Custom Middleware

Implement `Middleware` from `@rullama/mcp-server` for the MCP server pipeline:

```ts
import type { JsonRpcRequest } from "@rullama/mcp-client";
import {
  type Middleware,
  middlewareContinue,
  type MiddlewareResult,
  type RequestContext,
} from "@rullama/mcp-server";

class MetricsMiddleware implements Middleware {
  processRequest(
    request: JsonRpcRequest,
    _ctx: RequestContext,
  ): Promise<MiddlewareResult> {
    console.error(`Request: ${request.method}`);
    return Promise.resolve(middlewareContinue());
  }
}
```

Return `middlewareReject(jsonRpcError)` to short-circuit; implement the optional
`processResponse(response, ctx)` to observe replies.

## Custom Lifecycle Hooks

Intercept framework events with `LifecycleHook` from `@rullama/core` (`name` is
a property; `priority` / `filter` are optional methods):

```ts
import type { HookResult, LifecycleEvent, LifecycleHook } from "@rullama/core";

const loggingHook: LifecycleHook = {
  name: "logging",
  priority: () => 10,
  onEvent: (event: LifecycleEvent): Promise<HookResult> => {
    console.log(`[${event.type}]`);
    return Promise.resolve({ type: "continue" });
  },
};
```

`HookResult` is `{ type: "continue" }`, `{ type: "cancel"; reason }` or
`{ type: "modified"; data }`.

## Error Handling

Use `FrameworkError` for domain-specific errors:

```ts
import { FrameworkError } from "@rullama/core";

throw FrameworkError.providerAuth("my-provider", "Invalid API key");
throw FrameworkError.storageSchema("my-store", "Missing table");
```

## Where to Define Extensions

- **Types and interfaces** -- `@rullama/core`
- **Tool implementations** -- `@rullama/tool-builtins` (runtime and enforcement
  in `@rullama/tool-runtime`)
- **Agent coordination** -- `@rullama/agent` (runtime in `@rullama/inference`)
- **Storage backends** -- `@rullama/storage` (domain stores in
  `@rullama/stores`, tiered memory in `@rullama/memory`)
- **Network components** -- `@rullama/network`, `@rullama/mcp-server`

## Further Reading

- [Architecture](./architecture.md) for the package dependency graph
- [Providers](./providers.md) for the built-in provider implementations
- [Tools](./tools.md) for built-in tool examples and enforcement
- [Storage](./storage.md) for built-in storage backends
