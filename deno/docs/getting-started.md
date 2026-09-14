# Getting Started

Get from zero to a running agent in about 5 minutes.

## Prerequisites

- [Deno](https://deno.land/) 2.x
- An API key for at least one AI provider (Anthropic, OpenAI, Google, or a local
  Ollama instance)

## 1. Install packages

```sh
deno add jsr:@rullama/core jsr:@rullama/provider jsr:@rullama/inference
deno add jsr:@rullama/agent jsr:@rullama/tool-builtins
```

(`@rullama/tool-runtime` and `@rullama/permission` come in transitively.)

## 2. Create a provider

Every interaction starts with a `Provider` -- the bridge between your code and
an AI model.

```ts
import { ChatOptions, Message } from "@rullama/core";
import { AnthropicChatProvider } from "@rullama/provider";

const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
);

const messages = [Message.user("What is the Deno runtime?")];
const options = new ChatOptions({ max_tokens: 1024 });
const response = await provider.chat(messages, undefined, options);
console.log(response.message.text());
```

## 3. Create a tool executor

Tools give agents the ability to interact with the world.
`createBuiltinExecutor()` gives you bash, file ops, git, web fetch and code
search behind the enforcing executor (policy engine, capability profile, output
filtering).

```ts
import { createBuiltinExecutor } from "@rullama/tool-builtins";

const executor = createBuiltinExecutor({ mode: "auto" });
console.log(executor.availableTools().map((t) => t.name));
```

## 4. Run an agent

Combine a provider, the executor and a `Task` in an `AgentContext`, then run a
`TaskAgent`. The context enforces permissions by default (pass `false` as the
sixth argument to opt out).

```ts
import { Task } from "@rullama/core";
import { CommunicationHub, FileLockManager } from "@rullama/agent";
import { AgentContext, TaskAgent } from "@rullama/inference";

const context = new AgentContext(
  Deno.cwd(),
  executor,
  new CommunicationHub(),
  new FileLockManager(),
);

const task = new Task("demo-task", "List the files in the current directory.");
const agent = new TaskAgent("demo-agent", task, provider, context, {
  systemPrompt: "You are a helpful coding assistant.",
});
const result = await agent.execute();

console.log(`Success: ${result.success}`);
console.log(`Summary: ${result.summary}`);
```

## 5. Connect to an MCP server (optional)

Use external tool servers via the Model Context Protocol.

```ts
import { McpClient } from "@rullama/mcp-client";

const client = McpClient.createDefault();
await client.connect({
  name: "my-server",
  command: "npx",
  args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
});

const tools = await client.listTools("my-server");
console.log("Available tools:", tools.map((t) => t.name));
```

## Next steps

- See the full [quickstart example](../examples/core/quickstart.ts) and
  [agent quickstart](../examples/core/agent_quickstart.ts)
- Learn the [architecture](./architecture.md)
- Explore [providers](./providers.md), [tools](./tools.md), and
  [agents](./agents.md)
