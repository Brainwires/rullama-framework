# Tools

The tool system lives in two packages: `@rullama/tool-runtime` (the tool
registry, executor, smart routing, transactions, sanitization, OpenAPI/OAuth,
and pre-execution hooks) and `@rullama/tool-builtins` (the concrete built-in
tool implementations: bash, file ops, git, web, search, calendar, sessions).

## ToolRegistry

The registry holds tool definitions and their execution handlers. Register
tools, then pass them to an `AgentContext`.

```ts
import { ToolRegistry } from "@rullama/tool-runtime";
import { BashTool, FileOpsTool, GitTool } from "@rullama/tool-builtins";

const registry = new ToolRegistry();
registry.registerTools(BashTool.getTools());
registry.registerTools(FileOpsTool.getTools());
registry.registerTools(GitTool.getTools());

const allTools = registry.allTools(); // Tool[] for the provider
const tool = registry.get("bash"); // single lookup
```

## Built-in Tools

| Tool Class       | Operations                                                                                              | Description                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `BashTool`       | `bash`                                                                                                  | Shell command execution with proactive output management |
| `FileOpsTool`    | `read_file`, `write_file`, `edit_file`, `list_files`, `search_files`, `delete_file`, `create_directory` | Full filesystem operations                               |
| `GitTool`        | `status`, `diff`, `log`, `stage`, `commit`, `push`, `pull`, etc.                                        | Git operations                                           |
| `WebTool`        | `web_fetch`                                                                                             | URL fetching with content extraction                     |
| `SearchTool`     | `search`                                                                                                | Regex-based code search (respects .gitignore)            |
| `ValidationTool` | `validate`                                                                                              | Content validation checks                                |

See: `../examples/tools/tool_registry.ts`,
`../examples/tools/tool_execution.ts`.

## Custom Tool Creation

Define a `Tool` with an input schema and implement execution via `ToolExecutor`:

```ts
import { type ToolExecutor } from "@rullama/tool-runtime";
import {
  objectSchema,
  type Tool,
  ToolResult,
  type ToolUse,
} from "@rullama/core";

const myTool: Tool = {
  name: "weather",
  description: "Get weather for a city",
  input_schema: objectSchema({
    city: { type: "string", description: "City name" },
  }, ["city"]),
};

const executor: ToolExecutor = {
  availableTools: () => [myTool],
  execute: async (toolUse: ToolUse) => {
    const city = toolUse.input.city;
    return ToolResult.success(toolUse.id, `Weather in ${city}: sunny, 22C`);
  },
};
```

## OpenAPI Tool Generation

Automatically generate tools from an OpenAPI spec:

```ts
import { executeOpenApiTool, openApiToTools } from "@rullama/tool-runtime";

const tools = openApiToTools(openApiSpec);
```

## Smart Routing

Smart routing analyzes the conversation and selects only relevant tools,
reducing token usage:

```ts
import { analyzeQuery, getSmartTools } from "@rullama/tool-runtime";

const relevantTools = getSmartTools(messages, allTools);
```

See: `../examples/tools/smart_routing.ts`.

## Transaction Manager

`TransactionManager` provides atomic multi-step tool operations with rollback:

```ts
import { TransactionManager } from "@rullama/tool-runtime";

const tx = new TransactionManager();
// Operations within the transaction can be committed or rolled back
```

See: `../examples/tools/tool_transactions.ts`.

## Pre-execution Hooks

Use `ToolPreHook` to gate or modify tool calls before execution:

```ts
import { allow, reject, type ToolPreHook } from "@rullama/tool-runtime";

const safetyHook: ToolPreHook = {
  beforeExecute: (toolUse) => {
    if (toolUse.name === "bash" && toolUse.input.command?.includes("rm -rf")) {
      return reject("Destructive command blocked");
    }
    return allow();
  },
};
```

## Content Sanitization

The package includes utilities for input/output sanitization:

- `containsSensitiveData` -- detect secrets in tool output
- `redactSensitiveData` -- redact detected secrets
- `isInjectionAttempt` -- detect prompt injection in external content
- `sanitizeExternalContent` -- wrap content with source metadata

See: `../examples/tools/tool_filtering.ts`.

## Further Reading

- [Agents](./agents.md) for using tools in agent loops
- [Extensibility](./extensibility.md) for custom tool executors
