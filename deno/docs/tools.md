# Tools

The tool system lives in two packages:

- `@rullama/tool-runtime` -- the framework: `ToolRegistry`, the `ToolExecutor` /
  `ToolPreHook` interfaces, the **enforcing executor** (`EnforcingExecutor` /
  `enforce()`), sanitization, the shared guards (`confinePath`,
  `compileBoundedRegex`, `safeFetch`), smart routing, `TransactionManager`,
  OpenAPI tool generation, OAuth, tool search and `ValidationTool`.
- `@rullama/tool-builtins` -- the concrete built-in tools (bash, file ops, git,
  web, search, semantic search, calendar, sessions) and `BuiltinToolExecutor` /
  `createBuiltinExecutor()`.

## The default executor

`createBuiltinExecutor()` is the one-liner most applications want: the default
tool set (`BashTool`, `FileOpsTool`, `GitTool`, `SearchTool`, `WebTool`) behind
the enforcing executor. Its options are `EnforcementOptions` plus `providers` to
swap the tool set.

```ts
import { ToolContext } from "@rullama/core";
import { createBuiltinExecutor } from "@rullama/tool-builtins";

const executor = createBuiltinExecutor({ mode: "auto" });
const tools = executor.availableTools(); // Tool[] for the provider

const result = await executor.execute(
  { id: "call-1", name: "read_file", input: { path: "README.md" } },
  new ToolContext({ working_directory: Deno.cwd() }),
);
console.log(result.is_error, result.content);
```

`BuiltinToolExecutor` (unwrapped) dispatches by tool name to any `ToolProvider`
-- an object with static-style `getTools()` and
`execute(toolUseId, toolName, input, context)`; the built-in classes satisfy it
and `DEFAULT_TOOL_PROVIDERS` lists the default set.

## Enforcement

Until v0.12.0 every `Tool.requires_approval` flag, policy rule and sanitizer was
advisory. `EnforcingExecutor` wraps any `ToolExecutor` and applies them, in this
order, before and after each call:

1. **Permission mode** (`PermissionMode` from core: `"read-only"`, `"auto"`,
   `"full"`) -- `"read-only"` admits only read/search/planning tools.
2. **Capabilities** -- an optional `AgentCapabilities` profile narrows tools,
   file paths, domains and git operations.
3. **Policy** -- the `PolicyEngine` decides allow / deny / require approval
   (default `PolicyEngine.withDefaults()`, which denies `.env`, secret and
   credential files and asks before `git reset` / `git rebase`). A
   `require_approval` decision -- and, in `"auto"` mode, a tool flagged
   `requires_approval` -- goes to the `ApprovalHandler`; without one it is
   denied.
4. **Pre-execute hooks** -- every `ToolPreHook` may still `reject()`.
5. **Output filtering** -- secrets are redacted from every result and the output
   of web-fetching tools (`DEFAULT_EXTERNAL_CONTENT_TOOLS`) is wrapped as
   untrusted external content.

```ts
import { AgentCapabilities, PolicyEngine } from "@rullama/permission";
import {
  type ApprovalHandler,
  enforce,
  type EnforcementEvent,
  type ToolExecutor,
} from "@rullama/tool-runtime";

const approve: ApprovalHandler = (toolUse, decision) => {
  console.log(`approval requested for ${toolUse.name}: ${decision.reason}`);
  return false; // or prompt a human
};

const enforcing = enforce(myExecutor as ToolExecutor, {
  mode: "auto",
  capabilities: AgentCapabilities.standardDev(),
  policy: PolicyEngine.withDefaults(),
  approve,
  preHooks: [safetyHook],
  filterOutput: true,
  onDecision: (event: EnforcementEvent) => audit.push(event),
});
```

`enforce()` is idempotent (an already-enforcing executor is returned as-is).
`@rullama/inference`'s `AgentContext` wraps the executor it is given with this
class **by default**; pass `enforcement: false` as its sixth argument to opt
out. Every decision is reported to `onDecision` as an `EnforcementEvent`
(`outcome`: `allowed` / `denied` / `approved` / `rejected`).

Helpers for building policy requests from a tool call are exported too:
`policyRequestForToolUse`, `filePathForToolUse`, `domainForToolUse`,
`gitOperationForToolUse`.

## Built-in tool safety limits

The built-in tools enforce their own limits regardless of the executor:

| Tool                 | Limit                                                                                                                                                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `execute_command`    | Timeout (default 30 s) kills the whole process tree; child env scrubbed of credential-like variables (`scrubEnv` / `SECRET_ENV_PATTERN`); stdout/stderr capped at `MAX_OUTPUT_BYTES` (256 KiB); destructive patterns blocked |
| file ops             | Every path confined to the working directory (`confinePath` follows symlinks, `confinePathLexical` for the rest); reads capped at `MAX_READ_BYTES` (1 MiB)                                                                   |
| `search_code`        | Regex compiled with `compileBoundedRegex` (`MAX_REGEX_LENGTH` = 200); files over `MAX_SEARCH_FILE_BYTES` skipped; at most 100 matches                                                                                        |
| `fetch_url`          | `safeFetch`: http/https only, loopback / link-local / private / CGNAT / multicast addresses refused (also on every redirect), 30 s deadline, body capped by `readCappedText` (`DEFAULT_MAX_BODY_BYTES` = 1 MiB)              |
| git tools            | Refs/branches/remotes validated by `assertSafeRef` (`SAFE_REF_PATTERN`), file lists by `assertSafeFiles` -- arguments cannot become options or remote helpers                                                                |
| `TransactionManager` | Target paths confined to the project root; staged names via `safeFileName`                                                                                                                                                   |

`safeFetch(url, { allowPrivate, maxRedirects, timeoutMs })` and `checkUrl` are
exported for your own tools.

## ToolRegistry

The registry holds tool definitions (not handlers) for a provider: register
them, filter by category, search, and split the initial set from deferred tools.

```ts
import { ToolRegistry } from "@rullama/tool-runtime";
import { BashTool, FileOpsTool, GitTool } from "@rullama/tool-builtins";

const registry = new ToolRegistry();
registry.registerTools(BashTool.getTools());
registry.registerTools(FileOpsTool.getTools());
registry.registerTools(GitTool.getTools());

const allTools = registry.getAll(); // readonly Tool[]
const tool = registry.get("execute_command"); // single lookup
const gitTools = registry.getByCategory("Git");
```

## Built-in Tools

| Tool Class           | Tool names                                                                                                                                      | Notes                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `BashTool`           | `execute_command`                                                                                                                               | Shell via `Deno.Command`; output modes                 |
| `FileOpsTool`        | `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `delete_file`, `create_directory`                                     | Deno FS APIs                                           |
| `GitTool`            | `git_status`, `git_diff`, `git_log`, `git_stage`, `git_unstage`, `git_commit`, `git_push`, `git_pull`, `git_fetch`, `git_branch`, `git_discard` | Git subprocess                                         |
| `WebTool`            | `fetch_url`                                                                                                                                     | `safeFetch`                                            |
| `SearchTool`         | `search_code`                                                                                                                                   | Regex search, respects `.gitignore`                    |
| `SemanticSearchTool` | `index_codebase`, `query_codebase`, `search_with_filters`, `search_git_history`, `get_rag_statistics`, `clear_rag_index`                        | Needs an injected `RagClient` (`@rullama/rag`)         |
| `CalendarTool`       | `calendar_list_events`, `calendar_create_event`, `calendar_update_event`, `calendar_delete_event`, `calendar_find_free_time`                    | Google Calendar / CalDAV                               |
| `SessionsTool`       | `sessions_list`, `sessions_history`, `sessions_send`, `sessions_spawn`                                                                          | Instance-based; needs a `SessionBroker`                |
| `ValidationTool`     | `check_duplicates`, `verify_build`, `check_syntax`                                                                                              | In `@rullama/tool-runtime`                             |
| `ToolSearchTool`     | `search_tools`                                                                                                                                  | In `@rullama/tool-runtime`; keyword / regex / semantic |

See: `../examples/tools/tool_registry.ts`,
`../examples/tools/tool_execution.ts`.

## Custom Tool Creation

Define a `Tool` with an input schema and implement execution via `ToolExecutor`,
then wrap it with `enforce()`:

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
  execute: (toolUse: ToolUse) => {
    const city = toolUse.input.city;
    return Promise.resolve(
      ToolResult.success(toolUse.id, `Weather in ${city}: sunny, 22C`),
    );
  },
};
```

## OpenAPI Tool Generation

```ts
import { executeOpenApiTool, openApiToTools } from "@rullama/tool-runtime";

const tools = openApiToTools(openApiSpec); // one Tool per operation
```

## Smart Routing

Smart routing analyzes the conversation and selects only the relevant tool
categories, reducing token usage. It is keyword based -- no model call.

```ts
import { analyzeQuery, getSmartTools } from "@rullama/tool-runtime";

const categories = analyzeQuery("run the tests and commit"); // ToolCategory[]
const relevantTools = getSmartTools(messages, registry);
```

See: `../examples/tools/smart_routing.ts`.

## Transaction Manager

`TransactionManager` implements `@rullama/core`'s `StagingBackend`: stage file
writes, then commit them all or roll back.

See: `../examples/tools/tool_transactions.ts`.

## Pre-execution Hooks

Use `ToolPreHook` to gate tool calls before execution; the enforcing executor
runs them after the policy check.

```ts
import { allow, reject, type ToolPreHook } from "@rullama/tool-runtime";

const safetyHook: ToolPreHook = {
  beforeExecute: (toolUse) => {
    if (
      toolUse.name === "execute_command" &&
      String(toolUse.input.command ?? "").includes("rm -rf")
    ) {
      return Promise.resolve(reject("Destructive command blocked"));
    }
    return Promise.resolve(allow());
  },
};
```

## Content Sanitization

- `containsSensitiveData` / `redactSensitiveData` -- detect and redact secrets
- `isInjectionAttempt` -- detect prompt injection in external content
- `sanitizeExternalContent` / `wrapWithContentSource` -- wrap content with
  source metadata
- `filterToolOutput` -- what the enforcing executor applies to every result

See: `../examples/tools/tool_filtering.ts`.

## Further Reading

- [Permissions](./permissions.md) for capability profiles and the policy engine
- [Agents](./agents.md) for using tools in agent loops
- [Extensibility](./extensibility.md) for custom tool executors
