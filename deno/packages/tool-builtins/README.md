# @rullama/tool-builtins

Built-in tool implementations -- `BashTool`, `FileOpsTool`, `GitTool`,
`WebTool`, `SearchTool`, `SemanticSearchTool`, `CalendarTool`, `SessionsTool` --
plus `BuiltinToolExecutor` and `createBuiltinExecutor()`, the default
`ToolExecutor` most applications hand to `AgentContext`.

Extracted from the old `@rullama/tools` package in v0.11.0 to mirror Rust's
`rullama-tool-builtins` crate. The execution framework (`ToolRegistry`,
`ToolExecutor`, `EnforcingExecutor`, sanitization, smart router, transaction
manager) lives in `@rullama/tool-runtime`, which this package depends on.

Native-only tools (`code_exec` / `interpreters`, `sandbox_executor`, `browser`,
`email`, `system`) are intentionally Rust-only -- see `SKIPPED.md`.

## Safety limits (v0.12.0)

- `execute_command`: timeout (default 30 s) kills the whole process tree; child
  env scrubbed of credential-like variables (`scrubEnv`); stdout / stderr capped
  at `MAX_OUTPUT_BYTES`; destructive patterns blocked.
- File tools: every path confined to the working directory; reads capped at
  `MAX_READ_BYTES`.
- `search_code`: bounded regex (`compileBoundedRegex`), 1 MiB file cap, 100
  matches.
- `fetch_url`: `safeFetch` -- http/https only, private / loopback addresses
  refused on every hop, deadline, 1 MiB body cap.
- Git tools: refs validated by `assertSafeRef`, file lists by `assertSafeFiles`.

## Install

```sh
deno add jsr:@rullama/tool-builtins
```

## Quick Example

```ts
import { ToolContext } from "@rullama/core";
import { AgentCapabilities } from "@rullama/permission";
import {
  BashTool,
  createBuiltinExecutor,
  DEFAULT_TOOL_PROVIDERS,
  FileOpsTool,
} from "@rullama/tool-builtins";

// The default tool set behind the enforcing executor from @rullama/tool-runtime
const executor = createBuiltinExecutor({
  mode: "auto",
  capabilities: AgentCapabilities.standardDev(),
  onDecision: (e) => console.log(e.toolUse.name, e.outcome, e.reason ?? ""),
});
console.log(executor.availableTools().map((t) => t.name));

const ctx = new ToolContext({ working_directory: Deno.cwd() });
const listing = await executor.execute(
  { id: "call-1", name: "list_directory", input: { path: "." } },
  ctx,
);
console.log(listing.is_error, listing.content.slice(0, 200));

// Denied by the default policy engine (secret files)
const denied = await executor.execute(
  { id: "call-2", name: "read_file", input: { path: ".env" } },
  ctx,
);
console.log(denied.is_error, denied.content);

// A narrower tool set: only bash + file ops
const minimal = createBuiltinExecutor({ providers: [BashTool, FileOpsTool] });
console.log(minimal.availableTools().length, DEFAULT_TOOL_PROVIDERS.length);
```

Each tool class also exposes `getTools()` and
`execute(toolUseId, toolName, input, context)` for direct use, e.g.
`FileOpsTool.getTools()` to register definitions in a `ToolRegistry`.
