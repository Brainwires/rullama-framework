# @rullama/tool-runtime

Tool execution framework: `ToolRegistry`, the `ToolExecutor` / `ToolPreHook`
interfaces, the **enforcing executor** (`EnforcingExecutor` / `enforce()`),
error taxonomy, sanitization, shared guards (`confinePath`,
`compileBoundedRegex`, `safeFetch`), smart routing, `TransactionManager`,
OpenAPI tool generation, OAuth, tool search / embedding and `ValidationTool`.

Extracted from the old `@rullama/tools` package in v0.11.0 to mirror Rust's
`rullama-tool-runtime` crate. Depends on `@rullama/core` and
`@rullama/permission`.

Built-in tool implementations (`BashTool`, `FileOpsTool`, `GitTool`, `WebTool`,
`SearchTool`, `SemanticSearchTool`, `CalendarTool`, `SessionsTool`) live in the
sibling package `@rullama/tool-builtins`, whose `createBuiltinExecutor()`
returns an already-enforcing executor.

## Enforcement

`enforce(executor, options)` wraps any `ToolExecutor` and, before and after
every call, applies in order: **permission mode** (`"read-only"` admits only
read/search/planning tools), **capabilities** (`AgentCapabilities` narrows
tools, paths, domains, git ops), **policy** (`PolicyEngine.withDefaults()`
denies `.env` / secret / credential files and asks before `git reset` /
`git rebase`; `require_approval` goes to your `ApprovalHandler`, denied without
one), **pre-execute hooks** (`ToolPreHook.beforeExecute` may `reject()`), and
**output filtering** (secrets redacted, web-fetched content wrapped as
untrusted). `@rullama/inference`'s `AgentContext` applies this by default.

## Install

```sh
deno add jsr:@rullama/tool-runtime
```

## Quick Example

```ts
import {
  objectSchema,
  type Tool,
  ToolContext,
  ToolResult,
  type ToolUse,
} from "@rullama/core";
import { AgentCapabilities, PolicyEngine } from "@rullama/permission";
import {
  allow,
  enforce,
  type EnforcementEvent,
  reject,
  type ToolExecutor,
  type ToolPreHook,
  ToolRegistry,
} from "@rullama/tool-runtime";

// 1. A custom executor
const weather: Tool = {
  name: "weather",
  description: "Get weather for a city",
  input_schema: objectSchema({ city: { type: "string" } }, ["city"]),
};
const raw: ToolExecutor = {
  availableTools: () => [weather],
  execute: (use: ToolUse) =>
    Promise.resolve(
      ToolResult.success(use.id, `Weather in ${use.input.city}: sunny`),
    ),
};

// 2. A pre-execute hook
const noParis: ToolPreHook = {
  beforeExecute: (use) =>
    Promise.resolve(
      use.input.city === "Paris" ? reject("Paris is blocked") : allow(),
    ),
};

// 3. Enforce: mode + capabilities + policy + hooks + output filtering
const audit: EnforcementEvent[] = [];
const executor = enforce(raw, {
  mode: "auto",
  capabilities: AgentCapabilities.standardDev(),
  policy: PolicyEngine.withDefaults(),
  preHooks: [noParis],
  approve: (use) => {
    console.log(`approval requested for ${use.name}`);
    return false;
  },
  onDecision: (event) => audit.push(event),
});

const ctx = new ToolContext({ working_directory: Deno.cwd() });
const ok = await executor.execute(
  { id: "1", name: "weather", input: { city: "Oslo" } },
  ctx,
);
const blocked = await executor.execute(
  { id: "2", name: "weather", input: { city: "Paris" } },
  ctx,
);
console.log(ok.content, blocked.is_error, audit.map((e) => e.outcome));

// 4. Registry of definitions for the provider
const registry = new ToolRegistry();
registry.registerTools(executor.availableTools());
console.log(registry.getAll().length, registry.get("weather")?.description);
```

## Sub-path exports

`@rullama/tool-runtime/registry`, `/executor`, `/router`, `/oauth`, `/openapi`,
`/search`, `/transaction`, `/validation`.
