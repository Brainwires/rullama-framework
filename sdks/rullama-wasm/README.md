# rullama-wasm

[![Crates.io](https://img.shields.io/crates/v/rullama-wasm.svg)](https://crates.io/crates/rullama-wasm)
[![Documentation](https://img.shields.io/docsrs/rullama-wasm)](https://docs.rs/rullama-wasm)
[![License](https://img.shields.io/crates/l/rullama-wasm.svg)](LICENSE)

WebAssembly bindings for the Brainwires Agent Framework.

## Overview

`rullama-wasm` provides a JavaScript-friendly API for running the Brainwires Agent Framework in browser and Node.js environments via WebAssembly. The crate exposes core type validation, conversation history serialization, and a sandboxed tool orchestrator that lets AI models execute Rhai scripts calling registered JavaScript tool callbacks — all with configurable resource limits to prevent runaway execution.

**Design principles:**

- **Browser-native** — compiles to `cdylib` WASM module consumable by `wasm-pack`, `wasm-bindgen`, or any bundler; no OS-specific dependencies
- **Zero-copy validation** — parse and re-serialize Messages and Tools through the canonical Rust types to guarantee schema conformance from JavaScript
- **Sandboxed orchestration** — Rhai script engine runs with operation limits, call-depth caps, string/array size bounds, and real-time timeouts to prevent abuse
- **Incremental opt-in** — default build includes only validation and serialization; heavier features (`interpreters`, `orchestrator`) are behind Cargo feature flags
- **Transparent interop** — all public functions accept and return JSON strings or `JsValue`, making TypeScript integration straightforward

```text
  ┌───────────────────────────────────────────────────────────────────────┐
  │                          rullama-wasm                             │
  │                                                                      │
  │  JavaScript / TypeScript                                             │
  │      │                                                               │
  │      ▼                                                               │
  │  ┌─── Core Bindings ──────────────────────────────────────────────┐  │
  │  │  version()            → framework version string               │  │
  │  │  validate_message()   → parse + re-serialize Message JSON      │  │
  │  │  validate_tool()      → parse + re-serialize Tool JSON         │  │
  │  │  serialize_history()  → messages → stateless protocol format   │  │
  │  └────────────────────────────────────────────────────────────────┘  │
  │                                                                      │
  │  ┌─── Re-exports (Rust consumers) ────────────────────────────────┐  │
  │  │  rullama_core   — Message, Tool, Content, Role, …          │  │
  │  │  rullama_mdap   — MdapConfig, MdapPreset, MdapMetrics, …   │  │
  │  │  rullama_code_interpreters  (interpreters feature)          │  │
  │  └────────────────────────────────────────────────────────────────┘  │
  │                                                                      │
  │  ┌─── WasmOrchestrator (orchestrator feature) ────────────────────┐  │
  │  │  register_tool(name, js_callback)                              │  │
  │  │  registered_tools() → [String]                                 │  │
  │  │  execute(script, limits) → OrchestratorResult                  │  │
  │  │                                                                │  │
  │  │  ExecutionLimits                                               │  │
  │  │    ::new()      — default (100k ops, 50 calls)                 │  │
  │  │    ::quick()    — constrained (10k ops, 10 calls)              │  │
  │  │    ::extended() — generous (for complex orchestration)         │  │
  │  │                                                                │  │
  │  │  Rhai Engine (sandboxed)                                       │  │
  │  │    max_operations • max_tool_calls • timeout_ms                │  │
  │  │    max_string_size • max_array_size • max_map_size             │  │
  │  │    max_expr_depth(64) • max_call_depth(64)                     │  │
  │  └────────────────────────────────────────────────────────────────┘  │
  └───────────────────────────────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
rullama-wasm = "0.11"
```

Build with `wasm-pack`:

```bash
wasm-pack build --target web
```

Use from JavaScript / TypeScript:

```js
import init, {
  version,
  validate_message,
  validate_tool,
  serialize_history,
} from "./pkg/rullama_wasm.js";

await init();

// Check framework version
console.log(version()); // → "0.2.0"

// Validate a message
const msg = JSON.stringify({
  role: "user",
  content: [{ type: "text", text: "Hello" }],
});
const normalized = validate_message(msg);
console.log(normalized); // → canonical JSON

// Validate a tool definition
const tool = JSON.stringify({
  name: "read_file",
  description: "Read a file from disk",
  input_schema: { type: "object", properties: { path: { type: "string" } } },
});
const normalizedTool = validate_tool(tool);

// Serialize conversation history for API requests
const history = JSON.stringify([
  { role: "user", content: [{ type: "text", text: "Hello" }] },
  { role: "assistant", content: [{ type: "text", text: "Hi there!" }] },
]);
const stateless = serialize_history(history);
console.log(stateless); // → stateless protocol format
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| *(none)* | Yes | Core bindings: `version`, `validate_message`, `validate_tool`, `serialize_history` |
| `interpreters` | No | Enables `rullama-tool-builtins/interpreters` re-export for WASM code execution |
| `orchestrator` | No | Enables `WasmOrchestrator` and `ExecutionLimits` with Rhai script engine, `js-sys`, `web-sys`, and real-time timeout support |

```toml
# Default (validation + serialization only)
rullama-wasm = "0.11"

# With code interpreters
rullama-wasm = { version = "0.11", features = ["interpreters"] }

# With tool orchestration
rullama-wasm = { version = "0.11", features = ["orchestrator"] }

# Everything enabled
rullama-wasm = { version = "0.11", features = ["interpreters", "orchestrator"] }
```

## Architecture

### Core Bindings

Free functions exposed to JavaScript via `#[wasm_bindgen]`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `version` | `() → String` | Returns the crate version (`CARGO_PKG_VERSION`) |
| `validate_message` | `(json: &str) → Result<String, String>` | Parse JSON into `rullama_core::Message`, re-serialize to canonical form |
| `validate_tool` | `(json: &str) → Result<String, String>` | Parse JSON into `rullama_core::Tool`, re-serialize to canonical form |
| `serialize_history` | `(messages_json: &str) → Result<String, String>` | Convert `Vec<Message>` JSON to stateless protocol format via `serialize_messages_to_stateless_history` |

All functions accept plain JSON strings and return JSON strings or descriptive error messages, making them easy to call from any JavaScript runtime.

### ExecutionLimits (requires `orchestrator` feature)

WASM-compatible wrapper around `rullama_tool_runtime::orchestrator::ExecutionLimits` with JavaScript getter/setter bindings.

| Constructor | Description |
|-------------|-------------|
| `new()` | Default limits — balanced for typical scripts |
| `quick()` | Constrained limits for simple, fast scripts |
| `extended()` | Generous limits for complex orchestration |

**Default preset values:**

| Property | `new()` | `quick()` | `extended()` |
|----------|---------|-----------|--------------|
| `max_operations` | 100,000 | 10,000 | *(extended)* |
| `max_tool_calls` | 50 | 10 | *(extended)* |
| `timeout_ms` | *(default)* | *(quick)* | *(extended)* |
| `max_string_size` | *(default)* | *(quick)* | *(extended)* |
| `max_array_size` | *(default)* | *(quick)* | *(extended)* |

All properties have JavaScript-compatible getters and setters:

| Property | Type | Description |
|----------|------|-------------|
| `max_operations` | `u64` | Maximum Rhai operations before termination |
| `max_tool_calls` | `usize` | Maximum tool invocations per execution |
| `timeout_ms` | `u64` | Real-time wall-clock timeout in milliseconds |
| `max_string_size` | `usize` | Maximum allowed string length in the script |
| `max_array_size` | `usize` | Maximum allowed array length in the script |

### WasmOrchestrator (requires `orchestrator` feature)

JavaScript-compatible tool orchestrator that executes Rhai scripts with registered tool callbacks.

| Method | Description |
|--------|-------------|
| `new()` | Create orchestrator; sets up `console_error_panic_hook` for better browser error messages |
| `register_tool(name, callback)` | Register a JavaScript function as a tool executor; callback receives JSON string, returns string |
| `registered_tools()` | List names of all registered tools → `Vec<String>` |
| `execute(script, limits)` | Execute a Rhai script with resource limits → `Result<JsValue, JsValue>` containing `OrchestratorResult` |

**Engine safety configuration:**

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_EXPR_DEPTH` | 64 | Maximum expression nesting depth (prevents stack overflow) |
| `MAX_CALL_DEPTH` | 64 | Maximum function call nesting depth (prevents deep recursion) |

**`OrchestratorResult` (returned by `execute`):**

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether script completed without errors |
| `output` | `String` | Script return value (or error message) |
| `tool_calls` | `Vec<ToolCall>` | Record of every tool invocation during execution |
| `execution_time_ms` | `u64` | Total wall-clock execution time |

**`ToolCall` (per-invocation record):**

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `String` | Name of the tool called |
| `input` | `Value` | JSON input passed to the tool |
| `output` | `String` | String result returned by the tool |
| `success` | `bool` | Whether the call succeeded |
| `duration_ms` | `u64` | Wall-clock duration of this call |

**Error handling during execution:**

| Error | Reported As |
|-------|-------------|
| Script compilation failure | `OrchestratorResult::error` with `"Compilation error: ..."` |
| Operation limit exceeded | `"Script exceeded maximum operations (N)"` |
| Timeout exceeded | `"Script execution timed out after Nms"` |
| Tool call limit exceeded | Tool returns `"ERROR: Maximum tool calls (N) exceeded"` |
| JavaScript callback throws | Tool returns `"Tool error: ..."` |
| Other Rhai runtime errors | `"Execution error: ..."` |

### Re-exports

For Rust consumers, the crate re-exports WASM-safe framework crates:

| Re-export | Always | Feature |
|-----------|--------|---------|
| `rullama_core` | Yes | — |
| `rullama_mdap` | Yes | — |
| `rullama_code_interpreters` | No | `interpreters` |
| `WasmOrchestrator` | No | `orchestrator` |
| `WasmExecutionLimits` | No | `orchestrator` |

## Usage Examples

### Validate messages from JavaScript

```js
import init, { validate_message } from "./pkg/rullama_wasm.js";
await init();

// Valid message — returns normalized JSON
const valid = validate_message(
  '{"role":"user","content":[{"type":"text","text":"Hello"}]}'
);
console.log(JSON.parse(valid));

// Invalid message — returns descriptive error
try {
  validate_message('{"role":"invalid"}');
} catch (e) {
  console.error(e); // → "Invalid message JSON: ..."
}
```

### Validate tool definitions

```js
import init, { validate_tool } from "./pkg/rullama_wasm.js";
await init();

const tool = validate_tool(
  JSON.stringify({
    name: "search",
    description: "Search the codebase",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
      },
      required: ["query"],
    },
  })
);
console.log(JSON.parse(tool));
```

### Serialize conversation history for API calls

```js
import init, { serialize_history } from "./pkg/rullama_wasm.js";
await init();

const messages = [
  { role: "user", content: [{ type: "text", text: "What is Rust?" }] },
  {
    role: "assistant",
    content: [{ type: "text", text: "Rust is a systems programming language." }],
  },
  { role: "user", content: [{ type: "text", text: "Show me an example." }] },
];

const stateless = serialize_history(JSON.stringify(messages));
// → Stateless protocol format ready for API requests
```

### Execute Rhai scripts with tool orchestration

```js
import init, {
  WasmOrchestrator,
  ExecutionLimits,
} from "./pkg/rullama_wasm.js";
await init();

const orchestrator = new WasmOrchestrator();

// Register tools as JavaScript callbacks
orchestrator.register_tool("read_file", (jsonInput) => {
  const { path } = JSON.parse(jsonInput);
  return `Contents of ${path}: hello world`;
});

orchestrator.register_tool("write_file", (jsonInput) => {
  const { path, content } = JSON.parse(jsonInput);
  return `Wrote ${content.length} bytes to ${path}`;
});

console.log(orchestrator.registered_tools());
// → ["read_file", "write_file"]

// Execute a Rhai script that calls the registered tools
const limits = new ExecutionLimits(); // default: 100k ops, 50 calls
const result = orchestrator.execute(
  `
    let data = read_file(#{ path: "config.json" });
    let summary = "Processed: " + data;
    write_file(#{ path: "output.txt", content: summary });
    summary
  `,
  limits
);

console.log(result.success); // → true
console.log(result.output); // → "Processed: Contents of config.json: hello world"
console.log(result.tool_calls.length); // → 2
console.log(result.execution_time_ms); // → 5
```

### Use quick limits for simple scripts

```js
import init, {
  WasmOrchestrator,
  ExecutionLimits,
} from "./pkg/rullama_wasm.js";
await init();

const orchestrator = new WasmOrchestrator();
orchestrator.register_tool("greet", (jsonInput) => {
  const { name } = JSON.parse(jsonInput);
  return `Hello, ${name}!`;
});

// Quick limits: 10k operations, 10 tool calls
const limits = ExecutionLimits.quick();
const result = orchestrator.execute(
  `greet(#{ name: "World" })`,
  limits
);
console.log(result.output); // → "Hello, World!"
```

### Custom execution limits

```js
import init, { ExecutionLimits } from "./pkg/rullama_wasm.js";
await init();

const limits = new ExecutionLimits();

// Customize individual properties
limits.max_operations = 50_000;
limits.max_tool_calls = 20;
limits.timeout_ms = 5000;
limits.max_string_size = 1024 * 1024; // 1 MB
limits.max_array_size = 10_000;

console.log(limits.max_operations); // → 50000
console.log(limits.max_tool_calls); // → 20
```

### Use from Rust (re-exports)

```rust
use rullama_wasm::rullama_core::{Message, Tool, Role};
use rullama_wasm::rullama_mdap::MdapConfig;

// Access all core types through the WASM crate
let msg = Message {
    role: Role::User,
    content: vec![],
    ..Default::default()
};

// WASM bindings also available
let version = rullama_wasm::version();
```

## Integration

Use via the `rullama` facade crate with the `wasm` feature, or depend on `rullama-wasm` directly:

```toml
# Via facade
[dependencies]
rullama = { version = "0.11", features = ["wasm"] }

# Direct
[dependencies]
rullama-wasm = "0.11"
```

The crate re-exports all components at the top level:

```rust
use rullama_wasm::{
    // WASM bindings (always available)
    version, validate_message, validate_tool, serialize_history,

    // Core framework types (Rust consumers)
    rullama_core,
    rullama_mdap,
};

// With `interpreters` feature
#[cfg(feature = "interpreters")]
use rullama_wasm::rullama_code_interpreters;

// With `orchestrator` feature
#[cfg(feature = "orchestrator")]
use rullama_wasm::{WasmExecutionLimits, WasmOrchestrator};
use rullama_wasm::wasm_orchestrator::{ExecutionLimits, WasmOrchestrator};
```

### Building for different targets

```bash
# Browser (ES module)
wasm-pack build --target web

# Bundler (webpack, vite, etc.)
wasm-pack build --target bundler

# Node.js
wasm-pack build --target nodejs

# No bundler (standalone)
wasm-pack build --target no-modules
```

## License

Licensed under the MIT License. See [LICENSE](../../LICENSE) for details.
