// Example: Tool definition and execution
// Shows how to define tools, create tool results, and use the idempotency registry.
// Run: deno run deno/examples/core/tool_usage.ts

import {
  defaultToolInputSchema,
  IdempotencyRegistry,
  Message,
  objectSchema,
  type Tool,
  ToolContext,
  type ToolMode,
  toolModeDisplayName,
  ToolResult,
} from "@rullama/core";

async function main() {
  console.log("=== Tool Definition and Usage ===");

  // 1. Define tools with JSON Schema input specifications
  console.log("\n=== Defining Tools ===");

  const readFileTool: Tool = {
    name: "read_file",
    description: "Read the contents of a file at the given path",
    input_schema: objectSchema(
      {
        path: { type: "string", description: "Absolute path to the file" },
        encoding: {
          type: "string",
          description: "File encoding",
          default: "utf-8",
        },
      },
      ["path"],
    ),
  };

  const searchTool: Tool = {
    name: "search_codebase",
    description: "Search for code patterns across the project",
    input_schema: objectSchema(
      {
        query: { type: "string", description: "Search query or regex pattern" },
        file_glob: {
          type: "string",
          description: "Glob pattern to filter files",
        },
        max_results: {
          type: "number",
          description: "Maximum results to return",
          default: 10,
        },
      },
      ["query"],
    ),
    requires_approval: false,
    allowed_callers: ["direct"],
  };

  const writeFileTool: Tool = {
    name: "write_file",
    description: "Write content to a file (requires approval)",
    input_schema: objectSchema(
      {
        path: { type: "string", description: "Target file path" },
        content: { type: "string", description: "Content to write" },
      },
      ["path", "content"],
    ),
    requires_approval: true,
  };

  const statusTool: Tool = {
    name: "get_status",
    description: "Get the current system status",
    input_schema: defaultToolInputSchema(),
  };

  const tools = [readFileTool, searchTool, writeFileTool, statusTool];

  for (const tool of tools) {
    const required = tool.input_schema.required ?? [];
    const approval = tool.requires_approval ? " [requires approval]" : "";
    console.log(
      `  ${tool.name}: ${required.length} required params${approval}`,
    );
  }

  // 2. Create tool execution results
  console.log("\n=== Tool Results ===");

  const successResult = ToolResult.success(
    "call-001",
    "File contents: export function main() { ... }",
  );
  console.log(
    `Success result: tool_use_id=${successResult.tool_use_id}, is_error=${successResult.is_error}`,
  );
  console.log(`  Content: ${successResult.content.slice(0, 60)}...`);

  const errorResult = ToolResult.error(
    "call-002",
    "ENOENT: file not found — /src/missing.ts",
  );
  console.log(
    `Error result: tool_use_id=${errorResult.tool_use_id}, is_error=${errorResult.is_error}`,
  );
  console.log(`  Content: ${errorResult.content}`);

  // 3. Tool context with idempotency
  console.log("\n=== Tool Context & Idempotency ===");

  const ctx = new ToolContext({
    working_directory: "/home/user/project",
    user_id: "dev-1",
    metadata: { session: "abc123", run_id: "run-42" },
  });
  ctx.withIdempotencyRegistry();

  console.log(`Working directory: ${ctx.working_directory}`);
  console.log(`User: ${ctx.user_id}`);
  console.log(`Metadata: ${JSON.stringify(ctx.metadata)}`);

  // Simulate idempotent write operations
  const registry = ctx.idempotency_registry!;

  // First write to a file — executes normally
  const key1 = "write:/src/config.ts";
  const existing = registry.get(key1);
  if (existing) {
    console.log(`\n  [cached] ${key1} => ${existing.cached_result}`);
  } else {
    console.log(`\n  [execute] ${key1} — first time, performing write`);
    registry.record(key1, "wrote 42 bytes to /src/config.ts");
  }

  // Second write to the same file — returns cached result
  const cached = registry.get(key1);
  if (cached) {
    console.log(`  [cached] ${key1} => ${cached.cached_result}`);
  }

  console.log(
    `  Registry size: ${registry.length}, empty: ${registry.isEmpty()}`,
  );

  // 4. Tool modes
  console.log("\n=== Tool Modes ===");

  const modes: ToolMode[] = [
    { type: "full" },
    { type: "explicit", tools: ["read_file", "search_codebase"] },
    { type: "smart" },
    { type: "core" },
    { type: "none" },
  ];

  for (const mode of modes) {
    const display = toolModeDisplayName(mode);
    const detail = mode.type === "explicit"
      ? ` (${mode.tools.join(", ")})`
      : "";
    console.log(`  ${display}${detail}`);
  }

  // 5. Messages with tool use blocks
  console.log("\n=== Messages with Tool Use ===");

  const assistantMsg = new Message({
    role: "assistant",
    content: [
      { type: "text", text: "Let me read that file for you." },
      {
        type: "tool_use",
        id: "call-003",
        name: "read_file",
        input: { path: "/src/main.ts" },
      },
    ],
  });
  console.log(`Assistant message: ${assistantMsg.textOrSummary()}`);

  const toolResultMsg = Message.toolResult(
    "call-003",
    "export function main() { console.log('hello'); }",
  );
  console.log(`Tool result message role: ${toolResultMsg.role}`);

  console.log(
    "\nDone! Define tools with JSON Schema and let the AI agent use them.",
  );
}

await main();
