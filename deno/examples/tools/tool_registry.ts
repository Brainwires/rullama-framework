// Example: Tool Registry
// Shows how to create a ToolRegistry, register built-in and custom tools,
// list tools by category, look up metadata, and search.
// Run: deno run deno/examples/tool-system/tool_registry.ts

import type { Tool } from "@rullama/core";
import { objectSchema } from "@rullama/core";
import {
  BashTool,
  FileOpsTool,
  GitTool,
  SearchTool,
  ToolRegistry,
  ValidationTool,
  WebTool,
} from "@rullama/tools";
import type { ToolCategory } from "@rullama/tools";

/** Helper: create a custom tool definition with the given name and description. */
function makeCustomTool(name: string, description: string): Tool {
  return {
    name,
    description,
    input_schema: objectSchema(
      { input: { type: "string", description: "The input value" } },
      ["input"],
    ),
    requires_approval: false,
    defer_loading: false,
  };
}

async function main() {
  console.log("=== Tool Registry Example ===\n");

  // 1. Create a registry and populate it with built-in tools
  const registry = new ToolRegistry();
  registry.registerTools(BashTool.getTools());
  registry.registerTools(FileOpsTool.getTools());
  registry.registerTools(GitTool.getTools());
  registry.registerTools(SearchTool.getTools());
  registry.registerTools(WebTool.getTools());
  registry.registerTools(ValidationTool.getTools());
  console.log(`Registry created with ${registry.length} built-in tool(s).`);

  // 2. Register custom tools
  const customTools = [
    makeCustomTool("translate_text", "Translate text between languages"),
    makeCustomTool("summarize", "Summarize a block of text"),
  ];
  registry.registerTools(customTools);
  console.log(`After adding custom tools: ${registry.length} total.`);

  // Register a single tool that requires approval before execution.
  const sensitiveTool: Tool = {
    name: "deploy_production",
    description: "Deploy the application to production",
    input_schema: objectSchema({}, []),
    requires_approval: true,
    defer_loading: true,
  };
  registry.register(sensitiveTool);
  console.log(
    "Registered 'deploy_production' (requires approval, deferred).\n",
  );

  // 3. List tools by category
  const categories: [string, ToolCategory][] = [
    ["FileOps", "FileOps"],
    ["Git", "Git"],
    ["Search", "Search"],
    ["Bash", "Bash"],
    ["Web", "Web"],
    ["Validation", "Validation"],
  ];

  console.log("Tools by category:");
  for (const [label, cat] of categories) {
    const tools = registry.getByCategory(cat);
    if (tools.length === 0) {
      console.log(`  ${label}: (none registered)`);
    } else {
      const names = tools.map((t: { name: string }) => t.name);
      console.log(`  ${label}: ${names.join(", ")}`);
    }
  }

  // 4. Look up a tool by name and inspect its metadata
  console.log("\nTool metadata lookup:");
  const tool = registry.get("translate_text");
  if (tool) {
    console.log(`  Name:              ${tool.name}`);
    console.log(`  Description:       ${tool.description}`);
    console.log(`  Requires approval: ${tool.requires_approval ?? false}`);
    console.log(`  Defer loading:     ${tool.defer_loading ?? false}`);
    console.log(
      `  Schema:            ${JSON.stringify(tool.input_schema, null, 2)}`,
    );
  }

  // 5. Search tools by keyword
  const query = "file";
  const results = registry.searchTools(query);
  console.log(`\nSearch for "${query}": ${results.length} result(s)`);
  for (const t of results) {
    console.log(`  - ${t.name} : ${t.description}`);
  }

  // 6. Initial vs. deferred tools
  const initial = registry.getInitialTools();
  const deferred = registry.getDeferredTools();
  console.log(
    `\nInitial tools: ${initial.length}, Deferred tools: ${deferred.length}`,
  );

  // 7. Core tools subset
  const core = registry.getCore();
  console.log(
    `Core tools (${core.length}): ${
      core.map((t: { name: string }) => t.name).join(", ")
    }`,
  );

  console.log("\nDone.");
}

await main();
