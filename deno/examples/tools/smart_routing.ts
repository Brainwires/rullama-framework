// Example: Smart Routing
// Demonstrates the smart tool router that analyzes user queries to determine
// which tool categories are relevant, using keyword-based pattern matching.
// Run: deno run deno/examples/tool-system/smart_routing.ts

import { Message } from "@rullama/core";
import {
  analyzeMessages,
  analyzeQuery,
  getContextForAnalysis,
  getSmartTools,
  getSmartToolsWithMcp,
  getToolsForCategories,
  ToolRegistry,
  ValidationTool,
} from "@rullama/tool-runtime";
import {
  BashTool,
  FileOpsTool,
  GitTool,
  SearchTool,
  WebTool,
} from "@rullama/tool-builtins";
import type { Tool } from "@rullama/core";

async function main() {
  console.log("=== Smart Routing Example ===\n");

  // 1. Set up a registry with built-in tools
  const registry = new ToolRegistry();
  registry.registerTools(BashTool.getTools());
  registry.registerTools(FileOpsTool.getTools());
  registry.registerTools(GitTool.getTools());
  registry.registerTools(SearchTool.getTools());
  registry.registerTools(WebTool.getTools());
  registry.registerTools(ValidationTool.getTools());
  console.log(`Registry loaded with ${registry.length} tools.\n`);

  // 2. Analyze individual queries
  console.log("=== Query Analysis ===\n");

  const queries = [
    "Read the file src/main.ts and edit it",
    "Search for all TODO comments in the codebase",
    "Show me the git log and diff for the last commit",
    "Run npm test and build the project",
    "Fetch the API docs from https://example.com",
    "Plan the architecture for the new auth module",
    "Create a task to refactor the database layer",
    "Tell me a joke",
  ];

  for (const query of queries) {
    const categories = analyzeQuery(query);
    console.log(`  Query: "${query}"`);
    console.log(`  Categories: [${categories.join(", ")}]\n`);
  }

  // 3. Analyze conversation messages with adaptive context window
  console.log("=== Message Analysis with Adaptive Context ===\n");

  // Short prompt -- uses last 3 messages for context
  const shortConversation = [
    Message.user("Fix the bug"),
    Message.assistant("Which bug?"),
    Message.user("The git issue"),
  ];

  const shortContext = getContextForAnalysis(shortConversation);
  const shortCategories = analyzeMessages(shortConversation);
  console.log(`  Short prompt context: "${shortContext.substring(0, 80)}..."`);
  console.log(`  Detected categories: [${shortCategories.join(", ")}]\n`);

  // Detailed prompt -- uses current message only
  const detailedConversation = [
    Message.user(
      "Please read the file src/config.ts, search for any deprecated API calls, " +
        "then edit the file to replace them with the new API. After that, run the tests " +
        "to make sure everything works and commit the changes with a descriptive message.",
    ),
  ];

  const detailedContext = getContextForAnalysis(detailedConversation);
  const detailedCategories = analyzeMessages(detailedConversation);
  console.log(
    `  Detailed prompt context: "${detailedContext.substring(0, 80)}..."`,
  );
  console.log(`  Detected categories: [${detailedCategories.join(", ")}]\n`);

  // 4. Get smart-routed tools for a conversation
  console.log("=== Smart Tool Selection ===\n");

  const conversation = [
    Message.user("Check git status and read the README"),
  ];

  const smartTools = getSmartTools(conversation, registry);
  console.log(`  Query: "Check git status and read the README"`);
  console.log(`  Selected tools (${smartTools.length}):`);
  for (const tool of smartTools) {
    console.log(`    - ${tool.name}: ${tool.description.substring(0, 60)}`);
  }

  // 5. Get tools for explicit categories
  console.log("\n=== Tools for Explicit Categories ===\n");

  const explicitCategories = getToolsForCategories(registry, ["Git", "Search"]);
  console.log(`  Categories: Git, Search`);
  console.log(`  Tools (${explicitCategories.length}):`);
  for (const tool of explicitCategories) {
    console.log(`    - ${tool.name}`);
  }

  // 6. Smart routing with MCP tools
  console.log("\n=== Smart Routing with MCP Tools ===\n");

  const mcpTools: Tool[] = [
    {
      name: "mcp_database_query",
      description: "Execute a SQL query against the database",
      input_schema: {
        type: "object",
        properties: { sql: { type: "string" } },
        required: ["sql"],
      },
    },
    {
      name: "mcp_docker_exec",
      description: "Execute a command in a Docker container",
      input_schema: {
        type: "object",
        properties: { cmd: { type: "string" } },
        required: ["cmd"],
      },
    },
    {
      name: "mcp_slack_send",
      description: "Send a message to a Slack channel",
      input_schema: {
        type: "object",
        properties: { msg: { type: "string" } },
        required: ["msg"],
      },
    },
  ];

  const mcpConversation = [
    Message.user("Run the database migration and execute the tests"),
  ];

  const withMcp = getSmartToolsWithMcp(mcpConversation, registry, mcpTools);
  console.log(`  Query: "Run the database migration and execute the tests"`);
  console.log(`  Selected tools including MCP (${withMcp.length}):`);
  for (const tool of withMcp) {
    const isMcp = tool.name.startsWith("mcp_");
    console.log(`    - ${tool.name}${isMcp ? " [MCP]" : ""}`);
  }

  // 7. Default fallback when no categories match
  console.log("\n=== Default Fallback ===\n");

  const ambiguousQuery = "Tell me a joke";
  const fallbackCategories = analyzeQuery(ambiguousQuery);
  console.log(`  Query: "${ambiguousQuery}"`);
  console.log(`  Fallback categories: [${fallbackCategories.join(", ")}]`);
  console.log(
    "  (FileOps, Search, Bash are used as defaults when nothing matches)",
  );

  console.log("\nDone.");
}

await main();
