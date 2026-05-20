import { assert, assertEquals } from "@std/assert";
import { SemanticSearchTool } from "./semantic_search.ts";

Deno.test("getTools returns six semantic-search tools", () => {
  const tools = SemanticSearchTool.getTools();
  assertEquals(tools.length, 6);
  const names = tools.map((t) => t.name);
  assert(names.includes("index_codebase"));
  assert(names.includes("query_codebase"));
  assert(names.includes("search_with_filters"));
  assert(names.includes("get_rag_statistics"));
  assert(names.includes("clear_rag_index"));
  assert(names.includes("search_git_history"));
});

Deno.test("index_codebase tool definition", () => {
  const tool = SemanticSearchTool.getTools().find(
    (t) => t.name === "index_codebase",
  )!;
  assert(tool.description.includes("Index"));
  assert(tool.description.includes("semantic"));
  assertEquals(tool.requires_approval, false);
  assertEquals(tool.defer_loading, true);
});

Deno.test("query_codebase tool definition", () => {
  const tool = SemanticSearchTool.getTools().find(
    (t) => t.name === "query_codebase",
  )!;
  assert(tool.description.includes("Search"));
  assertEquals(tool.requires_approval, false);
});

Deno.test("clear_rag_index requires approval", () => {
  const tool = SemanticSearchTool.getTools().find(
    (t) => t.name === "clear_rag_index",
  )!;
  assert(tool.requires_approval);
});

Deno.test("every tool has a description", () => {
  for (const tool of SemanticSearchTool.getTools()) {
    assert(tool.description.length > 0, `${tool.name} missing description`);
  }
});

Deno.test("index_codebase requires path", () => {
  const tool = SemanticSearchTool.getTools().find(
    (t) => t.name === "index_codebase",
  )!;
  const req = tool.input_schema.required ?? [];
  assert(req.includes("path"));
});
