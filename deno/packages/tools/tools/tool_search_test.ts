import { assertEquals } from "@std/assert";
import { DEFAULT_SEARCH_MODE, ToolSearchTool } from "./tool_search.ts";

Deno.test("getTools returns exactly one tool named search_tools", () => {
  const tools = ToolSearchTool.getTools();
  assertEquals(tools.length, 1);
  assertEquals(tools[0].name, "search_tools");
});

Deno.test("default search mode is keyword", () => {
  assertEquals(DEFAULT_SEARCH_MODE, "keyword");
});
