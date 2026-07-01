/**
 * Cross-package integration test: Tool registry and smart router.
 *
 * Verifies that @rullama/tools ToolRegistry correctly registers
 * built-in tools (WebTool, SearchTool) and that the smart router detects
 * appropriate categories for various queries.
 */

import {
  assert,
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  analyzeQuery,
  getToolsForCategories,
  SearchTool,
  type ToolCategory,
  ToolRegistry,
  WebTool,
} from "@rullama/tools";

// ---------------------------------------------------------------------------
// Tool registration tests
// ---------------------------------------------------------------------------

Deno.test("ToolRegistry registers WebTool tools", () => {
  const registry = new ToolRegistry();
  registry.registerTools(WebTool.getTools());

  assert(registry.length > 0, "registry should have at least one tool");
  const fetchUrl = registry.get("fetch_url");
  assert(fetchUrl !== undefined, "fetch_url tool should be registered");
  assertEquals(fetchUrl!.name, "fetch_url");
  assert(fetchUrl!.description.length > 0, "description should be non-empty");
});

Deno.test("ToolRegistry registers SearchTool tools", () => {
  const registry = new ToolRegistry();
  registry.registerTools(SearchTool.getTools());

  const searchCode = registry.get("search_code");
  assert(searchCode !== undefined, "search_code tool should be registered");
  assertEquals(searchCode!.name, "search_code");
});

Deno.test("ToolRegistry registers multiple tool sets", () => {
  const registry = new ToolRegistry();
  registry.registerTools(WebTool.getTools());
  registry.registerTools(SearchTool.getTools());

  // Both tools should be found
  assert(registry.get("fetch_url") !== undefined);
  assert(registry.get("search_code") !== undefined);
  assert(registry.length >= 2, "registry should have at least 2 tools");
});

Deno.test("ToolRegistry.getAll returns all registered tools", () => {
  const registry = new ToolRegistry();
  registry.registerTools(WebTool.getTools());
  registry.registerTools(SearchTool.getTools());

  const all = registry.getAll();
  const names = all.map((t) => t.name);
  assert(names.includes("fetch_url"), "should include fetch_url");
  assert(names.includes("search_code"), "should include search_code");
});

Deno.test("ToolRegistry.getByCategory returns Web tools", () => {
  const registry = new ToolRegistry();
  registry.registerTools(WebTool.getTools());

  const webTools = registry.getByCategory("Web");
  assertEquals(webTools.length, 1);
  assertEquals(webTools[0].name, "fetch_url");
});

Deno.test("ToolRegistry.getByCategory returns Search tools", () => {
  const registry = new ToolRegistry();
  registry.registerTools(SearchTool.getTools());

  const searchTools = registry.getByCategory("Search");
  assertEquals(searchTools.length, 1);
  assertEquals(searchTools[0].name, "search_code");
});

Deno.test("ToolRegistry.searchTools finds by keyword", () => {
  const registry = new ToolRegistry();
  registry.registerTools(WebTool.getTools());
  registry.registerTools(SearchTool.getTools());

  const results = registry.searchTools("fetch");
  assert(results.length >= 1, "should find at least one tool matching 'fetch'");
  assert(results.some((t) => t.name === "fetch_url"));
});

// ---------------------------------------------------------------------------
// Smart router category detection tests
// ---------------------------------------------------------------------------

Deno.test("analyzeQuery detects Git category", () => {
  const categories = analyzeQuery("show me the git diff for the last commit");
  assert(categories.includes("Git"), "should detect Git category");
});

Deno.test("analyzeQuery detects Web category", () => {
  const categories = analyzeQuery("fetch the API endpoint at this url");
  assert(categories.includes("Web"), "should detect Web category");
});

Deno.test("analyzeQuery detects Search category", () => {
  const categories = analyzeQuery("search for the pattern 'TODO' in all files");
  assert(categories.includes("Search"), "should detect Search category");
});

Deno.test("analyzeQuery detects Bash category", () => {
  const categories = analyzeQuery("run the npm build command");
  assert(categories.includes("Bash"), "should detect Bash category");
});

Deno.test("analyzeQuery detects FileOps category", () => {
  const categories = analyzeQuery("read the config file and edit it");
  assert(categories.includes("FileOps"), "should detect FileOps category");
});

Deno.test("analyzeQuery always includes FileOps", () => {
  const categories = analyzeQuery("commit the changes to git");
  assert(categories.includes("FileOps"), "FileOps should always be included");
});

Deno.test("analyzeQuery returns defaults for unrecognized query", () => {
  const categories = analyzeQuery("xyzzy plugh");
  // Default set: FileOps, Search, Bash
  assert(categories.includes("FileOps"), "defaults should include FileOps");
  assert(categories.includes("Search"), "defaults should include Search");
  assert(categories.includes("Bash"), "defaults should include Bash");
});

Deno.test("getToolsForCategories returns matching tools", () => {
  const registry = new ToolRegistry();
  registry.registerTools(WebTool.getTools());
  registry.registerTools(SearchTool.getTools());

  const tools = getToolsForCategories(registry, ["Web", "Search"]);
  const names = tools.map((t) => t.name);
  assert(names.includes("fetch_url"), "should include fetch_url for Web");
  assert(
    names.includes("search_code"),
    "should include search_code for Search",
  );
});

Deno.test("getToolsForCategories deduplicates tools", () => {
  const registry = new ToolRegistry();
  registry.registerTools(SearchTool.getTools());

  // search_code appears in both Search and FileOps(search_files)
  // but search_code is only in Search category mapping
  const tools = getToolsForCategories(registry, ["Search", "Search"]);
  const searchCodeCount = tools.filter((t) => t.name === "search_code").length;
  assertEquals(searchCodeCount, 1, "each tool should appear at most once");
});
