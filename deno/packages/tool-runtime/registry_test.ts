import { assertEquals, assertExists } from "@std/assert";
import { ToolRegistry } from "./registry.ts";
import type { Tool } from "@rullama/core";
import { objectSchema } from "@rullama/core";

function makeTool(name: string, defer = false): Tool {
  return {
    name,
    description: `A ${name} tool`,
    input_schema: objectSchema({}, []),
    requires_approval: false,
    defer_loading: defer,
  };
}

Deno.test("ToolRegistry - new is empty", () => {
  const registry = new ToolRegistry();
  assertEquals(registry.isEmpty(), true);
  assertEquals(registry.length, 0);
});

Deno.test("ToolRegistry - register single", () => {
  const registry = new ToolRegistry();
  registry.register(makeTool("test_tool"));
  assertEquals(registry.length, 1);
  assertExists(registry.get("test_tool"));
});

Deno.test("ToolRegistry - register multiple", () => {
  const registry = new ToolRegistry();
  registry.registerTools([makeTool("tool1"), makeTool("tool2")]);
  assertEquals(registry.length, 2);
});

Deno.test("ToolRegistry - get by name", () => {
  const registry = new ToolRegistry();
  registry.register(makeTool("my_tool"));
  assertExists(registry.get("my_tool"));
  assertEquals(registry.get("nonexistent"), undefined);
});

Deno.test("ToolRegistry - initial vs deferred", () => {
  const registry = new ToolRegistry();
  registry.register(makeTool("initial", false));
  registry.register(makeTool("deferred", true));

  const initial = registry.getInitialTools();
  assertEquals(initial.length, 1);
  assertEquals(initial[0].name, "initial");

  const deferred = registry.getDeferredTools();
  assertEquals(deferred.length, 1);
  assertEquals(deferred[0].name, "deferred");
});

Deno.test("ToolRegistry - search tools", () => {
  const registry = new ToolRegistry();
  registry.register({
    name: "read_file",
    description: "Read a file from disk",
    input_schema: objectSchema({}, []),
  });
  registry.register({
    name: "write_file",
    description: "Write content to a file",
    input_schema: objectSchema({}, []),
  });
  registry.register({
    name: "execute_command",
    description: "Execute a bash command",
    input_schema: objectSchema({}, []),
  });

  const fileResults = registry.searchTools("file");
  assertEquals(fileResults.length, 2);

  const bashResults = registry.searchTools("bash");
  assertEquals(bashResults.length, 1);
});

Deno.test("ToolRegistry - get all with extra", () => {
  const registry = new ToolRegistry();
  registry.register(makeTool("builtin"));

  const extra = [makeTool("mcp_tool")];
  const all = registry.getAllWithExtra(extra);
  assertEquals(all.length, 2);
});

Deno.test("ToolRegistry - get by category", () => {
  const registry = new ToolRegistry();
  registry.register(makeTool("read_file"));
  registry.register(makeTool("write_file"));
  registry.register(makeTool("execute_command"));

  const fileOps = registry.getByCategory("FileOps");
  assertEquals(fileOps.length, 2);

  const bash = registry.getByCategory("Bash");
  assertEquals(bash.length, 1);
  assertEquals(bash[0].name, "execute_command");
});
