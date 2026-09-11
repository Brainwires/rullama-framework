import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  objectSchema,
  type Tool,
  ToolContext,
  ToolResult,
} from "@rullama/core";
import {
  BuiltinToolExecutor,
  createBuiltinExecutor,
  type ToolProvider,
} from "./executor.ts";

const echo: ToolProvider = {
  getTools(): Tool[] {
    return [{
      name: "echo",
      description: "echo",
      input_schema: objectSchema({}, []),
    }];
  },
  execute(id, name, input) {
    return Promise.resolve(
      ToolResult.success(id, `${name}:${JSON.stringify(input)}`),
    );
  },
};
const ctx = () => new ToolContext({ working_directory: "/app" });

Deno.test("BuiltinToolExecutor dispatches by tool name and reports unknown tools", async () => {
  const exec = new BuiltinToolExecutor([echo]);
  assertEquals(exec.availableTools().map((t) => t.name), ["echo"]);
  const ok = await exec.execute(
    { id: "1", name: "echo", input: { a: 1 } },
    ctx(),
  );
  assertEquals(ok.content, 'echo:{"a":1}');
  const bad = await exec.execute({ id: "2", name: "nope", input: {} }, ctx());
  assert(bad.is_error);
});

Deno.test("a tool name owned by two providers is rejected at construction", () => {
  assertThrows(
    () => new BuiltinToolExecutor([echo, echo]),
    Error,
    "registered twice",
  );
});

Deno.test("withDefaults registers the built-in tool set", () => {
  const names = BuiltinToolExecutor.withDefaults().availableTools().map((t) =>
    t.name
  );
  for (
    const expected of [
      "execute_command",
      "read_file",
      "write_file",
      "git_status",
      "search_code",
      "fetch_url",
    ]
  ) {
    assert(names.includes(expected), `missing ${expected}`);
  }
});

Deno.test("createBuiltinExecutor enforces the default policy", async () => {
  const exec = createBuiltinExecutor({ providers: [echo] });
  const denied = await exec.execute(
    { id: "3", name: "read_file", input: { path: "/app/.env" } },
    ctx(),
  );
  assert(denied.is_error);
  const ok = await exec.execute({ id: "4", name: "echo", input: {} }, ctx());
  assert(!ok.is_error);
});
