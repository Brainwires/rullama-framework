import { assert, assertStringIncludes } from "@std/assert";
import { ToolContext } from "@rullama/core";
import { SearchTool } from "./search.ts";

Deno.test("search_code bounds the regex and confines the path", async () => {
  const root = await Deno.makeTempDir();
  try {
    await Deno.writeTextFile(`${root}/a.ts`, "const needle = 1;\n");
    const ctx = new ToolContext({ working_directory: root });
    const ok = await SearchTool.execute("1", "search_code", {
      pattern: "needle",
    }, ctx);
    assertStringIncludes(ok.content, "Matches: 1");
    const long = await SearchTool.execute("2", "search_code", {
      pattern: "a".repeat(201),
    }, ctx);
    assert(long.is_error);
    assertStringIncludes(long.content, "maximum length");
    const escape = await SearchTool.execute("3", "search_code", {
      pattern: "x",
      path: "/etc",
    }, ctx);
    assert(escape.is_error);
    assertStringIncludes(escape.content, "outside the working directory");
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});
