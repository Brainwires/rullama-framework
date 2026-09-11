import { assert, assertStringIncludes } from "@std/assert";
import { ToolContext } from "@rullama/core";
import { WebTool } from "./web.ts";

const ctx = () => new ToolContext({ working_directory: Deno.cwd() });

Deno.test("fetch_url refuses loopback, link-local and non-http destinations", async () => {
  for (
    const url of [
      "http://127.0.0.1:11434/api/tags",
      "http://localhost/",
      "http://169.254.169.254/latest/meta-data/",
      "file:///etc/passwd",
      "http://[::1]/",
    ]
  ) {
    const res = await WebTool.execute("1", "fetch_url", { url }, ctx());
    assert(res.is_error, `${url} should be refused`);
    assertStringIncludes(res.content, "refusing to fetch");
  }
});
