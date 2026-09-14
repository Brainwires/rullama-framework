import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { BashTool, scrubEnv } from "./bash.ts";
import { ToolContext } from "@rullama/core";

Deno.test("BashTool - getTools returns 1 tool", () => {
  const tools = BashTool.getTools();
  assertEquals(tools.length, 1);
  assertEquals(tools[0].name, "execute_command");
  assertEquals(tools[0].requires_approval, true);
});

Deno.test("BashTool - execute simple command", async () => {
  const context = new ToolContext({ working_directory: Deno.cwd() });
  const result = await BashTool.execute(
    "bash-123",
    "execute_command",
    { command: "echo 'Hello World'", timeout: 5 },
    context,
  );
  assertEquals(result.is_error, false);
  assertStringIncludes(result.content, "Hello World");
  assertStringIncludes(result.content, "Exit Code: 0");
});

Deno.test("BashTool - unknown tool name", async () => {
  const context = new ToolContext({ working_directory: Deno.cwd() });
  const result = await BashTool.execute(
    "bash-456",
    "unknown_tool",
    { command: "echo test" },
    context,
  );
  assertEquals(result.is_error, true);
});

Deno.test("BashTool - validate dangerous command", () => {
  try {
    BashTool.validateCommand("rm -rf /");
    throw new Error("Should have thrown");
  } catch (e) {
    assertStringIncludes((e as Error).message, "dangerous");
  }
});

Deno.test("BashTool - validate safe command", () => {
  // Should not throw
  BashTool.validateCommand("ls -la");
});

Deno.test("BashTool - isInteractiveCommand", () => {
  assertEquals(BashTool.isInteractiveCommand("vim file.txt"), true);
  assertEquals(BashTool.isInteractiveCommand("sudo vim file.txt"), true);
  assertEquals(BashTool.isInteractiveCommand("ls -la"), false);
  assertEquals(BashTool.isInteractiveCommand("cargo build"), false);
});

Deno.test("BashTool - smart limits for cargo build", () => {
  const limits = BashTool.getSmartLimits("cargo build");
  assertEquals(limits.maxLines, 80);
  assertEquals(limits.outputMode, "head");
});

Deno.test("BashTool - transform command with no limits", () => {
  const result = BashTool.transformCommand("echo test", {
    outputMode: "full",
    stderrMode: "separate",
    autoLimit: false,
  });
  assertEquals(result, "echo test");
});

Deno.test("BashTool - transform command with head limit", () => {
  const result = BashTool.transformCommand("cat file.txt", {
    maxLines: 50,
    outputMode: "head",
    stderrMode: "separate",
    autoLimit: false,
  });
  assertStringIncludes(result, "head -n 50");
});

Deno.test("BashTool - the timeout is enforced", async () => {
  const context = new ToolContext({ working_directory: Deno.cwd() });
  const started = Date.now();
  const result = await BashTool.execute(
    "t",
    "execute_command",
    { command: "sleep 5", timeout: 1, output_mode: "full" },
    context,
  );
  assert(result.is_error, result.content);
  assertStringIncludes(result.content, "timed out");
  assert(Date.now() - started < 4_000, "must not wait for the sleep to finish");
});

Deno.test("BashTool - credential-looking environment variables are withheld", async () => {
  const context = new ToolContext({ working_directory: Deno.cwd() });
  Deno.env.set("RULLAMA_TEST_API_KEY", "sk-should-not-leak");
  Deno.env.set("RULLAMA_TEST_PLAIN", "visible");
  try {
    const result = await BashTool.execute(
      "e",
      "execute_command",
      { command: "printenv | sort", output_mode: "full", timeout: 10 },
      context,
    );
    assert(!result.is_error, result.content);
    assert(!result.content.includes("sk-should-not-leak"));
    assertStringIncludes(result.content, "RULLAMA_TEST_PLAIN=visible");
  } finally {
    Deno.env.delete("RULLAMA_TEST_API_KEY");
    Deno.env.delete("RULLAMA_TEST_PLAIN");
  }
});

Deno.test("scrubEnv drops secrets and keeps the rest", () => {
  const out = scrubEnv({
    PATH: "/bin",
    HOME: "/h",
    ANTHROPIC_API_KEY: "x",
    OPENAI_API_KEY: "x",
    AWS_SECRET_ACCESS_KEY: "x",
    AWS_REGION: "us-east-1",
    GITHUB_TOKEN: "x",
    DB_PASSWORD: "x",
    NODE_ENV: "test",
  });
  assertEquals(Object.keys(out).sort(), ["HOME", "NODE_ENV", "PATH"]);
});

Deno.test("BashTool - output is capped", async () => {
  const context = new ToolContext({ working_directory: Deno.cwd() });
  const result = await BashTool.execute(
    "c",
    "execute_command",
    {
      command: "head -c 600000 /dev/zero | tr '\\0' 'x'",
      output_mode: "full",
      timeout: 10,
    },
    context,
  );
  assert(!result.is_error, result.content);
  assertStringIncludes(result.content, "[output truncated at");
  assert(result.content.length < 300_000);
});
