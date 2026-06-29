import { assertEquals, assertStringIncludes } from "@std/assert";
import { BashTool } from "./bash.ts";
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
