import { assert, assertEquals } from "@std/assert";
import {
  objectSchema,
  type Tool,
  ToolContext,
  ToolResult,
  type ToolUse,
} from "@rullama/core";
import { CommunicationHub, FileLockManager } from "@rullama/agent";
import {
  EnforcingExecutor,
  reject,
  type ToolExecutor,
} from "@rullama/tool-runtime";
import { AgentContext } from "./context.ts";

class Passthrough implements ToolExecutor {
  calls = 0;
  availableTools(): Tool[] {
    return [{
      name: "read_file",
      description: "r",
      input_schema: objectSchema({}, []),
    }];
  }
  execute(toolUse: ToolUse): Promise<ToolResult> {
    this.calls++;
    return Promise.resolve(ToolResult.success(toolUse.id, "content"));
  }
}
const make = (inner: ToolExecutor, enforcement?: false) =>
  new AgentContext(
    "/app",
    inner,
    new CommunicationHub(),
    new FileLockManager(),
    undefined,
    enforcement,
  );
const use = (path: string): ToolUse => ({
  id: "1",
  name: "read_file",
  input: { path },
});
const ctx = () => new ToolContext({ working_directory: "/app" });

Deno.test("AgentContext enforces the default policy unless opted out", async () => {
  const inner = new Passthrough();
  const context = make(inner);
  assert(context.toolExecutor instanceof EnforcingExecutor);
  const denied = await context.toolExecutor.execute(use("/app/.env"), ctx());
  assert(denied.is_error);
  assertEquals(inner.calls, 0);

  const raw = make(new Passthrough(), false);
  assert(!(raw.toolExecutor instanceof EnforcingExecutor));
  const ok = await raw.toolExecutor.execute(use("/app/.env"), ctx());
  assert(!ok.is_error);
});

Deno.test("preExecuteHook set after construction is consulted on every call", async () => {
  const inner = new Passthrough();
  const context = make(inner);
  const before = await context.toolExecutor.execute(use("/app/a.ts"), ctx());
  assert(!before.is_error);
  context.withPreExecuteHook({
    beforeExecute: () => Promise.resolve(reject("blocked by hook")),
  });
  const after = await context.toolExecutor.execute(use("/app/a.ts"), ctx());
  assert(after.is_error);
  assertEquals(inner.calls, 1);
});
