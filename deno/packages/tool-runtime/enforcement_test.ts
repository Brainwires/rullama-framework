import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import {
  objectSchema,
  type Tool,
  ToolContext,
  ToolResult,
  type ToolUse,
} from "@rullama/core";
import {
  AgentCapabilities,
  PolicyActions,
  PolicyEngine,
} from "@rullama/permission";
import type { ToolExecutor } from "./executor.ts";
import { allow, reject } from "./executor.ts";
import {
  domainForToolUse,
  enforce,
  type EnforcementEvent,
  EnforcingExecutor,
  filePathForToolUse,
  gitOperationForToolUse,
  policyRequestForToolUse,
} from "./enforcement.ts";

const tool = (name: string, requires_approval = false): Tool => ({
  name,
  description: name,
  input_schema: objectSchema({}, []),
  requires_approval,
});

/** Records every call and answers with a fixed payload. */
class RecordingExecutor implements ToolExecutor {
  calls: ToolUse[] = [];
  constructor(private readonly payload = "ok") {}
  availableTools(): Tool[] {
    return [
      tool("read_file"),
      tool("write_file", true),
      tool("execute_command", true),
      tool("fetch_url"),
      tool("git_reset", true),
    ];
  }
  execute(toolUse: ToolUse): Promise<ToolResult> {
    this.calls.push(toolUse);
    return Promise.resolve(ToolResult.success(toolUse.id, this.payload));
  }
}

const use = (name: string, input: Record<string, unknown> = {}): ToolUse => ({
  id: `use-${name}`,
  name,
  input,
});
const ctx = () => new ToolContext({ working_directory: "/app" });

Deno.test("request derivation: path, domain and git operation come from the input", () => {
  assertEquals(
    filePathForToolUse(use("read_file", { path: "src/a.ts" })),
    "src/a.ts",
  );
  assertEquals(
    filePathForToolUse(use("read_file", { file_path: "b.ts" })),
    "b.ts",
  );
  assertEquals(
    domainForToolUse(use("fetch_url", { url: "https://example.com/x" })),
    "example.com",
  );
  assertEquals(
    domainForToolUse(use("fetch_url", { url: "not a url" })),
    undefined,
  );
  assertEquals(gitOperationForToolUse(use("git_push")), "Push");
  assertEquals(
    gitOperationForToolUse(use("git_push", { force: true })),
    "ForcePush",
  );
  assertEquals(gitOperationForToolUse(use("read_file")), undefined);
  const req = policyRequestForToolUse(use("write_file", { path: "/app/x" }));
  assertEquals(req.tool_category, "FileWrite");
  assertEquals(req.file_path, "/app/x");
});

Deno.test("default policy denies reading a .env file and lets ordinary reads through", async () => {
  const inner = new RecordingExecutor();
  const exec = enforce(inner);
  const denied = await exec.execute(
    use("read_file", { path: "/app/.env" }),
    ctx(),
  );
  assert(denied.is_error);
  assertStringIncludes(denied.content, ".env");
  assertEquals(inner.calls.length, 0);

  const ok = await exec.execute(
    use("read_file", { path: "/app/src/a.ts" }),
    ctx(),
  );
  assert(!ok.is_error);
  assertEquals(ok.content, "ok");
  assertEquals(inner.calls.length, 1);
});

Deno.test("read-only mode admits only read/search/planning tools", async () => {
  const inner = new RecordingExecutor();
  const exec = new EnforcingExecutor(inner, { mode: "read-only" });
  const write = await exec.execute(
    use("write_file", { path: "/app/a" }),
    ctx(),
  );
  assert(write.is_error);
  assertStringIncludes(write.content, "read-only");
  const read = await exec.execute(use("read_file", { path: "/app/a" }), ctx());
  assert(!read.is_error);
});

Deno.test("capabilities narrow tools, paths and domains", async () => {
  const inner = new RecordingExecutor();
  const exec = new EnforcingExecutor(inner, {
    capabilities: AgentCapabilities.readOnly(),
  });
  const bash = await exec.execute(
    use("execute_command", { command: "ls" }),
    ctx(),
  );
  assert(bash.is_error);
  assertStringIncludes(bash.content, "capabilities");
  const web = await exec.execute(
    use("fetch_url", { url: "https://evil.example/x" }),
    ctx(),
  );
  assert(web.is_error);
  assertEquals(inner.calls.length, 0);
});

Deno.test("full mode skips capability checks and grants approvals, but policy denies still hold", async () => {
  const inner = new RecordingExecutor();
  const exec = new EnforcingExecutor(inner, {
    mode: "full",
    capabilities: AgentCapabilities.readOnly(),
  });
  const reset = await exec.execute(use("git_reset"), ctx());
  assert(!reset.is_error, reset.content);
  const env = await exec.execute(
    use("read_file", { path: "/app/.env" }),
    ctx(),
  );
  assert(env.is_error);
});

Deno.test("require_approval policies consult the approval handler", async () => {
  const inner = new RecordingExecutor();
  const asked: string[] = [];
  const events: EnforcementEvent[] = [];
  const exec = new EnforcingExecutor(inner, {
    approve: (toolUse) => {
      asked.push(toolUse.name);
      return toolUse.name !== "git_reset";
    },
    onDecision: (e) => events.push(e),
  });
  const reset = await exec.execute(use("git_reset"), ctx());
  assert(reset.is_error);
  assertStringIncludes(reset.content, "requires approval");
  assertEquals(asked, ["git_reset"]);
  assertEquals(events.at(-1)?.outcome, "rejected");

  // A tool flagged requires_approval is routed through the handler in auto mode.
  const write = await exec.execute(
    use("write_file", { path: "/app/a" }),
    ctx(),
  );
  assert(!write.is_error);
  assertEquals(asked, ["git_reset", "write_file"]);
  assertEquals(events.at(-1)?.outcome, "approved");
});

Deno.test("without an approval handler, flagged tools still run in auto mode", async () => {
  const inner = new RecordingExecutor();
  const exec = enforce(inner);
  const write = await exec.execute(
    use("write_file", { path: "/app/a" }),
    ctx(),
  );
  assert(!write.is_error);
  const reset = await exec.execute(use("git_reset"), ctx());
  assert(
    reset.is_error,
    "policy require_approval with no handler is a rejection",
  );
});

Deno.test("a custom policy engine's deny_with_message is surfaced verbatim", async () => {
  const policy = new PolicyEngine();
  policy.setDefaultAction(PolicyActions.DenyWithMessage("nope"));
  const exec = new EnforcingExecutor(new RecordingExecutor(), { policy });
  const res = await exec.execute(use("read_file", { path: "/app/a" }), ctx());
  assert(res.is_error);
  assertEquals(res.content, "nope");
});

Deno.test("pre-execute hooks can reject after policy allows", async () => {
  const inner = new RecordingExecutor();
  const exec = new EnforcingExecutor(inner, {
    preHooks: [
      { beforeExecute: () => Promise.resolve(allow()) },
      {
        beforeExecute: (u) =>
          Promise.resolve(u.name === "read_file" ? reject("not now") : allow()),
      },
    ],
  });
  const res = await exec.execute(use("read_file", { path: "/app/a" }), ctx());
  assert(res.is_error);
  assertStringIncludes(res.content, "not now");
  assertEquals(inner.calls.length, 0);
});

Deno.test("results are redacted, and web content is wrapped as external", async () => {
  const secret = "token sk-ant-abcdefghijklmnopqrstuvwxyz0123456789";
  const inner = new RecordingExecutor(secret);
  const exec = enforce(inner);
  const read = await exec.execute(use("read_file", { path: "/app/a" }), ctx());
  assert(!read.content.includes("sk-ant-abcdefghijklmnopqrstuvwxyz0123456789"));
  const web = await exec.execute(
    use("fetch_url", { url: "https://example.com" }),
    ctx(),
  );
  assertStringIncludes(web.content, "[EXTERNAL CONTENT");
  assertStringIncludes(web.content, "[END EXTERNAL CONTENT]");

  const raw = new EnforcingExecutor(inner, { filterOutput: false });
  const untouched = await raw.execute(
    use("read_file", { path: "/app/a" }),
    ctx(),
  );
  assertEquals(untouched.content, secret);
});

Deno.test("enforce() is idempotent and availableTools passes through", () => {
  const inner = new RecordingExecutor();
  const once = enforce(inner);
  assert(enforce(once) === once);
  assertEquals(once.availableTools().length, inner.availableTools().length);
});
