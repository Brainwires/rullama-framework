// Example: Tool Execution
// Demonstrates the ToolExecutor and ToolPreHook interfaces for executing tools
// with pre-execution validation hooks (safety guards, audit logging).
// Run: deno run deno/examples/tool-system/tool_execution.ts

import { ToolContext, ToolResult } from "@rullama/core";
import type { Tool, ToolUse } from "@rullama/core";
import { allow, reject } from "@rullama/tool-runtime";
import type {
  PreHookDecision,
  ToolExecutor,
  ToolPreHook,
} from "@rullama/tool-runtime";

// 1. Define a safety-check hook
// This hook blocks destructive tools and rejects calls with suspicious input patterns.

class SafetyGuardHook implements ToolPreHook {
  private blockedTools: string[];
  private blockedPatterns: string[];

  constructor() {
    this.blockedTools = ["delete_file", "deploy_production"];
    this.blockedPatterns = ["rm -rf", "DROP TABLE"];
  }

  async beforeExecute(
    toolUse: ToolUse,
    _context: ToolContext,
  ): Promise<PreHookDecision> {
    // Check if the tool itself is blocked
    if (this.blockedTools.includes(toolUse.name)) {
      return reject(`Tool '${toolUse.name}' is blocked by safety policy.`);
    }

    // Check if the input contains any blocked patterns
    const inputStr = JSON.stringify(toolUse.input);
    for (const pattern of this.blockedPatterns) {
      if (inputStr.includes(pattern)) {
        return reject(`Input contains blocked pattern: '${pattern}'`);
      }
    }

    // Otherwise, allow the call to proceed
    return allow();
  }
}

// 2. Define a logging/audit hook

class AuditLogHook implements ToolPreHook {
  async beforeExecute(
    toolUse: ToolUse,
    context: ToolContext,
  ): Promise<PreHookDecision> {
    console.log(
      `[AUDIT] Tool='${toolUse.name}' id='${toolUse.id}' ` +
        `cwd='${context.working_directory}' input=${
          JSON.stringify(toolUse.input)
        }`,
    );
    return allow();
  }
}

// 3. Define a simple mock executor that uses hooks

class MockExecutor implements ToolExecutor {
  private hooks: ToolPreHook[];
  private tools_: Tool[];

  constructor(hooks: ToolPreHook[], tools: Tool[]) {
    this.hooks = hooks;
    this.tools_ = tools;
  }

  async execute(toolUse: ToolUse, context: ToolContext): Promise<ToolResult> {
    // Run all pre-hooks
    for (const hook of this.hooks) {
      const decision = await hook.beforeExecute(toolUse, context);
      if (decision.type === "Reject") {
        return ToolResult.error(toolUse.id, `Rejected: ${decision.reason}`);
      }
    }
    // Simulate execution
    return ToolResult.success(
      toolUse.id,
      `Executed ${toolUse.name} successfully`,
    );
  }

  availableTools(): Tool[] {
    return this.tools_;
  }
}

// 4. Helper to build a mock ToolUse

function mockToolUse(name: string, input: Record<string, unknown>): ToolUse {
  return { id: `call_${name}`, name, input };
}

async function main() {
  console.log("=== Tool Execution Example ===\n");

  const safety = new SafetyGuardHook();
  const audit = new AuditLogHook();
  const ctx = new ToolContext({
    working_directory: "/home/user/project",
    user_id: "demo-user",
  });

  // Scenario A: A safe tool call -- should be allowed
  console.log("Scenario A (read_file):");
  const safeCall = mockToolUse("read_file", {
    path: "/home/user/project/README.md",
  });
  const decisionA = await safety.beforeExecute(safeCall, ctx);
  console.log(`  Decision: ${JSON.stringify(decisionA)}`);
  await audit.beforeExecute(safeCall, ctx);

  // Scenario B: A blocked tool -- should be rejected
  console.log("\nScenario B (delete_file):");
  const blockedCall = mockToolUse("delete_file", {
    path: "/etc/important.conf",
  });
  const decisionB = await safety.beforeExecute(blockedCall, ctx);
  console.log(`  Decision: ${JSON.stringify(decisionB)}`);

  // Scenario C: Input contains a dangerous pattern -- should be rejected
  console.log("\nScenario C (execute_command with 'rm -rf'):");
  const dangerousInput = mockToolUse("execute_command", {
    command: "rm -rf /important/data",
  });
  const decisionC = await safety.beforeExecute(dangerousInput, ctx);
  console.log(`  Decision: ${JSON.stringify(decisionC)}`);

  // Scenario D: A safe command -- should be allowed
  console.log("\nScenario D (execute_command with 'ls -la'):");
  const safeCmd = mockToolUse("execute_command", { command: "ls -la" });
  const decisionD = await safety.beforeExecute(safeCmd, ctx);
  console.log(`  Decision: ${JSON.stringify(decisionD)}`);

  // 5. Full executor demonstration with hooks
  console.log("\n=== Full Executor with Hooks ===\n");
  const executor = new MockExecutor([audit, safety], []);

  const calls: ToolUse[] = [
    mockToolUse("read_file", { path: "src/main.ts" }),
    mockToolUse("delete_file", { path: "/etc/passwd" }),
    mockToolUse("execute_command", { command: "ls -la" }),
  ];

  for (const call of calls) {
    const result = await executor.execute(call, ctx);
    console.log(
      `  ${call.name}: ${result.is_error ? "ERROR" : "OK"} - ${result.content}`,
    );
  }

  console.log("\nAll scenarios passed.");
}

await main();
