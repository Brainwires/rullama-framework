import { assertEquals } from "@std/assert";

import {
  type ChatResponse,
  createUsage,
  Message,
  ToolResult,
  type ToolUse,
} from "@rullama/core";

import { CommunicationHub } from "@rullama/agent";
import { FileLockManager, type LockType } from "@rullama/agent";
import { type AgentRuntime, runAgentLoop } from "./runtime.ts";

// ---------------------------------------------------------------------------
// Test agent that completes after N iterations
// ---------------------------------------------------------------------------

class TestAgent implements AgentRuntime {
  private callCount = 0;

  constructor(
    private id: string,
    private maxIters: number,
    private completeAfter: number,
  ) {}

  agentId(): string {
    return this.id;
  }
  maxIterations(): number {
    return this.maxIters;
  }

  // deno-lint-ignore require-await
  async callProvider(): Promise<ChatResponse> {
    const count = this.callCount++;
    const finish_reason = count >= this.completeAfter ? "end_turn" : undefined;
    return {
      message: new Message({
        role: "assistant",
        content: `Response #${count}`,
      }),
      usage: createUsage(10, 20),
      finish_reason,
    };
  }

  extractToolUses(_response: ChatResponse): ToolUse[] {
    return [];
  }

  isCompletion(response: ChatResponse): boolean {
    return response.finish_reason === "end_turn" ||
      response.finish_reason === "stop";
  }

  // deno-lint-ignore require-await
  async executeTool(toolUse: ToolUse): Promise<ToolResult> {
    return new ToolResult(toolUse.id, "ok", false);
  }

  getLockRequirement(_toolUse: ToolUse): [string, LockType] | undefined {
    return undefined;
  }

  onProviderResponse(_response: ChatResponse): void {}
  onToolResult(_toolUse: ToolUse, _result: ToolResult): void {}

  // deno-lint-ignore require-await
  async onCompletion(response: ChatResponse): Promise<string | undefined> {
    if (
      response.finish_reason === "end_turn" ||
      response.finish_reason === "stop"
    ) {
      return typeof response.message.content === "string"
        ? response.message.content
        : "completed";
    }
    return undefined;
  }

  onIterationLimit(iterations: number): string {
    return `Hit iteration limit at ${iterations}`;
  }
}

// ---------------------------------------------------------------------------
// Tool-using agent
// ---------------------------------------------------------------------------

class ToolUsingAgent implements AgentRuntime {
  private callCount = 0;

  constructor(private id: string) {}

  agentId(): string {
    return this.id;
  }
  maxIterations(): number {
    return 10;
  }

  // deno-lint-ignore require-await
  async callProvider(): Promise<ChatResponse> {
    const count = this.callCount++;
    if (count === 0) {
      return {
        message: new Message({
          role: "assistant",
          content: [
            {
              type: "tool_use" as const,
              id: "tool-1",
              name: "read_file",
              input: { path: "/tmp/test.txt" },
            },
          ],
        }),
        usage: createUsage(10, 20),
      };
    }
    return {
      message: new Message({ role: "assistant", content: "Done!" }),
      usage: createUsage(10, 20),
      finish_reason: "end_turn",
    };
  }

  extractToolUses(response: ChatResponse): ToolUse[] {
    const content = response.message.content;
    if (!Array.isArray(content)) return [];
    return content
      // deno-lint-ignore no-explicit-any
      .filter((b: any) => b.type === "tool_use")
      // deno-lint-ignore no-explicit-any
      .map((b: any) => ({ id: b.id, name: b.name, input: b.input }));
  }

  isCompletion(response: ChatResponse): boolean {
    return response.finish_reason === "end_turn" ||
      response.finish_reason === "stop";
  }

  // deno-lint-ignore require-await
  async executeTool(toolUse: ToolUse): Promise<ToolResult> {
    return new ToolResult(toolUse.id, "file contents", false);
  }

  getLockRequirement(toolUse: ToolUse): [string, LockType] | undefined {
    if (toolUse.name === "read_file") {
      // deno-lint-ignore no-explicit-any
      const path = (toolUse.input as any).path as string;
      if (path) return [path, "read"];
    }
    return undefined;
  }

  onProviderResponse(_response: ChatResponse): void {}
  onToolResult(_toolUse: ToolUse, _result: ToolResult): void {}
  // deno-lint-ignore require-await
  async onCompletion(_response: ChatResponse): Promise<string | undefined> {
    return "Done!";
  }
  onIterationLimit(iterations: number): string {
    return `Limit at ${iterations}`;
  }
}

// ---------------------------------------------------------------------------
// Looping agent (triggers loop detection)
// ---------------------------------------------------------------------------

class LoopingAgent implements AgentRuntime {
  constructor(private id: string) {}

  agentId(): string {
    return this.id;
  }
  maxIterations(): number {
    return 100;
  }

  // deno-lint-ignore require-await
  async callProvider(): Promise<ChatResponse> {
    return {
      message: new Message({
        role: "assistant",
        content: [
          {
            type: "tool_use" as const,
            id: "t",
            name: "bash",
            input: { command: "ls" },
          },
        ],
      }),
      usage: createUsage(10, 20),
    };
  }

  extractToolUses(response: ChatResponse): ToolUse[] {
    const content = response.message.content;
    if (!Array.isArray(content)) return [];
    return content
      // deno-lint-ignore no-explicit-any
      .filter((b: any) => b.type === "tool_use")
      // deno-lint-ignore no-explicit-any
      .map((b: any) => ({ id: b.id, name: b.name, input: b.input }));
  }

  isCompletion(_response: ChatResponse): boolean {
    return false;
  }

  // deno-lint-ignore require-await
  async executeTool(toolUse: ToolUse): Promise<ToolResult> {
    return new ToolResult(toolUse.id, "ok", false);
  }

  getLockRequirement(_toolUse: ToolUse): [string, LockType] | undefined {
    return undefined;
  }

  onProviderResponse(_response: ChatResponse): void {}
  onToolResult(_toolUse: ToolUse, _result: ToolResult): void {}
  // deno-lint-ignore require-await
  async onCompletion(_response: ChatResponse): Promise<string | undefined> {
    return undefined;
  }
  onIterationLimit(iterations: number): string {
    return `Limit at ${iterations}`;
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

Deno.test("agent completes successfully", async () => {
  const agent = new TestAgent("test-1", 10, 2);
  const hub = new CommunicationHub();
  const locks = new FileLockManager();

  const result = await runAgentLoop(agent, hub, locks);

  assertEquals(result.success, true);
  assertEquals(result.agentId, "test-1");
  assertEquals(result.iterations, 3);
  assertEquals(result.toolsUsed.length, 0);
});

Deno.test("agent hits iteration limit", async () => {
  const agent = new TestAgent("test-2", 3, 100);
  const hub = new CommunicationHub();
  const locks = new FileLockManager();

  const result = await runAgentLoop(agent, hub, locks);

  assertEquals(result.success, false);
  assertEquals(result.iterations, 3);
  assertEquals(result.output.includes("iteration limit"), true);
});

Deno.test("agent with tool use", async () => {
  const agent = new ToolUsingAgent("test-3");
  const hub = new CommunicationHub();
  const locks = new FileLockManager();

  const result = await runAgentLoop(agent, hub, locks);

  assertEquals(result.success, true);
  assertEquals(result.iterations, 2);
  assertEquals(result.toolsUsed, ["read_file"]);
});

Deno.test("agent unregisters on completion", async () => {
  const agent = new TestAgent("test-4", 10, 0);
  const hub = new CommunicationHub();
  const locks = new FileLockManager();

  await runAgentLoop(agent, hub, locks);

  assertEquals(hub.isRegistered("test-4"), false);
});

Deno.test("loop detection aborts", async () => {
  const agent = new LoopingAgent("loop-agent");
  const hub = new CommunicationHub();
  const locks = new FileLockManager();

  const result = await runAgentLoop(agent, hub, locks);

  assertEquals(result.success, false);
  assertEquals(result.output.includes("Loop detected"), true);
  assertEquals(result.toolsUsed.length, 5);
});
