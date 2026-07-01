/**
 * Cross-package integration test: Agent loop with mock provider and tools.
 *
 * Verifies that @rullama/agents runAgentLoop works correctly with
 * @rullama/core types (Provider, Message, Tool, ToolResult, etc.)
 * and the CommunicationHub + FileLockManager infrastructure.
 */

import {
  assert,
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  ChatOptions,
  type ChatResponse,
  createUsage,
  Message,
  type Provider,
  type StreamChunk,
  type Tool,
  ToolResult,
  type ToolUse,
} from "@rullama/core";
import {
  type AgentRuntime,
  CommunicationHub,
  FileLockManager,
  runAgentLoop,
} from "@rullama/agent";
import { ToolRegistry } from "@rullama/tools";

// ---------------------------------------------------------------------------
// Mock provider that returns canned responses
// ---------------------------------------------------------------------------

class MockProvider implements Provider {
  readonly name = "mock";
  private callCount = 0;
  private readonly responses: ChatResponse[];

  constructor(responses: ChatResponse[]) {
    this.responses = responses;
  }

  chat(
    _messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): Promise<ChatResponse> {
    const response = this.responses[this.callCount] ??
      this.responses[this.responses.length - 1];
    this.callCount++;
    return Promise.resolve(response);
  }

  async *streamChat(
    _messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    yield { type: "done" };
  }
}

// ---------------------------------------------------------------------------
// Minimal AgentRuntime backed by MockProvider
// ---------------------------------------------------------------------------

class MockAgentRuntime implements AgentRuntime {
  private provider: MockProvider;
  private toolCallLog: string[] = [];
  private conversationMessages: Message[] = [];
  private maxIter: number;

  constructor(provider: MockProvider, maxIter = 10) {
    this.provider = provider;
    this.maxIter = maxIter;
  }

  agentId(): string {
    return "test-agent";
  }

  maxIterations(): number {
    return this.maxIter;
  }

  async callProvider(): Promise<ChatResponse> {
    return this.provider.chat(
      this.conversationMessages,
      undefined,
      ChatOptions.create(),
    );
  }

  extractToolUses(response: ChatResponse): ToolUse[] {
    const content = response.message.content;
    if (!Array.isArray(content)) return [];
    return content
      .filter((
        b,
      ): b is { type: "tool_use"; id: string; name: string; input: unknown } =>
        b.type === "tool_use"
      )
      .map((b) => ({ id: b.id, name: b.name, input: b.input }));
  }

  isCompletion(response: ChatResponse): boolean {
    return response.finish_reason === "end_turn" ||
      response.finish_reason === "stop";
  }

  async executeTool(toolUse: ToolUse): Promise<ToolResult> {
    this.toolCallLog.push(toolUse.name);
    return ToolResult.success(toolUse.id, `result-of-${toolUse.name}`);
  }

  getLockRequirement(
    _toolUse: ToolUse,
  ): [string, "read" | "write"] | undefined {
    return undefined;
  }

  onProviderResponse(_response: ChatResponse): void {
    // no-op
  }

  onToolResult(_toolUse: ToolUse, _result: ToolResult): void {
    // no-op
  }

  async onCompletion(response: ChatResponse): Promise<string | undefined> {
    const text = response.message.text();
    return text ?? "done";
  }

  onIterationLimit(iterations: number): string {
    return `Hit limit at ${iterations}`;
  }

  getToolCallLog(): string[] {
    return this.toolCallLog;
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

Deno.test("agent loop completes after tool call then end_turn", async () => {
  // Response 1: assistant calls a tool
  const toolCallResponse: ChatResponse = {
    message: new Message({
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "call_1",
          name: "greet",
          input: { name: "world" },
        },
      ],
    }),
    usage: createUsage(100, 50),
    finish_reason: "tool_use",
  };

  // Response 2: assistant completes
  const completionResponse: ChatResponse = {
    message: Message.assistant("Hello, world! Task complete."),
    usage: createUsage(150, 30),
    finish_reason: "end_turn",
  };

  const provider = new MockProvider([toolCallResponse, completionResponse]);
  const runtime = new MockAgentRuntime(provider);
  const hub = new CommunicationHub();
  const lockManager = new FileLockManager();

  const result = await runAgentLoop(runtime, hub, lockManager);

  assertEquals(result.agentId, "test-agent");
  assertEquals(result.success, true);
  assertEquals(result.iterations, 2);
  assertEquals(result.toolsUsed, ["greet"]);
  assertEquals(result.output, "Hello, world! Task complete.");
});

Deno.test("agent loop hits iteration limit", async () => {
  // Provider always returns a tool call, never completes
  const toolCallResponse: ChatResponse = {
    message: new Message({
      role: "assistant",
      content: [
        { type: "tool_use", id: "call_loop", name: "noop", input: {} },
      ],
    }),
    usage: createUsage(100, 50),
    finish_reason: "tool_use",
  };

  const provider = new MockProvider([toolCallResponse]);
  // Max 3 iterations — loop detection fires at window=5, so 3 is safe
  const runtime = new MockAgentRuntime(provider, 3);
  const hub = new CommunicationHub();
  const lockManager = new FileLockManager();

  const result = await runAgentLoop(runtime, hub, lockManager);

  assertEquals(result.success, false);
  assert(
    result.output.includes("3"),
    "output should mention the iteration count",
  );
  assertEquals(result.iterations, 3);
});

Deno.test("tool registry integration - register and lookup", () => {
  const registry = new ToolRegistry();
  const mockTool: Tool = {
    name: "greet",
    description: "Says hello",
    input_schema: {
      type: "object",
      properties: { name: { type: "string" } },
      required: ["name"],
    },
  };

  registry.register(mockTool);

  assertEquals(registry.length, 1);
  const found = registry.get("greet");
  assertEquals(found?.name, "greet");
  assertEquals(found?.description, "Says hello");
});
