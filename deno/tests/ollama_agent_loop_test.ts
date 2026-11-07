/**
 * Cross-package integration test: `@rullama/inference`'s TaskAgent (and so
 * `runAgentLoop`) driving the native `OllamaChatProvider` with tools, against
 * a fake `/api/chat` (no Ollama server or model is involved).
 */

import { assert, assertEquals } from "@std/assert";
import {
  objectSchema,
  Task,
  type Tool,
  ToolResult,
  type ToolUse,
} from "@rullama/core";
import { CommunicationHub, FileLockManager } from "@rullama/agent";
import { AgentContext, TaskAgent } from "@rullama/inference";
import { OllamaChatProvider } from "@rullama/provider";
import type { ToolExecutor } from "@rullama/tool-runtime";

class WeatherTools implements ToolExecutor {
  calls: ToolUse[] = [];
  availableTools(): Tool[] {
    return [{
      name: "get_weather",
      description: "Weather for a city",
      input_schema: objectSchema({ city: { type: "string" } }, ["city"]),
    }];
  }
  execute(toolUse: ToolUse): Promise<ToolResult> {
    this.calls.push(toolUse);
    return Promise.resolve(ToolResult.success(toolUse.id, "Sunny, 21C"));
  }
}

function makeAgent(tools: ToolExecutor): TaskAgent {
  const context = new AgentContext(
    "/tmp",
    tools,
    new CommunicationHub(),
    new FileLockManager(),
    undefined,
    false,
  );
  return new TaskAgent(
    "ollama-agent",
    new Task("t1", "Report the weather in Oslo"),
    new OllamaChatProvider("qwen3", "http://ollama.test"),
    context,
    { maxIterations: 5 },
  );
}

/** Run `fn` with `fetch` replaced by `handler`. */
async function withFetch(
  handler: (init: RequestInit) => Promise<Response>,
  fn: () => Promise<void>,
): Promise<void> {
  const original = globalThis.fetch;
  globalThis.fetch = (_input, init) => handler(init ?? {});
  try {
    await fn();
  } finally {
    globalThis.fetch = original;
  }
}

Deno.test("TaskAgent runs native Ollama tool calls and finishes", async () => {
  // deno-lint-ignore no-explicit-any
  const bodies: any[] = [];
  const replies = [
    {
      message: {
        role: "assistant",
        content: "",
        tool_calls: [{
          function: { name: "get_weather", arguments: { city: "Oslo" } },
        }],
      },
      done: true,
      done_reason: "stop",
    },
    {
      message: { role: "assistant", content: "It is sunny in Oslo." },
      done: true,
      done_reason: "stop",
    },
  ];
  const tools = new WeatherTools();
  await withFetch((init) => {
    bodies.push(JSON.parse(String(init.body)));
    return Promise.resolve(Response.json(replies.shift()));
  }, async () => {
    const result = await makeAgent(tools).execute();
    assert(result.success, result.summary);
    assertEquals(result.summary, "It is sunny in Oslo.");
    assertEquals(result.iterations, 2);
  });

  assertEquals(tools.calls.map((c) => [c.name, c.input]), [
    ["get_weather", { city: "Oslo" }],
  ]);
  assertEquals(bodies[0].tools[0].function.name, "get_weather");
  const sent = bodies[1].messages;
  assertEquals(sent.at(-2).tool_calls[0].function.name, "get_weather");
  assertEquals(sent.at(-1), {
    role: "tool",
    content: "Sunny, 21C",
    tool_call_id: "call_0",
    tool_name: "get_weather",
  });
});

Deno.test("Aborting a TaskAgent cancels the Ollama request in flight", async () => {
  const controller = new AbortController();
  let requestSignal: AbortSignal | undefined;
  await withFetch((init) => {
    requestSignal = init.signal ?? undefined;
    // Never answers; rejects like real fetch once the signal fires.
    return new Promise((_resolve, reject) => {
      init.signal?.addEventListener("abort", () => reject(init.signal!.reason));
      setTimeout(() => controller.abort(), 0);
    });
  }, async () => {
    const result = await makeAgent(new WeatherTools()).execute(
      controller.signal,
    );
    assertEquals(result.success, false);
    assertEquals(result.summary, "Aborted by signal");
  });
  assert(requestSignal?.aborted);
});
