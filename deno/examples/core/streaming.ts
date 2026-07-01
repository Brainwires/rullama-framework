// Example: Streaming chat responses
// Shows how to implement and consume a streaming provider with chunk handling.
// Run: deno run deno/examples/core/streaming.ts

import {
  ChatOptions,
  type ChatResponse,
  createUsage,
  Message,
  type Provider,
  type StreamChunk,
  type Tool,
} from "@rullama/core";

// 1. Implement a streaming provider
// Simulates word-by-word streaming like a real LLM would produce.

class StreamingDemoProvider implements Provider {
  readonly name = "streaming-demo";

  async chat(
    messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): Promise<ChatResponse> {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    const text = lastUser?.text() ?? "";
    const reply = `Here is my response to: "${text}"`;

    return {
      message: Message.assistant(reply),
      usage: createUsage(text.length, reply.length),
      finish_reason: "stop",
    };
  }

  async *streamChat(
    messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    const prompt = lastUser?.text() ?? "";

    // Simulate streaming word by word
    const words = [
      "The",
      "rullama",
      "framework",
      "provides",
      "a",
      "modular",
      "architecture",
      "for",
      "building",
      "AI",
      "agent",
      "systems",
      "with",
      "full",
      "control",
      "over",
      "providers,",
      "tools,",
      "and",
      "lifecycle",
      "events.",
    ];

    for (const word of words) {
      yield { type: "text", text: word + " " };
    }

    // Emit usage statistics
    yield {
      type: "usage",
      usage: createUsage(prompt.length, words.join(" ").length),
    };

    // Signal completion
    yield { type: "done" };
  }
}

// 2. Implement a provider that streams tool calls

class ToolStreamingProvider implements Provider {
  readonly name = "tool-streaming-demo";

  async chat(
    _messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): Promise<ChatResponse> {
    return {
      message: Message.assistant("I will search for that."),
      usage: createUsage(10, 10),
      finish_reason: "stop",
    };
  }

  async *streamChat(
    _messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    // Stream a text preamble
    yield { type: "text", text: "Let me search for that information. " };

    // Stream a tool use request
    yield { type: "tool_use", id: "call-001", name: "search_codebase" };

    // Stream the tool input as incremental JSON deltas
    yield {
      type: "tool_input_delta",
      id: "call-001",
      partial_json: '{"query":',
    };
    yield {
      type: "tool_input_delta",
      id: "call-001",
      partial_json: ' "authentication"}',
    };

    yield { type: "usage", usage: createUsage(20, 15) };
    yield { type: "done" };
  }
}

// 3. Stream consumer helpers

function collectText(chunks: StreamChunk[]): string {
  return chunks
    .filter((c): c is StreamChunk & { type: "text" } => c.type === "text")
    .map((c) => c.text)
    .join("");
}

function collectToolInput(chunks: StreamChunk[], callId: string): string {
  return chunks
    .filter(
      (c): c is StreamChunk & { type: "tool_input_delta" } =>
        c.type === "tool_input_delta" && c.id === callId,
    )
    .map((c) => c.partial_json)
    .join("");
}

async function main() {
  console.log("=== Streaming Chat Responses ===");

  const messages = [
    Message.system("You are a helpful coding assistant."),
    Message.user("Explain the framework architecture."),
  ];
  const options = new ChatOptions({ temperature: 0.3, max_tokens: 1024 });

  // 4. Consume text streaming
  console.log("\n=== Text Streaming ===");
  const textProvider = new StreamingDemoProvider();
  const allChunks: StreamChunk[] = [];

  const encoder = new TextEncoder();
  await Deno.stdout.write(encoder.encode("  "));
  for await (
    const chunk of textProvider.streamChat(messages, undefined, options)
  ) {
    allChunks.push(chunk);
    switch (chunk.type) {
      case "text":
        await Deno.stdout.write(encoder.encode(chunk.text));
        break;
      case "usage":
        console.log(
          `\n  [usage] ${chunk.usage.prompt_tokens} prompt + ${chunk.usage.completion_tokens} completion = ${chunk.usage.total_tokens} total`,
        );
        break;
      case "done":
        console.log("  [stream complete]");
        break;
    }
  }

  // Collect the full text from chunks
  const fullText = collectText(allChunks);
  console.log(`  Collected: "${fullText.trim()}"`);

  // 5. Consume tool-calling stream
  console.log("\n=== Tool Call Streaming ===");
  const toolProvider = new ToolStreamingProvider();
  const toolChunks: StreamChunk[] = [];

  for await (
    const chunk of toolProvider.streamChat(messages, undefined, options)
  ) {
    toolChunks.push(chunk);
    switch (chunk.type) {
      case "text":
        console.log(`  [text] ${chunk.text}`);
        break;
      case "tool_use":
        console.log(`  [tool_use] id=${chunk.id}, name=${chunk.name}`);
        break;
      case "tool_input_delta":
        console.log(
          `  [tool_input_delta] id=${chunk.id}, json=${chunk.partial_json}`,
        );
        break;
      case "usage":
        console.log(`  [usage] ${chunk.usage.total_tokens} total tokens`);
        break;
      case "done":
        console.log("  [stream complete]");
        break;
    }
  }

  // Reassemble the streamed tool input
  const toolInput = collectToolInput(toolChunks, "call-001");
  console.log(`  Reassembled tool input: ${toolInput}`);

  try {
    const parsed = JSON.parse(toolInput);
    console.log(`  Parsed query: "${parsed.query}"`);
  } catch {
    console.log("  (could not parse tool input)");
  }

  // 6. Compare streaming vs non-streaming
  console.log("\n=== Streaming vs Non-Streaming ===");
  const nonStreamResponse = await textProvider.chat(
    messages,
    undefined,
    options,
  );
  console.log(`Non-streaming: "${nonStreamResponse.message.text()}"`);
  console.log(`Streaming collected: "${fullText.trim()}"`);

  console.log(
    "\nDone! Use streamChat() for real-time token delivery in interactive UIs.",
  );
}

await main();
