// Example: Quickstart — custom provider and chat
// Shows how to implement the Provider interface and send a chat request.
// Run: deno run deno/examples/core/quickstart.ts

import {
  ChatOptions,
  type ChatResponse,
  createUsage,
  Message,
  type Provider,
  type StreamChunk,
  type Tool,
} from "@rullama/core";

// 1. Implement a custom provider
// In a real app you would call an LLM API (OpenAI, Anthropic, Ollama, etc.).
// Here we create a simple echo provider for demonstration.

class EchoProvider implements Provider {
  readonly name = "echo";

  async chat(
    messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): Promise<ChatResponse> {
    // Find the last user message
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    const text = lastUser?.text() ?? "(no user message)";
    const tokenCount = text.length;

    return {
      message: Message.assistant(`Echo: ${text}`),
      usage: createUsage(tokenCount, tokenCount),
      finish_reason: "stop",
    };
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    // Wrap the non-streaming response for simplicity
    const response = await this.chat(messages, tools, options);
    const text = response.message.text() ?? "";
    yield { type: "text", text };
    yield { type: "usage", usage: response.usage };
    yield { type: "done" };
  }
}

// 2. Use the provider
async function main() {
  console.log("=== Brainwires Quickstart ===");

  const provider = new EchoProvider();
  console.log(`Provider: ${provider.name}`);

  // Build messages
  const messages = [
    Message.system("You are a helpful assistant."),
    Message.user("Hello, custom provider!"),
  ];

  // Configure chat options
  const options = new ChatOptions({ temperature: 0.7, max_tokens: 256 });

  // 3. Send a chat request
  console.log("\n=== Chat Request ===");
  const response = await provider.chat(messages, undefined, options);

  console.log(`Response: ${response.message.text()}`);
  console.log(
    `Tokens: ${response.usage.prompt_tokens} in, ${response.usage.completion_tokens} out`,
  );

  // 4. Demonstrate streaming
  console.log("\n=== Streaming Request ===");
  for await (const chunk of provider.streamChat(messages, undefined, options)) {
    switch (chunk.type) {
      case "text":
        console.log(`  [text] ${chunk.text}`);
        break;
      case "usage":
        console.log(`  [usage] ${chunk.usage.total_tokens} total tokens`);
        break;
      case "done":
        console.log("  [done]");
        break;
    }
  }

  // 5. Show ChatOptions presets
  console.log("\n=== ChatOptions Presets ===");
  const deterministic = ChatOptions.deterministic(100);
  console.log(
    `Deterministic: temp=${deterministic.temperature}, max_tokens=${deterministic.max_tokens}`,
  );

  const factual = ChatOptions.factual(2048);
  console.log(
    `Factual: temp=${factual.temperature}, max_tokens=${factual.max_tokens}`,
  );

  const creative = ChatOptions.creative(4096);
  console.log(
    `Creative: temp=${creative.temperature}, max_tokens=${creative.max_tokens}`,
  );

  console.log(
    "\nDone! Swap EchoProvider with a real LLM provider to get started.",
  );
}

await main();
