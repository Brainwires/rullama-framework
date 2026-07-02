# Providers

The `@rullama/provider` package implements AI chat providers that conform to
the `Provider` interface from `@rullama/core`.

## Provider Interface

Every provider implements `chat` and `streamChat`:

```ts
interface Provider {
  name(): string;
  chat(
    messages: Message[],
    tools?: Tool[],
    options?: ChatOptions,
  ): Promise<ChatResponse>;
  streamChat(
    messages: Message[],
    tools?: Tool[],
    options?: ChatOptions,
  ): AsyncIterable<StreamChunk>;
  maxOutputTokens?(): number;
}
```

## Supported Providers

| Class                     | Service          | Key Features                               |
| ------------------------- | ---------------- | ------------------------------------------ |
| `AnthropicChatProvider`   | Anthropic Claude | Tool use, extended thinking, SSE streaming |
| `OpenAiChatProvider`      | OpenAI GPT       | Chat completions API, function calling     |
| `OpenAiResponsesProvider` | OpenAI Responses | Responses API with built-in tools          |
| `GoogleChatProvider`      | Google Gemini    | Gemini API with tool support               |
| `OllamaChatProvider`      | Ollama (local)   | Local models, no API key required          |
| `BedrockProvider`         | AWS Bedrock      | AWS SigV4 auth, Claude on Bedrock          |
| `VertexAiProvider`        | Google Vertex AI | Google Cloud auth, Gemini on Vertex        |

## Factory Pattern

Use `ChatProviderFactory` to create providers from configuration:

```ts
import { ChatProviderFactory } from "@rullama/provider";

const factory = new ChatProviderFactory();
const provider = factory.create({
  providerType: "anthropic",
  apiKey: Deno.env.get("ANTHROPIC_API_KEY")!,
  model: "claude-sonnet-4-20250514",
});
```

The factory also supports `createProviderConfig` for building typed
configurations, and `PROVIDER_REGISTRY` / `lookup` for querying available
providers.

## SSE Streaming

Providers that support streaming return an `AsyncIterable<StreamChunk>`. The
package includes `parseSSEStream` and `parseNDJSONStream` utilities for parsing
raw HTTP streams.

```ts
for await (const chunk of provider.streamChat(messages, tools, options)) {
  if (chunk.type === "text") {
    process.stdout.write(chunk.text);
  }
}
```

## Rate Limiting

Wrap any HTTP client with `RateLimiter` or `RateLimitedClient` to respect
provider rate limits:

```ts
import { RateLimitedClient } from "@rullama/provider";

const client = new RateLimitedClient({
  requestsPerMinute: 60,
  tokensPerMinute: 100_000,
});
```

See: `../examples/providers/rate_limiting.ts`.

## Model Listing

Use `createModelLister` to dynamically list available models for a provider:

```ts
import { type AvailableModel, createModelLister } from "@rullama/provider";

const lister = createModelLister("openai", apiKey);
const models: AvailableModel[] = await lister.listModels();
```

## Further Reading

- [Getting Started](./getting-started.md) for basic provider setup
- [Agents](./agents.md) for using providers in agent loops
- [Extensibility](./extensibility.md) for implementing custom providers
- Full example: `../examples/providers/provider_factory.ts`
