# Providers

The `@rullama/provider` package implements AI chat providers that conform to the
`Provider` interface from `@rullama/core`.

## Provider Interface

Every provider has a `name` and implements `chat` and `streamChat`; all three
arguments are positional (pass `undefined` for no tools):

```ts
interface Provider {
  readonly name: string;
  maxOutputTokens?(): number | undefined;
  chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse>;
  streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk>;
}
```

`ChatResponse` is `{ message: Message; usage: Usage; finish_reason?: string }`
-- read the text with `response.message.text()`. `ChatOptions` carries
`temperature`, `max_tokens`, `top_p`, `stop`, `system` and (new in v0.12.0)
`model`, which overrides the provider's configured model for one call; build it
with `new ChatOptions({...})`, `ChatOptions.create().setModel(...)`, or the
presets `ChatOptions.deterministic(n)` / `.factual(n)` / `.creative(n)`.

## Supported Providers

| Class                     | Service          | Key Features                                                               |
| ------------------------- | ---------------- | -------------------------------------------------------------------------- |
| `AnthropicChatProvider`   | Anthropic Claude | Tool use, extended thinking, SSE streaming                                 |
| `OpenAiChatProvider`      | OpenAI GPT       | Chat Completions API; also Groq / Together / Fireworks / Anyscale          |
| `OpenAiResponsesProvider` | OpenAI Responses | Responses API; stateless by default, chain with `withPreviousResponseId()` |
| `GoogleChatProvider`      | Google Gemini    | Gemini API with tool support                                               |
| `OllamaChatProvider`      | Ollama (local)   | Local models, no API key required                                          |
| `BedrockProvider`         | AWS Bedrock      | AWS SigV4 auth                                                             |
| `VertexAiProvider`        | Google Vertex AI | Service-account JWT auth, Gemini on Vertex                                 |

```ts
import { ChatOptions, Message } from "@rullama/core";
import {
  AnthropicChatProvider,
  OpenAiResponsesProvider,
} from "@rullama/provider";

const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
);
const response = await provider.chat(
  [Message.user("What is the Deno runtime?")],
  undefined,
  new ChatOptions({ max_tokens: 1024 }),
);
console.log(response.message.text());

// Responses API: explicit conversation chaining
const responses = new OpenAiResponsesProvider(
  Deno.env.get("OPENAI_API_KEY")!,
  "gpt-4.1",
);
const first = await responses.chat(
  [Message.user("Hi")],
  undefined,
  new ChatOptions(),
);
const chained = responses.withPreviousResponseId(responses.getLastResponseId());
console.log(first.message.text(), chained.name);
```

## Factory Pattern

`ChatProviderFactory.create` (static) builds any provider from a
`ProviderConfig` by consulting `PROVIDER_REGISTRY`:

```ts
import { ChatProviderFactory, createProviderConfig } from "@rullama/provider";

const config = createProviderConfig("anthropic", "claude-sonnet-4-20250514");
config.api_key = Deno.env.get("ANTHROPIC_API_KEY");
const provider = ChatProviderFactory.create(config);
```

`ProviderConfig` fields: `provider` (`ProviderType`), `model`, `api_key?`,
`base_url?`, `options?`. Query the registry with `lookup(providerType)`,
`requiresApiKey`, `defaultModel`, `parseProviderType`.

See: `../examples/providers/provider_factory.ts`.

## SSE Streaming

`streamChat` returns an `AsyncIterable<StreamChunk>`; chunk types are `text`,
`tool_use`, `tool_input_delta`, `tool_call`, `usage` and `done`. The package
also exports `parseSSEStream` and `parseNDJSONStream` for raw HTTP streams.

```ts
for await (const chunk of provider.streamChat(messages, undefined, options)) {
  if (chunk.type === "text") {
    await Deno.stdout.write(new TextEncoder().encode(chunk.text));
  }
}
```

See: `../examples/core/streaming.ts`.

## Rate Limiting

`RateLimiter` (token bucket, requests per minute) and `RateLimitedClient` wrap
any async function:

```ts
import { RateLimitedClient } from "@rullama/provider";

const limited = new RateLimitedClient(
  (endpoint: string) => fetch(endpoint).then((r) => r.text()),
  { requestsPerMinute: 60 },
);
const body = await limited.execute("https://example.com");
```

See: `../examples/providers/rate_limiting.ts`.

## Model Listing

`createModelLister(providerType, apiKey?, baseUrl?)` returns a `ModelLister`:

```ts
import { type AvailableModel, createModelLister } from "@rullama/provider";

const lister = createModelLister("openai", Deno.env.get("OPENAI_API_KEY"));
const models: AvailableModel[] = await lister.listModels();
```

## Call policies

`@rullama/call-policy` wraps any `Provider` with composable decorators --
`RetryProvider`, `BudgetProvider`, `CircuitBreakerProvider`, `CachedProvider`
(over a `CacheBackend` such as `MemoryCache`, whose optional constructor
argument is the maximum number of entries).

## Speech

`@rullama/provider-speech` ships the TTS/STT/ASR HTTP clients (Azure Speech,
Cartesia, Deepgram, ElevenLabs, Fish Audio, Google TTS, Murf); they are
independent of the chat providers.

## Further Reading

- [Getting Started](./getting-started.md) for basic provider setup
- [Agents](./agents.md) for using providers in agent loops
- [Extensibility](./extensibility.md) for implementing custom providers
- Full example: `../examples/providers/provider_factory.ts`
