# @rullama/provider

AI chat provider implementations for rullama. Wraps multiple AI APIs behind the
unified `Provider` interface from `@rullama/core`, using plain `fetch()` — no
vendor SDKs.

Equivalent to the Rust `rullama-providers` crate.

## Install

```sh
deno add @rullama/provider
```

## Quick Example

```ts
import { ChatOptions, Message } from "@rullama/core";
import { AnthropicChatProvider, ChatProviderFactory } from "@rullama/provider";

// Direct construction: (apiKey, model, providerName?, baseUrl?)
const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
);

// chat(messages, tools | undefined, options) -> ChatResponse
const response = await provider.chat(
  [Message.user("Hello!")],
  undefined,
  new ChatOptions({ max_tokens: 512 }),
);
// ChatResponse = { message: Message; usage: Usage; finish_reason?: string }
console.log(response.message.content);
console.log(response.usage);

// Streaming: an AsyncIterable<StreamChunk>
for await (
  const chunk of provider.streamChat(
    [Message.user("Count to three.")],
    undefined,
    new ChatOptions(),
  )
) {
  if (chunk.type === "text") console.log(chunk.text);
}

// Or use the factory with a ProviderConfig
const provider2 = ChatProviderFactory.create({
  provider: "openai",
  model: "gpt-5-mini",
  api_key: Deno.env.get("OPENAI_API_KEY")!,
});
console.log(provider2.name); // "openai"
```

`ProviderConfig` fields are `provider`, `model`, `api_key?`, `base_url?` and
`options?`. Bedrock and Vertex AI read their credentials from `options`
(`region`, `access_key_id` / `secret_access_key`, or `project_id` /
`credentials`); Bedrock falls back to the `AWS_*` environment variables.

## Supported Providers

| Provider                            | Class                     | Protocol                | Credentials                             |
| ----------------------------------- | ------------------------- | ----------------------- | --------------------------------------- |
| Anthropic (Claude)                  | `AnthropicChatProvider`   | Anthropic Messages      | `ANTHROPIC_API_KEY`                     |
| OpenAI                              | `OpenAiChatProvider`      | OpenAI Chat Completions | `OPENAI_API_KEY`                        |
| OpenAI (Responses API)              | `OpenAiResponsesProvider` | OpenAI Responses        | `OPENAI_API_KEY`                        |
| Google (Gemini)                     | `GoogleChatProvider`      | Gemini GenerateContent  | `GOOGLE_API_KEY`                        |
| Google Vertex AI                    | `VertexAiProvider`        | Gemini GenerateContent  | Service-account JSON (JWT, no SDK)      |
| AWS Bedrock (Claude models)         | `BedrockProvider`         | Anthropic Messages      | AWS access key + secret (SigV4, no SDK) |
| Ollama                              | `OllamaChatProvider`      | Ollama Chat             | None (local, `http://localhost:11434`)  |
| Groq, Together, Fireworks, Anyscale | `OpenAiChatProvider`      | OpenAI-compatible       | Provider API key via `api_key`          |

Use `ChatProviderFactory.create()` to construct any provider from a
`ProviderConfig`. The factory dispatches on the `chat_protocol` recorded for the
provider in `PROVIDER_REGISTRY` (see `lookup()`); Bedrock and Vertex AI are
routed to their own classes because they use different signing and endpoints.

## Also exported

- `parseSSEStream` / `parseNDJSONStream` — streaming body parsers used by the
  providers.
- `RateLimiter` / `RateLimitedClient` — re-exported from `@rullama/core`.
- `createModelLister` / `inferOpenaiCapabilities` / `isChatCapable` — list a
  provider's available models (`AvailableModel`, `ModelCapability`).
- `parseProviderType` / `defaultModel` / `requiresApiKey` — helpers over
  `ProviderType`.
