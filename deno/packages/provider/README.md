# @rullama/providers

AI chat provider implementations for the Brainwires Agent Framework. Wraps
multiple AI APIs behind the unified `Provider` interface from
`@rullama/core`.

Equivalent to the Rust `rullama-providers` crate.

## Install

```sh
deno add @rullama/providers
```

## Quick Example

```ts
import { ChatOptions, Message } from "@rullama/core";
import {
  AnthropicChatProvider,
  ChatProviderFactory,
} from "@rullama/provider";

// Direct construction
const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
  "anthropic",
);

const response = await provider.chat(
  [Message.user("Hello!")],
  undefined,
  new ChatOptions({ max_tokens: 512 }),
);
console.log(response.content);

// Or use the factory with a config object
const provider2 = ChatProviderFactory.create({
  provider: "openai",
  model: "gpt-4o",
  api_key: Deno.env.get("OPENAI_API_KEY")!,
});
```

## Supported Providers

| Provider                            | Class                   | Protocol                | API Key Env Var     |
| ----------------------------------- | ----------------------- | ----------------------- | ------------------- |
| Anthropic (Claude)                  | `AnthropicChatProvider` | Anthropic Messages      | `ANTHROPIC_API_KEY` |
| OpenAI                              | `OpenAiChatProvider`    | OpenAI Chat Completions | `OPENAI_API_KEY`    |
| Google (Gemini)                     | `GoogleChatProvider`    | Gemini GenerateContent  | `GOOGLE_API_KEY`    |
| Ollama                              | `OllamaChatProvider`    | Ollama Chat             | None (local)        |
| Groq, Together, Fireworks, Anyscale | `OpenAiChatProvider`    | OpenAI-compatible       | Varies              |

Use `ChatProviderFactory.create()` to construct any provider from a
`ProviderConfig`. The factory dispatches to the correct protocol handler based
on the provider registry.
