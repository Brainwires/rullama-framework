# brainwires-providers (DEPRECATED)

This crate has been **renamed** and **split** as of 0.10.1.

| Old (`brainwires-providers` 0.10.x) | New |
|---|---|
| LLM chat clients (Anthropic, OpenAI Chat + Responses, Google Gemini, Ollama, Bedrock, Vertex AI, local llama.cpp / Candle) | [`brainwires-provider`](https://crates.io/crates/brainwires-provider) (singular) |
| Speech: Azure Speech, Cartesia, Deepgram, ElevenLabs, Fish, Google TTS, Murf, browser `web_speech` | [`brainwires-provider-speech`](https://crates.io/crates/brainwires-provider-speech) |

There is no re-export shim — depending on this crate gets you nothing.

## Migration

### Cargo.toml

```toml
# Before
brainwires-providers = "0.10"

# After — pick whichever stack you need:
brainwires-provider = "0.11"          # LLM chat clients
brainwires-provider-speech = "0.11"   # TTS / STT
```

### Imports

| Before | After |
|---|---|
| `brainwires_providers::AnthropicClient` (and other LLM clients) | `brainwires_provider::AnthropicClient` |
| `brainwires_providers::OpenAiClient` / `OpenAiChatProvider` / `OpenAiResponsesProvider` | `brainwires_provider::*` |
| `brainwires_providers::GoogleClient` / `GoogleChatProvider` | `brainwires_provider::*` |
| `brainwires_providers::OllamaProvider` / `OllamaChatProvider` | `brainwires_provider::*` |
| `brainwires_providers::ChatProviderFactory` | `brainwires_provider::ChatProviderFactory` |
| `brainwires_providers::azure_speech::*` | `brainwires_provider_speech::azure_speech::*` |
| `brainwires_providers::cartesia::*` | `brainwires_provider_speech::cartesia::*` |
| `brainwires_providers::deepgram::*` | `brainwires_provider_speech::deepgram::*` |
| `brainwires_providers::elevenlabs::*` | `brainwires_provider_speech::elevenlabs::*` |
| `brainwires_providers::fish::*` | `brainwires_provider_speech::fish::*` |
| `brainwires_providers::google_tts::*` | `brainwires_provider_speech::google_tts::*` |
| `brainwires_providers::murf::*` | `brainwires_provider_speech::murf::*` |
| `brainwires_providers::web_speech::*` | `brainwires_provider_speech::web_speech::*` |

### Cargo features

The Cargo features on `brainwires-providers` map to the same names on
the new crates:

- LLM-side (`bedrock`, `vertex-ai`, `llama-cpp-2`, `local-llm-candle`, `local-llm-vision`, `candle-wgpu`) → on `brainwires-provider`.
- Speech (`web-speech`) → on `brainwires-provider-speech`.
- `native`, `wasm`, `telemetry` exist on both.

## Why split

Mixing 6 LLM chat APIs (with their candle / llama.cpp / aws-sigv4 /
gcp_auth weight) with 8 speech clients (with their wasm-bindgen
overhead) forced every consumer to compile both stacks. The split lets
voice apps depend on `brainwires-provider-speech` alone, and chat apps
depend on `brainwires-provider` alone, without dragging in the other.

## See also

- [ADR-0001](https://github.com/Brainwires/brainwires-framework/blob/main/docs/adr/ADR-0001-crate-split-discipline.md) — crate split discipline.
