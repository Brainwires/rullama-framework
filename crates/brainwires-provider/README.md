# brainwires-provider

[![Crates.io](https://img.shields.io/crates/v/brainwires-provider.svg)](https://crates.io/crates/brainwires-provider)
[![Documentation](https://img.shields.io/docsrs/brainwires-provider)](https://docs.rs/brainwires-provider)
[![License](https://img.shields.io/crates/l/brainwires-provider.svg)](LICENSE)

AI provider implementations for the Brainwires Agent Framework.

## Overview

`brainwires-provider` provides concrete implementations of the `Provider` trait for multiple AI services: Anthropic (Claude), OpenAI (GPT), Google (Gemini), Ollama, and local LLM inference via llama.cpp. Every provider converts to and from the unified `brainwires-core` message types, so callers can swap backends without changing application code.

**Design principles:**

- **Unified interface** — all providers implement the same `Provider` trait from `brainwires-core` (chat, streaming, tool calling)
- **Feature-gated backends** — cloud providers compile under `native` (default); local LLM compiles always; llama.cpp is behind `llama-cpp-2`
- **Rate limiting built in** — token-bucket `RateLimiter` and `RateLimitedClient` available to any provider
- **Streaming-first** — every provider returns `BoxStream<Result<StreamChunk>>` via `async_stream`
- **Tool calling** — Anthropic, OpenAI, Google, and Ollama all support function calling mapped to/from `brainwires_core::Tool`
- **Local inference** — CPU-optimized GGUF model support with model registry, preset configs, and inference parameter tuning

```text
  ┌───────────────────────────────────────────────────────────────────────┐
  │                        brainwires-provider                           │
  │                                                                       │
  │  ┌─── Provider trait (brainwires-core) ────────────────────────────┐  │
  │  │  chat()        ──► ChatResponse                                 │  │
  │  │  stream_chat() ──► BoxStream<StreamChunk>                       │  │
  │  │  name()        ──► &str                                         │  │
  │  └─────────────────────────────────────────────────────────────────┘  │
  │           │                                                           │
  │           ▼                                                           │
  │  ┌─── Cloud Providers (feature = "native") ────────────────────────┐  │
  │  │                                                                  │  │
  │  │  AnthropicProvider ──► SSE streaming ──► api.anthropic.com      │  │
  │  │  OpenAIProvider    ──► JSON Lines    ──► api.openai.com         │  │
  │  │  GoogleProvider    ──► event-stream  ──► generativelanguage.…   │  │
  │  │  OllamaProvider    ──► line-delim JSON ► localhost:11434        │  │
  │  │           │                                                      │  │
  │  │           ▼                                                      │  │
  │  │  RateLimitedClient ──► RateLimiter (token-bucket)               │  │
  │  └──────────────────────────────────────────────────────────────────┘  │
  │                                                                       │
  │  ┌─── Local LLM (always compiled, llama.cpp optional) ─────────────┐  │
  │  │                                                                  │  │
  │  │  LocalLlmProvider ──► generate() / route() / process()         │  │
  │  │       │                                                          │  │
  │  │       ▼                                                          │  │
  │  │  LocalLlmConfig ◄── LocalModelRegistry ◄── scan_models_dir()   │  │
  │  │  LocalModelType  ◄── chat_template() / stop_tokens()           │  │
  │  │  LocalInferenceParams ◄── factual() / creative() / routing()   │  │
  │  │  LocalLlmPool    ──► round-robin multi-instance inference       │  │
  │  └──────────────────────────────────────────────────────────────────┘  │
  │                                                                       │
  │  ┌─── Shared Types ───────────────────────────────────────────────┐   │
  │  │  ProviderType (Anthropic | OpenAI | Google | Ollama | Custom)  │   │
  │  │  ProviderConfig (provider, model, api_key, base_url, options)  │   │
  │  └────────────────────────────────────────────────────────────────┘   │
  └───────────────────────────────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires-provider = "0.11"
```

Send a chat request with the Anthropic provider:

```rust
use brainwires_providers::{AnthropicProvider, Provider, ChatOptions};
use brainwires_core::message::Message;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = AnthropicProvider::new("sk-ant-...".into(), "claude-sonnet-4-20250514".into());

    let messages = vec![Message::user("Explain the borrow checker in one sentence.")];
    let options = ChatOptions::default();

    let response = provider.chat(&messages, None, &options).await?;
    println!("{}", response.message.text().unwrap_or_default());
    println!("Tokens: {} in, {} out", response.usage.input_tokens, response.usage.output_tokens);

    Ok(())
}
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `native` | Yes | Enables cloud providers (Anthropic, OpenAI, Google, Ollama), `RateLimiter`, `RateLimitedClient`, and their dependencies (`reqwest`, `tokio`, `async-stream`, `tracing`, `uuid`) |
| `llama-cpp-2` | No | Enables local LLM inference via llama.cpp bindings. Heavy dependency (~long compile). Adds `tracing` and `tokio` even without `native` |

```toml
# Default (cloud providers only)
brainwires-provider = "0.11"

# With local LLM support
brainwires-provider = { version = "0.11", features = ["llama-cpp-2"] }

# Local LLM only (no cloud providers)
brainwires-provider = { version = "0.11", default-features = false, features = ["llama-cpp-2"] }
```

## Architecture

### Provider Trait

All providers implement the `Provider` trait from `brainwires-core`. This is the unified interface that callers program against.

| Method | Description |
|--------|-------------|
| `name()` | Provider identifier string (e.g., `"anthropic"`, `"lfm2-350m"`) |
| `max_output_tokens()` | Optional maximum output token limit for the provider |
| `chat(messages, tools, options)` | Non-streaming chat completion returning `ChatResponse` |
| `stream_chat(messages, tools, options)` | Streaming chat returning `BoxStream<Result<StreamChunk>>` |

**`ChatOptions`** controls per-request behavior:

| Field | Type | Description |
|-------|------|-------------|
| `system` | `Option<String>` | System prompt |
| `temperature` | `Option<f32>` | Sampling temperature (0.0–2.0) |
| `max_tokens` | `Option<u32>` | Maximum tokens to generate |
| `stop` | `Option<Vec<String>>` | Stop sequences |

**`StreamChunk`** variants:

| Variant | Description |
|---------|-------------|
| `Text(String)` | Generated text token |
| `Usage(Usage)` | Token usage counts (input + output) |
| `Done` | Stream completion marker |

### ProviderType

Enum identifying the AI provider backend.

| Variant | `as_str()` | `default_model()` |
|---------|-----------|-------------------|
| `Anthropic` | `"anthropic"` | `claude-sonnet-4-20250514` |
| `OpenAI` | `"openai"` | `gpt-5-mini` |
| `Google` | `"google"` | `gemini-2.5-flash` |
| `Ollama` | `"ollama"` | `llama3.3` |
| `Custom` | `"custom"` | `claude-sonnet-4-20250514` |

`FromStr` also accepts aliases: `"gemini"` maps to `Google`, `"brainwires"` maps to `Custom`.

### ProviderConfig

Configuration struct for initializing a provider.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `ProviderType` | — | Provider backend |
| `model` | `String` | — | Model name |
| `api_key` | `Option<String>` | `None` | API key (skipped in serialization if absent) |
| `base_url` | `Option<String>` | `None` | Custom endpoint URL |
| `options` | `HashMap<String, Value>` | `{}` | Additional provider-specific options (flattened in JSON) |

Builder methods: `new(provider, model)`, `with_api_key(key)`, `with_base_url(url)`.

### RateLimiter

Token-bucket rate limiter using atomic operations for lock-free reads.

| Field | Type | Description |
|-------|------|-------------|
| `tokens` | `AtomicU32` | Current available tokens |
| `max_tokens` | `u32` | Configured requests-per-minute limit |
| `refill_interval` | `Duration` | Time between token refills (`60s / rpm`) |
| `last_refill` | `Mutex<Instant>` | Timestamp of last refill |

| Method | Description |
|--------|-------------|
| `new(requests_per_minute)` | Create a limiter with the given RPM cap |
| `acquire()` | Async — consume one token, wait if depleted |
| `available_tokens()` | Current token count (diagnostic) |
| `max_requests_per_minute()` | Configured limit |

### RateLimitedClient

Wraps `reqwest::Client` with an optional `RateLimiter`. Every outgoing request waits for a token before sending.

| Method | Description |
|--------|-------------|
| `new()` | Create client with no rate limiting |
| `with_rate_limit(rpm)` | Create client with the given RPM limit |
| `from_client(client, rpm)` | Wrap an existing `reqwest::Client` |
| `get(url)` | Build a GET request (waits for token first) |
| `post(url)` | Build a POST request (waits for token first) |
| `inner()` | Access the underlying `reqwest::Client` |
| `available_tokens()` | Returns `Option<u32>` — `None` if no limiter |

### AnthropicProvider

Implements the `Provider` trait for the Anthropic Messages API (`https://api.anthropic.com/v1/messages`, version `2023-06-01`).

| Constructor | Description |
|-------------|-------------|
| `new(api_key, model)` | Create without rate limiting |
| `with_rate_limit(api_key, model, rpm)` | Create with rate limiting |

**Streaming:** Parses Server-Sent Events (SSE) with `data: ` prefix. Events include `message_start`, `content_block_delta`, `message_delta`, and `message_stop`.

**Internal types:** `AnthropicMessage`, `AnthropicContentBlock` (Text, ToolUse, ToolResult), `AnthropicTool`, `AnthropicResponse`, `AnthropicStreamEvent`, `AnthropicDelta`.

**Message conversion:** System messages are extracted from the message list and sent as a top-level `system` field. All other messages are converted to Anthropic's role/content-block format.

### OpenAIProvider

Implements the `Provider` trait for the OpenAI Chat Completions API (`https://api.openai.com/v1/chat/completions`).

| Constructor | Description |
|-------------|-------------|
| `new(api_key, model)` | Create without rate limiting |
| `with_rate_limit(api_key, model, rpm)` | Create with rate limiting |
| `with_organization(org_id)` | Set the `OpenAI-Organization` header |

**Streaming:** Parses newline-delimited JSON (JSON Lines). Each line is a `data: {json}` SSE chunk with `choices[0].delta`.

**O1/O3 model detection:** `is_o1_model()` detects reasoning models (o1, o3 prefixes) which do not support `temperature`, `max_tokens`, or system messages.

**Image support:** Converts `ContentBlock::Image` to base64-encoded `image_url` content parts.

**Internal types:** `OpenAIMessage`, `OpenAIContent` (Text or Array), `OpenAIContentPart` (Text, ImageUrl, ToolCall), `OpenAITool`, `OpenAIResponse`.

### GoogleProvider

Implements the `Provider` trait for the Gemini API (`https://generativelanguage.googleapis.com/v1beta`).

| Constructor | Description |
|-------------|-------------|
| `new(api_key, model)` | Create without rate limiting |
| `with_rate_limit(api_key, model, rpm)` | Create with rate limiting |

**Streaming:** Uses `text/event-stream` with custom Gemini event format.

**Message conversion:** System messages are filtered out and sent via `systemInstruction`. The assistant role maps to `"model"` in Gemini's API.

**Image support:** Converts images to `inlineData` parts with MIME type and base64 data.

**Internal types:** `GeminiMessage`, `GeminiPart` (Text, InlineData, FunctionCall, FunctionResult), `GeminiTool`, `GeminiResponse`.

### OllamaProvider

Implements the `Provider` trait for the Ollama REST API (default: `http://localhost:11434`).

| Constructor | Description |
|-------------|-------------|
| `new(model, base_url)` | Create with model name and optional custom URL |
| `with_rate_limit(model, base_url, rpm)` | Create with rate limiting |

**Streaming:** Line-delimited JSON where each line contains a `message` field and a `done` boolean.

**Content handling:** Multiple content blocks are flattened into a single concatenated text string, since Ollama's API expects plain text.

**Internal types:** `OllamaMessage`, `OllamaTool`, `OllamaResponse`.

### Local LLM Subsystem

Always compiled (no feature gate). The actual llama.cpp inference requires the `llama-cpp-2` feature.

#### LocalLlmConfig

Configuration for a local GGUF model.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `String` | `"local-model"` | Unique model identifier |
| `name` | `String` | `"Local Model"` | Human-readable name |
| `model_path` | `PathBuf` | — | Path to the `.gguf` file |
| `context_size` | `u32` | `4096` | Context window size in tokens |
| `num_threads` | `Option<u32>` | `None` (auto) | CPU threads for inference |
| `batch_size` | `u32` | `512` | Prompt processing batch size |
| `gpu_layers` | `u32` | `0` | GPU layers to offload (0 = CPU only) |
| `use_mmap` | `bool` | `true` | Memory-map model file for faster loading |
| `use_mlock` | `bool` | `false` | Lock model in RAM to prevent swapping |
| `max_tokens` | `u32` | `2048` | Maximum tokens per response |
| `model_type` | `LocalModelType` | `Lfm2` | Model family for prompt formatting |
| `system_template` | `Option<String>` | `None` | Custom system prompt template |
| `supports_tools` | `bool` | `false` | Whether the model handles tool/function calling |
| `estimated_ram_mb` | `Option<u32>` | `None` | Estimated RAM usage (display only) |

**Preset constructors:**

| Preset | Context | RAM | Tools | Description |
|--------|---------|-----|-------|-------------|
| `lfm2_350m(path)` | 32K | 220 MB | No | Fastest, routing and binary decisions |
| `lfm2_1_2b(path)` | 32K | 700 MB | No | Sweet spot for agentic logic |
| `lfm2_2_6b_exp(path)` | 32K | 1.5 GB | Yes | Complex reasoning and tool-calling |
| `granite_nano_350m(path)` | 8K | 250 MB | No | Sub-second CPU responses |
| `granite_nano_1_5b(path)` | 8K | 900 MB | No | Balanced performance |

**Validation:** `validate()` checks model path exists, context size > 0, batch size > 0.

#### LocalModelType

Model family enum that determines chat template formatting and stop tokens.

| Variant | Chat Template Style | Stop Tokens |
|---------|-------------------|-------------|
| `Lfm2` | `<\|system\|>...<\|end\|>` | `<\|end\|>`, `<\|user\|>` |
| `Lfm2Agentic` | Same as Lfm2 | Same as Lfm2 |
| `Granite` | `<\|system\|>...\n` | `<\|user\|>`, `<\|system\|>` |
| `Qwen` | `<\|im_start\|>...<\|im_end\|>` | `<\|im_end\|>`, `<\|im_start\|>` |
| `Llama` | `<\|begin_of_text\|>...<\|eot_id\|>` | `<\|eot_id\|>`, `<\|start_header_id\|>` |
| `Phi` | Same as Lfm2 | Same as Lfm2 |
| `Generic` | `### System:...\n### User:...` | `### User:`, `### System:` |

Methods: `chat_template()`, `stop_tokens()`.

#### LocalInferenceParams

Per-request sampling parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `f32` | `0.7` | Sampling temperature (0.0 = deterministic) |
| `top_p` | `f32` | `0.9` | Nucleus sampling threshold |
| `top_k` | `u32` | `40` | Top-k sampling parameter |
| `repeat_penalty` | `f32` | `1.1` | Repetition penalty (1.0 = none) |
| `max_tokens` | `u32` | `2048` | Maximum tokens to generate |
| `stop_sequences` | `Vec<String>` | `[]` | Custom stop sequences |

**Presets:**

| Preset | Temperature | Top-k | Max Tokens | Use Case |
|--------|------------|-------|------------|----------|
| `factual()` | 0.1 | 20 | 1024 | Deterministic, factual responses |
| `creative()` | 0.9 | 50 | 2048 | Varied, creative output |
| `routing()` | 0.0 | 1 | 50 | Classification and routing |

#### LocalModelRegistry

Manages registered local models with persistence to `~/.config/brainwires/local_models.json`.

| Method | Description |
|--------|-------------|
| `new()` | Create an empty registry |
| `with_default_dir()` | Create with default models directory (`~/.local/share/brainwires/models/`) |
| `register(config)` | Add a model configuration |
| `get(id)` | Get model by ID |
| `get_default()` | Get the default model |
| `set_default(id)` | Set the default model (returns `false` if ID not found) |
| `remove(id)` | Remove a model (clears default if it was the removed model) |
| `list()` | List all registered models |
| `scan_models_dir()` | Auto-discover `.gguf` files and register them with detected model types |
| `load()` | Load registry from config file |
| `save()` | Save registry to config file |

**Auto-detection:** `scan_models_dir()` reads the models directory, infers `LocalModelType` from filenames (e.g., `lfm2` → `Lfm2`, `granite` → `Granite`), and estimates context size and RAM from model size indicators in the filename.

#### KnownModel

Pre-configured model definitions for easy discovery and downloading.

| Field | Description |
|-------|-------------|
| `id` | Model identifier (e.g., `"lfm2-1.2b"`) |
| `name` | Human-readable name |
| `huggingface_repo` | HuggingFace repository path |
| `filename` | Expected GGUF filename |
| `model_type` | `LocalModelType` variant |
| `context_size` | Context window size |
| `estimated_ram_mb` | RAM requirement |
| `supports_tools` | Tool-calling support |
| `description` | Short description |

Access via `known_models()` (full list) or `get_known_model(id)` (by ID).

#### LocalLlmProvider

Implements the `Provider` trait for local GGUF model inference. Lazy-loads the model on first use.

| Method | Description |
|--------|-------------|
| `new(config)` | Create provider (validates config, does not load model) |
| `lfm2_350m(path)` | Shorthand for LFM2 350M preset |
| `lfm2_1_2b(path)` | Shorthand for LFM2 1.2B preset |
| `config()` | Get the model configuration |
| `is_loaded()` | Check if model is in memory |
| `load()` | Load model into memory (initializes llama.cpp backend) |
| `unload()` | Release model from memory |
| `generate(prompt, params)` | Generate text with custom parameters |
| `route(prompt)` | Quick routing/classification (deterministic params) |
| `process(prompt)` | Summarization/processing (factual params) |

Without the `llama-cpp-2` feature, `load()` and `generate()` return an error directing the user to enable the feature.

#### LocalLlmPool

Round-robin pool of `LocalLlmProvider` instances for parallel inference.

| Method | Description |
|--------|-------------|
| `new(config, instances)` | Create pool with N identical provider instances |
| `next()` | Get the next provider (round-robin via `AtomicUsize`) |
| `load_all()` | Load all models in the pool |
| `unload_all()` | Unload all models |
| `size()` | Number of instances |
| `estimated_ram_mb()` | Total estimated RAM for the pool |

### LocalLlmConfigError

| Variant | Description |
|---------|-------------|
| `MissingModelPath` | Model path is empty |
| `ModelNotFound(PathBuf)` | File does not exist at the given path |
| `InvalidContextSize` | Context size is 0 |
| `InvalidBatchSize` | Batch size is 0 |
| `ModelLoadError(String)` | llama.cpp failed to load the model |
| `InferenceError(String)` | Error during token generation |

## Usage Examples

### Stream a response from OpenAI

```rust
use brainwires_providers::{OpenAIProvider, Provider, ChatOptions};
use brainwires_core::message::{Message, StreamChunk};
use futures::StreamExt;

let provider = OpenAIProvider::new("sk-...".into(), "gpt-5-mini".into());

let messages = vec![Message::user("Write a haiku about Rust.")];
let options = ChatOptions::default();

let mut stream = provider.stream_chat(&messages, None, &options);
while let Some(chunk) = stream.next().await {
    match chunk? {
        StreamChunk::Text(text) => print!("{}", text),
        StreamChunk::Usage(usage) => {
            println!("\n[{} in, {} out]", usage.input_tokens, usage.output_tokens);
        }
        StreamChunk::Done => break,
    }
}
```

### Use tools with the Anthropic provider

```rust
use brainwires_providers::{AnthropicProvider, Provider, ChatOptions};
use brainwires_core::message::Message;
use brainwires_core::tool::Tool;

let provider = AnthropicProvider::new("sk-ant-...".into(), "claude-sonnet-4-20250514".into());

let tools = vec![Tool {
    name: "get_weather".into(),
    description: Some("Get current weather for a city".into()),
    input_schema: serde_json::json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" }
        },
        "required": ["city"]
    }),
    ..Default::default()
}];

let messages = vec![Message::user("What's the weather in Seattle?")];
let options = ChatOptions::default();

let response = provider.chat(&messages, Some(&tools), &options).await?;
// response.message may contain tool_use content blocks
```

### Rate-limited HTTP requests

```rust
use brainwires_providers::{RateLimitedClient, RateLimiter};

// Standalone rate limiter
let limiter = RateLimiter::new(60); // 60 RPM
limiter.acquire().await; // blocks if depleted

// Rate-limited HTTP client
let client = RateLimitedClient::with_rate_limit(120); // 120 RPM
let response = client.post("https://api.example.com/v1/chat")
    .await
    .json(&body)
    .send()
    .await?;

println!("Tokens remaining: {:?}", client.available_tokens());
```

### Provider with rate limiting

```rust
use brainwires_providers::{AnthropicProvider, Provider, ChatOptions};
use brainwires_core::message::Message;

// Create provider with 60 requests-per-minute limit
let provider = AnthropicProvider::with_rate_limit(
    "sk-ant-...".into(),
    "claude-sonnet-4-20250514".into(),
    60,
);

let messages = vec![Message::user("Hello!")];
let response = provider.chat(&messages, None, &ChatOptions::default()).await?;
```

### Configure a provider with ProviderConfig

```rust
use brainwires_providers::{ProviderType, ProviderConfig};

let config = ProviderConfig::new(ProviderType::OpenAI, "gpt-5-mini".into())
    .with_api_key("sk-...")
    .with_base_url("https://custom-openai-proxy.example.com/v1");

assert_eq!(config.provider.default_model(), "gpt-5-mini");
assert_eq!(config.provider.as_str(), "openai");

// Parse provider from string
let provider_type: ProviderType = "gemini".parse()?; // → Google
```

### Local LLM inference

```rust
use brainwires_providers::{LocalLlmProvider, LocalLlmConfig, LocalInferenceParams};
use std::path::PathBuf;

// Create provider from a preset
let provider = LocalLlmProvider::lfm2_1_2b(PathBuf::from("/models/lfm2-1.2b-q8_0.gguf"))?;

// Load model into memory
provider.load().await?;

// Quick routing (deterministic, max 50 tokens)
let route = provider.route("Classify: 'fix the login bug' → [code, question, chat]").await?;

// Full inference with custom params
let result = provider.generate(
    "Explain ownership in Rust briefly.",
    &LocalInferenceParams::factual(),
).await?;

// Or use via the Provider trait
use brainwires_providers::{Provider, ChatOptions};
use brainwires_core::message::Message;

let messages = vec![Message::user("Summarize this code.")];
let response = provider.chat(&messages, None, &ChatOptions::default()).await?;

// Unload when done
provider.unload().await;
```

### Model registry and auto-discovery

```rust
use brainwires_providers::{LocalModelRegistry, LocalLlmConfig, known_models, get_known_model};
use std::path::PathBuf;

// Load or create registry
let mut registry = LocalModelRegistry::load()?;

// Register a model manually
registry.register(LocalLlmConfig::lfm2_350m(PathBuf::from("/models/lfm2-350m.gguf")));
registry.set_default("lfm2-350m");

// Auto-discover GGUF files in the models directory
let discovered = registry.scan_models_dir()?;
for id in &discovered {
    println!("Found: {}", id);
}

// Browse known/recommended models
for model in known_models() {
    println!("{}: {} ({}MB RAM) — {}", model.id, model.name, model.estimated_ram_mb, model.description);
}

// Save registry
registry.save()?;
```

### Local LLM pool for parallel inference

```rust
use brainwires_providers::{LocalLlmPool, LocalLlmConfig};
use std::path::PathBuf;

let config = LocalLlmConfig::lfm2_350m(PathBuf::from("/models/lfm2-350m.gguf"));
let pool = LocalLlmPool::new(config, 4)?; // 4 instances

pool.load_all().await?;
println!("Pool RAM: ~{}MB", pool.estimated_ram_mb().unwrap_or(0));

// Round-robin across instances
let provider = pool.next();
let result = provider.route("classify this input").await?;

pool.unload_all().await;
```

## Integration

Use via the `brainwires` facade crate with the `providers` feature, or depend on `brainwires-provider` directly:

```toml
# Via facade
[dependencies]
brainwires = { version = "0.11", features = ["providers"] }

# Direct
[dependencies]
brainwires-provider = "0.11"
```

Re-exports at crate root for convenience:

```rust
use brainwires_providers::{
    // Trait + options (from brainwires-core)
    Provider, ChatOptions,
    // Cloud providers (native)
    AnthropicProvider, OpenAIProvider, GoogleProvider, OllamaProvider,
    // Rate limiting (native)
    RateLimiter, RateLimitedClient,
    // Shared types
    ProviderType, ProviderConfig,
    // Local LLM (always available)
    LocalLlmProvider, LocalLlmConfig, LocalModelType,
    LocalInferenceParams, LocalModelRegistry, LocalLlmPool,
    LocalLlmConfigError, KnownModel, known_models, get_known_model,
};
```

## License

Licensed under the MIT License. See [LICENSE](../../LICENSE) for details.
