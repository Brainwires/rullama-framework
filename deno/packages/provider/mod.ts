/**
 * Chat provider implementations for rullama. Every class here implements the
 * `Provider` interface from `@rullama/core` over plain `fetch()`: Anthropic
 * Messages, OpenAI Chat Completions (also Groq / Together / Fireworks /
 * Anyscale), OpenAI Responses, AWS Bedrock (SigV4), Google Gemini, Vertex AI
 * (service-account JWT) and Ollama. `ChatProviderFactory` builds any of them
 * from a `ProviderConfig` by consulting `PROVIDER_REGISTRY`, and the module
 * also exposes the SSE / NDJSON stream parsers, the `RateLimiter` re-export
 * from core, and `createModelLister` for enumerating a provider's models.
 *
 * @module
 */

// Core types that appear in the exported provider signatures.
export type {
  ChatOptions,
  ChatResponse,
  Message,
  Provider,
  StreamChunk,
  Tool,
} from "@rullama/core";

// Types
export {
  type AuthScheme,
  type ChatProtocol,
  createProviderConfig,
  defaultModel,
  parseProviderType,
  type ProviderConfig,
  type ProviderType,
  requiresApiKey,
} from "./types.ts";

// Registry
export { lookup, PROVIDER_REGISTRY, type ProviderEntry } from "./registry.ts";

// SSE parsing utilities
export { parseNDJSONStream, parseSSEStream } from "./sse.ts";

// Providers
export { AnthropicChatProvider } from "./anthropic.ts";
export { OpenAiChatProvider } from "./openai.ts";
export { OpenAiResponsesProvider } from "./openai_responses.ts";
export { BedrockProvider } from "./bedrock.ts";
export { VertexAiProvider } from "./vertex.ts";
export { GoogleChatProvider } from "./gemini.ts";
export { OllamaChatProvider } from "./ollama.ts";

// Speech providers moved to @rullama/provider-speech.

// Factory
export { ChatProviderFactory } from "./factory.ts";

// Rate limiter
export {
  RateLimitedClient,
  type RateLimitedClientOptions,
  RateLimiter,
} from "./rate_limiter.ts";

// Model listing
export {
  type AvailableModel,
  createModelLister,
  inferOpenaiCapabilities,
  isChatCapable,
  type ModelCapability,
  type ModelLister,
} from "./model_lister.ts";
