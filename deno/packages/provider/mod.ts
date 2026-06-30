/**
 * @module @rullama/providers
 *
 * Provider layer for the rullama.
 * Contains chat provider implementations that wrap AI APIs with the
 * `Provider` interface from `@rullama/core`.
 *
 * Equivalent to Rust's `rullama-providers` crate.
 */

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
