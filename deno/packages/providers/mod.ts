/**
 * @module @brainwires/providers
 *
 * Provider layer for the Brainwires Agent Framework.
 * Contains chat provider implementations that wrap AI APIs with the
 * `Provider` interface from `@brainwires/core`.
 *
 * Equivalent to Rust's `brainwires-providers` crate.
 */

// Types
export {
  createProviderConfig,
  defaultModel,
  parseProviderType,
  requiresApiKey,
  type AuthScheme,
  type ChatProtocol,
  type ProviderConfig,
  type ProviderType,
} from "./types.ts";

// Registry
export {
  lookup,
  PROVIDER_REGISTRY,
  type ProviderEntry,
} from "./registry.ts";

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

// Brainwires Relay — HTTP-based backend that multiplexes upstream models
export {
  BrainwiresRelayProvider,
  DEFAULT_BACKEND_URL,
  DEV_BACKEND_URL,
  getBackendFromApiKey,
  maxOutputTokensForModel,
} from "./brainwires_relay.ts";

// Audio providers — TTS / STT / ASR HTTP clients
export {
  AzureSpeechClient,
  type AzureSttRequest,
  type AzureSttResponse,
  type AzureVoice,
  CARTESIA_API_BASE,
  CARTESIA_VERSION,
  CartesiaClient,
  type CartesiaOutputFormat,
  type CartesiaTtsRequest,
  type CartesiaVoice,
  DEEPGRAM_API_BASE,
  type DeepgramAlternative,
  type DeepgramChannel,
  DeepgramClient,
  type DeepgramListenRequest,
  type DeepgramListenResponse,
  type DeepgramResults,
  type DeepgramSpeakRequest,
  type DeepgramWord,
  ELEVENLABS_API_BASE,
  ElevenLabsClient,
  elevenLabsSerializeTtsRequest,
  type ElevenLabsSttRequest,
  type ElevenLabsSttResponse,
  type ElevenLabsTtsRequest,
  type ElevenLabsVoice,
  type ElevenLabsVoicesResponse,
  type ElevenLabsVoiceSettings,
  FISH_API_BASE,
  type FishAsrRequest,
  type FishAsrResponse,
  FishClient,
  type FishTtsRequest,
  GOOGLE_TTS_API_BASE,
  type GoogleTtsAudioConfig,
  GoogleTtsClient,
  type GoogleTtsInput,
  type GoogleTtsSynthesizeRequest,
  type GoogleTtsSynthesizeResponse,
  type GoogleTtsVoiceEntry,
  type GoogleTtsVoicesResponse,
  type GoogleTtsVoiceSelection,
  MURF_API_BASE,
  MurfClient,
  type MurfGenerateRequest,
  type MurfGenerateResponse,
  type MurfVoice,
  type MurfVoicesResponse,
} from "./audio/mod.ts";

// Factory
export { ChatProviderFactory } from "./factory.ts";

// Rate limiter
export { RateLimitedClient, RateLimiter, type RateLimitedClientOptions } from "./rate_limiter.ts";

// Model listing
export {
  createModelLister,
  inferOpenaiCapabilities,
  isChatCapable,
  type AvailableModel,
  type ModelCapability,
  type ModelLister,
} from "./model_lister.ts";
