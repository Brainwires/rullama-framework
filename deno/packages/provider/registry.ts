/**
 * Provider registry — connection details for all known providers.
 * Maps each ProviderType to its wire protocol, default endpoint, auth scheme,
 * and model-listing URL. Equivalent to Rust's `registry.rs`.
 */

import type { AuthScheme, ChatProtocol, ProviderType } from "./types.ts";

/** Static metadata for a known provider.
 * Equivalent to Rust's `ProviderEntry`. */
export interface ProviderEntry {
  /** Which provider this entry describes. */
  provider_type: ProviderType;
  /** Chat wire protocol. */
  chat_protocol: ChatProtocol;
  /** Default base URL for API requests. */
  default_base_url: string;
  /** Default model identifier. */
  default_model: string;
  /** Authentication scheme. */
  auth: AuthScheme;
  /** Whether the provider supports listing available models. */
  supports_model_listing: boolean;
  /** URL for the models endpoint (if supported). */
  models_url?: string;
}

/** Static table of all known chat providers.
 * Audio-only providers are not included since they don't implement the Provider (chat) trait. */
export const PROVIDER_REGISTRY: readonly ProviderEntry[] = [
  {
    provider_type: "openai",
    chat_protocol: "openai_chat_completions",
    default_base_url: "https://api.openai.com/v1/chat/completions",
    default_model: "gpt-4o",
    auth: { type: "bearer_token" },
    supports_model_listing: true,
    models_url: "https://api.openai.com/v1/models",
  },
  {
    provider_type: "groq",
    chat_protocol: "openai_chat_completions",
    default_base_url: "https://api.groq.com/openai/v1/chat/completions",
    default_model: "llama-3.3-70b-versatile",
    auth: { type: "bearer_token" },
    supports_model_listing: true,
    models_url: "https://api.groq.com/openai/v1/models",
  },
  {
    provider_type: "together",
    chat_protocol: "openai_chat_completions",
    default_base_url: "https://api.together.xyz/v1/chat/completions",
    default_model: "meta-llama/Llama-3.1-8B-Instruct",
    auth: { type: "bearer_token" },
    supports_model_listing: true,
    models_url: "https://api.together.xyz/v1/models",
  },
  {
    provider_type: "fireworks",
    chat_protocol: "openai_chat_completions",
    default_base_url: "https://api.fireworks.ai/inference/v1/chat/completions",
    default_model: "accounts/fireworks/models/llama-v3p1-8b-instruct",
    auth: { type: "bearer_token" },
    supports_model_listing: true,
    models_url: "https://api.fireworks.ai/inference/v1/models",
  },
  {
    provider_type: "anyscale",
    chat_protocol: "openai_chat_completions",
    default_base_url: "https://api.endpoints.anyscale.com/v1/chat/completions",
    default_model: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    auth: { type: "bearer_token" },
    supports_model_listing: true,
    models_url: "https://api.endpoints.anyscale.com/v1/models",
  },
  {
    provider_type: "anthropic",
    chat_protocol: "anthropic_messages",
    default_base_url: "https://api.anthropic.com/v1/messages",
    default_model: "claude-3-5-sonnet-20241022",
    auth: { type: "custom_header", header: "x-api-key" },
    supports_model_listing: true,
    models_url: "https://api.anthropic.com/v1/models",
  },
  {
    provider_type: "google",
    chat_protocol: "gemini_generate_content",
    default_base_url: "https://generativelanguage.googleapis.com",
    default_model: "gemini-2.0-flash-exp",
    auth: { type: "bearer_token" },
    supports_model_listing: true,
    models_url: "https://generativelanguage.googleapis.com/v1beta/models",
  },
  {
    provider_type: "ollama",
    chat_protocol: "ollama_chat",
    default_base_url: "http://localhost:11434",
    default_model: "llama3.1",
    auth: { type: "none" },
    supports_model_listing: true,
    models_url: "http://localhost:11434/api/tags",
  },
  {
    provider_type: "openai-responses",
    chat_protocol: "openai_responses",
    default_base_url: "https://api.openai.com/v1/responses",
    default_model: "gpt-4o",
    auth: { type: "bearer_token" },
    supports_model_listing: true,
    models_url: "https://api.openai.com/v1/models",
  },
  {
    provider_type: "bedrock",
    chat_protocol: "anthropic_messages",
    default_base_url: "https://bedrock-runtime.us-east-1.amazonaws.com",
    default_model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    auth: { type: "none" },
    supports_model_listing: false,
  },
  {
    provider_type: "vertex-ai",
    chat_protocol: "gemini_generate_content",
    default_base_url: "https://us-central1-aiplatform.googleapis.com",
    default_model: "gemini-2.0-flash",
    auth: { type: "bearer_token" },
    supports_model_listing: false,
  },
] as const;

/** Look up the registry entry for a given provider type. */
export function lookup(
  providerType: ProviderType,
): ProviderEntry | undefined {
  return PROVIDER_REGISTRY.find((e) => e.provider_type === providerType);
}
