/**
 * Provider types and configuration for the Brainwires providers package.
 * Equivalent to Rust's `ProviderType`, `ProviderConfig`, `ChatProtocol`, `AuthScheme`
 * in rullama-providers.
 */

// ---------------------------------------------------------------------------
// ProviderType
// ---------------------------------------------------------------------------

/** AI provider types.
 * Equivalent to Rust's `ProviderType` enum. */
export type ProviderType =
  | "anthropic"
  | "openai"
  | "google"
  | "groq"
  | "ollama"
  | "rullama"
  | "together"
  | "fireworks"
  | "anyscale"
  | "openai-responses"
  | "bedrock"
  | "vertex-ai"
  | "custom";

/** Parse a string into a ProviderType, or return undefined if unknown. */
export function parseProviderType(s: string): ProviderType | undefined {
  switch (s.toLowerCase()) {
    case "anthropic":
      return "anthropic";
    case "openai":
      return "openai";
    case "google":
    case "gemini":
      return "google";
    case "groq":
      return "groq";
    case "ollama":
      return "ollama";
    case "rullama":
      return "rullama";
    case "together":
      return "together";
    case "fireworks":
      return "fireworks";
    case "anyscale":
      return "anyscale";
    case "openai-responses":
    case "openai_responses":
      return "openai-responses";
    case "bedrock":
    case "aws-bedrock":
      return "bedrock";
    case "vertex-ai":
    case "vertex_ai":
    case "vertexai":
      return "vertex-ai";
    case "custom":
      return "custom";
    default:
      return undefined;
  }
}

/** Get the default model for a given provider type. */
export function defaultModel(provider: ProviderType): string {
  switch (provider) {
    case "anthropic":
      return "claude-sonnet-4-20250514";
    case "openai":
      return "gpt-5-mini";
    case "google":
      return "gemini-2.5-flash";
    case "groq":
      return "llama-3.3-70b-versatile";
    case "ollama":
      return "llama3.3";
    case "rullama":
      return "gpt-5-mini";
    case "together":
      return "meta-llama/Llama-3.1-8B-Instruct";
    case "fireworks":
      return "accounts/fireworks/models/llama-v3p1-8b-instruct";
    case "anyscale":
      return "meta-llama/Meta-Llama-3.1-8B-Instruct";
    case "openai-responses":
      return "gpt-5-mini";
    case "bedrock":
      return "anthropic.claude-3-5-sonnet-20241022-v2:0";
    case "vertex-ai":
      return "gemini-2.0-flash";
    case "custom":
      return "claude-sonnet-4-20250514";
  }
}

/** Whether a provider type requires an API key. */
export function requiresApiKey(provider: ProviderType): boolean {
  return provider !== "ollama" && provider !== "bedrock" &&
    provider !== "vertex-ai";
}

// ---------------------------------------------------------------------------
// ChatProtocol
// ---------------------------------------------------------------------------

/** Wire protocol spoken by a chat provider.
 * Equivalent to Rust's `ChatProtocol` enum. */
export type ChatProtocol =
  | "openai_chat_completions"
  | "openai_responses"
  | "anthropic_messages"
  | "gemini_generate_content"
  | "ollama_chat";

// ---------------------------------------------------------------------------
// AuthScheme
// ---------------------------------------------------------------------------

/** Authentication scheme used to authorize requests.
 * Equivalent to Rust's `AuthScheme` enum. */
export type AuthScheme =
  | { type: "bearer_token" }
  | { type: "custom_header"; header: string }
  | { type: "none" };

// ---------------------------------------------------------------------------
// ProviderConfig
// ---------------------------------------------------------------------------

/** Provider configuration.
 * Equivalent to Rust's `ProviderConfig`. */
export interface ProviderConfig {
  /** Provider type. */
  provider: ProviderType;
  /** Model name. */
  model: string;
  /** API key (if required). */
  api_key?: string;
  /** Base URL (for custom endpoints). */
  base_url?: string;
  /** Additional provider-specific options. */
  options?: Record<string, unknown>;
}

/** Create a new ProviderConfig. */
export function createProviderConfig(
  provider: ProviderType,
  model: string,
): ProviderConfig {
  return { provider, model };
}
