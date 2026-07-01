/**
 * Model listing and validation for AI providers.
 *
 * Each provider implements {@link ModelLister} to query available models from its API.
 *
 * Equivalent to Rust's `model_listing` module in `rullama-providers`.
 */

import type { ProviderType } from "./types.ts";

// ---------------------------------------------------------------------------
// ModelCapability
// ---------------------------------------------------------------------------

/**
 * Capabilities a model may support.
 *
 * Equivalent to Rust's `ModelCapability` enum.
 */
export type ModelCapability =
  | "chat"
  | "tool_use"
  | "vision"
  | "embedding"
  | "audio"
  | "image_generation";

// ---------------------------------------------------------------------------
// AvailableModel
// ---------------------------------------------------------------------------

/**
 * A model available from a provider.
 *
 * Equivalent to Rust's `AvailableModel` struct.
 */
export interface AvailableModel {
  /** Model identifier (e.g. "claude-sonnet-4-20250514", "gpt-4o"). */
  id: string;
  /** Human-readable name, if provided by the API. */
  displayName?: string;
  /** Which provider owns this model. */
  provider: ProviderType;
  /** What the model can do. */
  capabilities: ModelCapability[];
  /** Organization/owner string from the API. */
  ownedBy?: string;
  /** Maximum input context window (tokens). */
  contextWindow?: number;
  /** Maximum output tokens the model can produce. */
  maxOutputTokens?: number;
  /** Unix timestamp (seconds) when the model was created. */
  createdAt?: number;
}

/**
 * Whether the given model supports chat completions.
 *
 * Equivalent to Rust's `AvailableModel::is_chat_capable()`.
 */
export function isChatCapable(model: AvailableModel): boolean {
  return model.capabilities.includes("chat");
}

// ---------------------------------------------------------------------------
// ModelLister interface
// ---------------------------------------------------------------------------

/**
 * Interface for querying a provider's model catalogue.
 *
 * Equivalent to Rust's `ModelLister` trait.
 */
export interface ModelLister {
  /** Fetch all models available for this provider. */
  listModels(): Promise<AvailableModel[]>;
}

// ---------------------------------------------------------------------------
// Capability inference helpers
// ---------------------------------------------------------------------------

/**
 * Infer capabilities for an OpenAI-format model ID.
 *
 * Shared by OpenAI and Groq listers.
 * Equivalent to Rust's `infer_openai_capabilities()`.
 */
export function inferOpenaiCapabilities(modelId: string): ModelCapability[] {
  const id = modelId.toLowerCase();

  // Embedding models
  if (id.includes("embedding") || id.startsWith("text-embedding")) {
    return ["embedding"];
  }

  // Audio models
  if (id.startsWith("whisper") || id.startsWith("tts")) {
    return ["audio"];
  }

  // Image generation
  if (id.startsWith("dall-e")) {
    return ["image_generation"];
  }

  // Chat-capable models get Chat + ToolUse by default
  const caps: ModelCapability[] = ["chat", "tool_use"];

  // Vision-capable models
  if (
    id.includes("vision") ||
    id.includes("gpt-4o") ||
    id.includes("gpt-4-turbo") ||
    id.includes("gpt-5") ||
    (id.startsWith("o") && !id.startsWith("omni"))
  ) {
    caps.push("vision");
  }

  return caps;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Supported provider types for model listing.
 *
 * Providers not listed here will cause `createModelLister()` to throw.
 */
const LISTING_SUPPORTED: ReadonlySet<ProviderType> = new Set<ProviderType>([
  "anthropic",
  "openai",
  "google",
  "groq",
  "together",
  "fireworks",
  "anyscale",
  "ollama",
  "openai-responses",
]);

/**
 * Create a {@link ModelLister} for the given provider.
 *
 * This is a lightweight factory that returns a generic HTTP-based lister.
 * In a full implementation each provider would have its own lister class;
 * this factory validates inputs and returns a minimal stub that callers
 * can replace with a concrete implementation.
 *
 * Equivalent to Rust's `create_model_lister()`.
 *
 * @param providerType - The provider to create a lister for.
 * @param apiKey - Required for cloud providers, ignored for Ollama.
 * @param baseUrl - Optional URL override (for Ollama or custom endpoints).
 * @returns A ModelLister instance.
 * @throws If the provider is unsupported or a required API key is missing.
 */
export function createModelLister(
  providerType: ProviderType,
  apiKey?: string,
  baseUrl?: string,
): ModelLister {
  if (!LISTING_SUPPORTED.has(providerType)) {
    throw new Error(
      `Model listing is not supported for ${providerType} provider via this interface`,
    );
  }

  // Ollama does not require an API key
  if (providerType !== "ollama" && !apiKey) {
    throw new Error(`${providerType} requires an API key`);
  }

  // Return a stub lister — concrete HTTP-based implementations can be added
  // per-provider in future PRs, matching the Rust provider modules.
  return {
    // deno-lint-ignore require-await
    async listModels(): Promise<AvailableModel[]> {
      void baseUrl; // reserved for future use
      void apiKey;
      throw new Error(
        `listModels() not yet implemented for ${providerType}. ` +
          `Provide a concrete ModelLister implementation.`,
      );
    },
  };
}
