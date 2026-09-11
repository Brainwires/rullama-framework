/**
 * Model listing and validation for AI providers.
 *
 * Each provider implements {@link ModelLister} to query available models from its API.
 *
 * Equivalent to Rust's `model_listing` module in `rullama-providers`.
 */

import type { ProviderType } from "./types.ts";
import { lookup, type ProviderEntry } from "./registry.ts";

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

/** Vendor endpoints answer in three shapes; `models_url` in the registry picks one. */
type ListingShape = "openai" | "anthropic" | "google" | "ollama";

function listingShape(providerType: ProviderType): ListingShape {
  switch (providerType) {
    case "anthropic":
      return "anthropic";
    case "google":
      return "google";
    case "ollama":
      return "ollama";
    default:
      return "openai";
  }
}

/** Headers for the listing request per the registry's `auth` scheme. */
function listingHeaders(
  entry: ProviderEntry,
  apiKey: string | undefined,
): HeadersInit {
  switch (entry.auth.type) {
    case "bearer_token":
      return { Authorization: `Bearer ${apiKey ?? ""}` };
    case "custom_header":
      return {
        [entry.auth.header]: apiKey ?? "",
        ...(entry.provider_type === "anthropic"
          ? { "anthropic-version": "2023-06-01" }
          : {}),
      };
    default:
      return {};
  }
}

/** Parse a listing response body into models, per vendor shape. */
export function parseModelListing(
  providerType: ProviderType,
  body: unknown,
): AvailableModel[] {
  const b = (body ?? {}) as Record<string, unknown>;
  const shape = listingShape(providerType);
  if (shape === "google") {
    const models = (b.models ?? []) as {
      name: string;
      displayName?: string;
      supportedGenerationMethods?: string[];
    }[];
    return models.map((m) => ({
      id: m.name.replace(/^models\//, ""),
      displayName: m.displayName,
      provider: providerType,
      capabilities: m.supportedGenerationMethods?.includes("generateContent")
        ? ["chat", "tool_use", "vision"]
        : ["embedding"],
    }));
  }
  if (shape === "ollama") {
    const models = (b.models ?? []) as { name: string; modified_at?: string }[];
    return models.map((m) => ({
      id: m.name,
      provider: providerType,
      capabilities: ["chat", "tool_use"],
    }));
  }
  const data = (b.data ?? []) as {
    id: string;
    display_name?: string;
    owned_by?: string;
    created?: number;
    created_at?: string;
  }[];
  return data.map((m) => ({
    id: m.id,
    displayName: m.display_name,
    provider: providerType,
    capabilities: shape === "anthropic"
      ? ["chat", "tool_use", "vision"]
      : inferOpenaiCapabilities(m.id),
    ownedBy: m.owned_by,
    createdAt: m.created ??
      (m.created_at ? Date.parse(m.created_at) / 1000 : undefined),
  }));
}

/**
 * Create a {@link ModelLister} for the given provider: one `GET` of the
 * registry's `models_url` (or `baseUrl`), authenticated per the registry's
 * auth scheme, parsed per vendor shape.
 *
 * @param providerType The provider to list models for.
 * @param apiKey Required for cloud providers, ignored for Ollama.
 * @param baseUrl Optional URL override (for Ollama or custom endpoints).
 * @throws If the provider does not support listing or a required API key is missing.
 */
export function createModelLister(
  providerType: ProviderType,
  apiKey?: string,
  baseUrl?: string,
): ModelLister {
  const entry = lookup(providerType);
  if (!entry?.supports_model_listing || !entry.models_url) {
    throw new Error(
      `Model listing is not supported for ${providerType} provider via this interface`,
    );
  }
  if (providerType !== "ollama" && !apiKey) {
    throw new Error(`${providerType} requires an API key`);
  }
  const url = baseUrl ?? entry.models_url;
  const headers = listingHeaders(entry, apiKey);
  return {
    async listModels(): Promise<AvailableModel[]> {
      const res = await fetch(url, {
        headers,
        signal: AbortSignal.timeout(LISTING_TIMEOUT_MS),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(
          `${providerType} model listing failed (${res.status}): ${text}`,
        );
      }
      return parseModelListing(providerType, await res.json());
    },
  };
}

/** Deadline for a model-listing request. */
export const LISTING_TIMEOUT_MS = 15_000;
