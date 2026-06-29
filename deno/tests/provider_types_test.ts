/**
 * Cross-package integration test: Provider factory and registry.
 *
 * Verifies that @rullama/providers factory creates correct provider types,
 * and that the registry lookup returns correct entries for each ProviderType.
 */

import {
  assert,
  assertEquals,
  assertThrows,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  ChatProviderFactory,
  defaultModel,
  lookup,
  parseProviderType,
  PROVIDER_REGISTRY,
  type ProviderEntry,
  type ProviderType,
  requiresApiKey,
} from "@rullama/provider";

// ---------------------------------------------------------------------------
// Registry lookup tests
// ---------------------------------------------------------------------------

Deno.test("PROVIDER_REGISTRY contains all expected providers", () => {
  const expectedProviders: ProviderType[] = [
    "openai",
    "anthropic",
    "google",
    "groq",
    "ollama",
    "together",
    "fireworks",
    "anyscale",
    "openai-responses",
    "rullama",
  ];

  for (const pt of expectedProviders) {
    const entry = lookup(pt);
    assert(entry !== undefined, `Registry should contain entry for '${pt}'`);
    assertEquals(entry!.provider_type, pt);
  }
});

Deno.test("lookup returns undefined for unknown provider", () => {
  const entry = lookup("nonexistent" as ProviderType);
  assertEquals(entry, undefined);
});

Deno.test("registry entries have required fields", () => {
  for (const entry of PROVIDER_REGISTRY) {
    assert(entry.provider_type.length > 0, "provider_type should be non-empty");
    assert(entry.chat_protocol.length > 0, "chat_protocol should be non-empty");
    assert(
      entry.default_base_url.length > 0,
      "default_base_url should be non-empty",
    );
    assert(entry.default_model.length > 0, "default_model should be non-empty");
    assert(entry.auth !== undefined, "auth should be defined");
  }
});

// ---------------------------------------------------------------------------
// Protocol mapping tests
// ---------------------------------------------------------------------------

Deno.test("anthropic uses anthropic_messages protocol", () => {
  const entry = lookup("anthropic")!;
  assertEquals(entry.chat_protocol, "anthropic_messages");
});

Deno.test("openai uses openai_chat_completions protocol", () => {
  const entry = lookup("openai")!;
  assertEquals(entry.chat_protocol, "openai_chat_completions");
});

Deno.test("google uses gemini_generate_content protocol", () => {
  const entry = lookup("google")!;
  assertEquals(entry.chat_protocol, "gemini_generate_content");
});

Deno.test("ollama uses ollama_chat protocol", () => {
  const entry = lookup("ollama")!;
  assertEquals(entry.chat_protocol, "ollama_chat");
});

Deno.test("groq uses openai_chat_completions protocol (compatible)", () => {
  const entry = lookup("groq")!;
  assertEquals(entry.chat_protocol, "openai_chat_completions");
});

// ---------------------------------------------------------------------------
// Factory creates correct provider types
// ---------------------------------------------------------------------------

Deno.test("factory creates Anthropic provider", () => {
  const provider = ChatProviderFactory.create({
    provider: "anthropic",
    model: "claude-sonnet-4-20250514",
    api_key: "test-key",
  });
  assertEquals(provider.name, "anthropic");
});

Deno.test("factory creates OpenAI-compatible provider for openai", () => {
  const provider = ChatProviderFactory.create({
    provider: "openai",
    model: "gpt-4o",
    api_key: "test-key",
  });
  assertEquals(provider.name, "openai");
});

Deno.test("factory creates OpenAI-compatible provider for groq", () => {
  const provider = ChatProviderFactory.create({
    provider: "groq",
    model: "llama-3.3-70b-versatile",
    api_key: "test-key",
  });
  assertEquals(provider.name, "groq");
});

Deno.test("factory creates Google provider", () => {
  const provider = ChatProviderFactory.create({
    provider: "google",
    model: "gemini-2.0-flash",
    api_key: "test-key",
  });
  assertEquals(provider.name, "google");
});

Deno.test("factory creates Ollama provider without API key", () => {
  const provider = ChatProviderFactory.create({
    provider: "ollama",
    model: "llama3.1",
  });
  assertEquals(provider.name, "ollama");
});

Deno.test("factory throws for unknown provider type", () => {
  assertThrows(
    () =>
      ChatProviderFactory.create({
        provider: "nonexistent" as ProviderType,
        model: "test",
        api_key: "key",
      }),
    Error,
    "not a chat provider",
  );
});

// ---------------------------------------------------------------------------
// parseProviderType tests
// ---------------------------------------------------------------------------

Deno.test("parseProviderType parses valid types", () => {
  assertEquals(parseProviderType("anthropic"), "anthropic");
  assertEquals(parseProviderType("openai"), "openai");
  assertEquals(parseProviderType("google"), "google");
  assertEquals(parseProviderType("gemini"), "google"); // alias
  assertEquals(parseProviderType("ollama"), "ollama");
  assertEquals(parseProviderType("groq"), "groq");
});

Deno.test("parseProviderType returns undefined for invalid", () => {
  assertEquals(parseProviderType("invalid"), undefined);
  assertEquals(parseProviderType(""), undefined);
});

// ---------------------------------------------------------------------------
// defaultModel and requiresApiKey
// ---------------------------------------------------------------------------

Deno.test("defaultModel returns non-empty strings", () => {
  const providers: ProviderType[] = [
    "anthropic",
    "openai",
    "google",
    "groq",
    "ollama",
    "together",
    "fireworks",
    "anyscale",
  ];
  for (const pt of providers) {
    const model = defaultModel(pt);
    assert(model.length > 0, `defaultModel for '${pt}' should be non-empty`);
  }
});

Deno.test("requiresApiKey is false only for ollama", () => {
  assertEquals(requiresApiKey("ollama"), false);
  assertEquals(requiresApiKey("anthropic"), true);
  assertEquals(requiresApiKey("openai"), true);
  assertEquals(requiresApiKey("google"), true);
});
