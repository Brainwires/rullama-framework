import { assertEquals } from "@std/assert";
import {
  createProviderConfig,
  defaultModel,
  parseProviderType,
  requiresApiKey,
} from "./types.ts";

Deno.test("parseProviderType - known providers", () => {
  assertEquals(parseProviderType("anthropic"), "anthropic");
  assertEquals(parseProviderType("openai"), "openai");
  assertEquals(parseProviderType("google"), "google");
  assertEquals(parseProviderType("gemini"), "google");
  assertEquals(parseProviderType("groq"), "groq");
  assertEquals(parseProviderType("ollama"), "ollama");
  assertEquals(parseProviderType("rullama"), "rullama");
  assertEquals(parseProviderType("together"), "together");
  assertEquals(parseProviderType("fireworks"), "fireworks");
  assertEquals(parseProviderType("anyscale"), "anyscale");
  assertEquals(parseProviderType("openai-responses"), "openai-responses");
  assertEquals(parseProviderType("openai_responses"), "openai-responses");
  assertEquals(parseProviderType("custom"), "custom");
});

Deno.test("parseProviderType - unknown returns undefined", () => {
  assertEquals(parseProviderType("unknown"), undefined);
  assertEquals(parseProviderType("nonexistent"), undefined);
});

Deno.test("parseProviderType - case insensitive", () => {
  assertEquals(parseProviderType("ANTHROPIC"), "anthropic");
  assertEquals(parseProviderType("OpenAI"), "openai");
  assertEquals(parseProviderType("Google"), "google");
});

Deno.test("defaultModel - returns correct default for each provider", () => {
  assertEquals(defaultModel("anthropic"), "claude-sonnet-4-20250514");
  assertEquals(defaultModel("openai"), "gpt-5-mini");
  assertEquals(defaultModel("google"), "gemini-2.5-flash");
  assertEquals(defaultModel("groq"), "llama-3.3-70b-versatile");
  assertEquals(defaultModel("ollama"), "llama3.3");
  assertEquals(defaultModel("rullama"), "gpt-5-mini");
});

Deno.test("requiresApiKey - ollama does not require key", () => {
  assertEquals(requiresApiKey("ollama"), false);
});

Deno.test("requiresApiKey - cloud providers require keys", () => {
  assertEquals(requiresApiKey("anthropic"), true);
  assertEquals(requiresApiKey("openai"), true);
  assertEquals(requiresApiKey("google"), true);
  assertEquals(requiresApiKey("groq"), true);
});

Deno.test("createProviderConfig - minimal config", () => {
  const config = createProviderConfig("anthropic", "claude-3");
  assertEquals(config.provider, "anthropic");
  assertEquals(config.model, "claude-3");
  assertEquals(config.api_key, undefined);
  assertEquals(config.base_url, undefined);
});

Deno.test("createProviderConfig - with optional fields", () => {
  const config = createProviderConfig("openai", "gpt-4");
  config.api_key = "sk-test";
  config.base_url = "https://api.example.com";
  assertEquals(config.api_key, "sk-test");
  assertEquals(config.base_url, "https://api.example.com");
});
