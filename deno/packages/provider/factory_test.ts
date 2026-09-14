import { assert, assertEquals, assertThrows } from "@std/assert";
import { BedrockProvider } from "./bedrock.ts";
import { VertexAiProvider } from "./vertex.ts";
import { PROVIDER_REGISTRY } from "./registry.ts";
import { ChatProviderFactory } from "./factory.ts";
import { defaultModel, type ProviderConfig } from "./types.ts";

Deno.test("ChatProviderFactory.create - ollama no key required", () => {
  const config: ProviderConfig = {
    provider: "ollama",
    model: "llama3.1",
  };
  const provider = ChatProviderFactory.create(config);
  assertEquals(provider.name, "ollama");
});

Deno.test("ChatProviderFactory.create - anthropic requires key", () => {
  const config: ProviderConfig = {
    provider: "anthropic",
    model: "claude-3",
  };
  assertThrows(
    () => ChatProviderFactory.create(config),
    Error,
    "requires an API key",
  );
});

Deno.test("ChatProviderFactory.create - anthropic with key", () => {
  const config: ProviderConfig = {
    provider: "anthropic",
    model: "claude-3-sonnet",
    api_key: "sk-ant-test",
  };
  const provider = ChatProviderFactory.create(config);
  assertEquals(provider.name, "anthropic");
});

Deno.test("ChatProviderFactory.create - openai with key", () => {
  const config: ProviderConfig = {
    provider: "openai",
    model: "gpt-4",
    api_key: "sk-test",
  };
  const provider = ChatProviderFactory.create(config);
  assertEquals(provider.name, "openai");
});

Deno.test("ChatProviderFactory.create - groq with key", () => {
  const config: ProviderConfig = {
    provider: "groq",
    model: "llama-3.3-70b-versatile",
    api_key: "gsk_test",
  };
  const provider = ChatProviderFactory.create(config);
  assertEquals(provider.name, "groq");
});

Deno.test("ChatProviderFactory.create - together with key", () => {
  const config: ProviderConfig = {
    provider: "together",
    model: "meta-llama/Llama-3.1-8B-Instruct",
    api_key: "tok_test",
  };
  const provider = ChatProviderFactory.create(config);
  assertEquals(provider.name, "together");
});

Deno.test("ChatProviderFactory.create - fireworks with key", () => {
  const config: ProviderConfig = {
    provider: "fireworks",
    model: "llama-v3p1-8b-instruct",
    api_key: "fw_test",
  };
  const provider = ChatProviderFactory.create(config);
  assertEquals(provider.name, "fireworks");
});

Deno.test("ChatProviderFactory.create - google with key", () => {
  const config: ProviderConfig = {
    provider: "google",
    model: "gemini-pro",
    api_key: "test-key",
  };
  const provider = ChatProviderFactory.create(config);
  assertEquals(provider.name, "google");
});

Deno.test("ChatProviderFactory.create - google requires key", () => {
  const config: ProviderConfig = {
    provider: "google",
    model: "gemini-pro",
  };
  assertThrows(
    () => ChatProviderFactory.create(config),
    Error,
    "requires an API key",
  );
});

Deno.test("ChatProviderFactory.create - openai requires key", () => {
  const config: ProviderConfig = {
    provider: "openai",
    model: "gpt-4",
  };
  assertThrows(
    () => ChatProviderFactory.create(config),
    Error,
    "requires an API key",
  );
});

Deno.test("ChatProviderFactory.create - custom provider rejected", () => {
  const config: ProviderConfig = {
    provider: "custom",
    model: "custom-model",
    api_key: "key",
  };
  assertThrows(
    () => ChatProviderFactory.create(config),
    Error,
    "is not a chat provider",
  );
});

Deno.test("ChatProviderFactory.create - ollama with custom url", () => {
  const config: ProviderConfig = {
    provider: "ollama",
    model: "llama3.1",
    base_url: "http://custom:8080",
  };
  const provider = ChatProviderFactory.create(config);
  assertEquals(provider.name, "ollama");
});

Deno.test("factory: bedrock and vertex-ai resolve to their own providers, not the wire-format lookalikes", () => {
  const bedrock = ChatProviderFactory.create({
    provider: "bedrock",
    model: "anthropic.claude-sonnet-4-20250514-v1:0",
    options: {
      region: "us-east-1",
      access_key_id: "AKIA",
      secret_access_key: "s",
    },
  } as ProviderConfig);
  assert(bedrock instanceof BedrockProvider);
  const vertex = ChatProviderFactory.create({
    provider: "vertex-ai",
    model: "gemini-2.0-flash",
    options: {
      project_id: "p",
      credentials: {
        client_email: "e",
        private_key: "k",
        token_uri: "https://oauth2.googleapis.com/token",
      },
    },
  } as ProviderConfig);
  assert(vertex instanceof VertexAiProvider);
});

Deno.test("registry default models agree with defaultModel()", () => {
  for (const entry of PROVIDER_REGISTRY) {
    assertEquals(
      entry.default_model,
      defaultModel(entry.provider_type),
      entry.provider_type,
    );
  }
});
