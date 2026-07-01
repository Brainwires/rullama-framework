// Example: Provider Factory & Model Listing
// Demonstrates browsing the provider registry, building configs, creating
// providers via ChatProviderFactory, and inspecting model capabilities.
// Run: deno run deno/examples/providers/provider_factory.ts

import {
  ChatProviderFactory,
  createModelLister,
  createProviderConfig,
  defaultModel,
  inferOpenaiCapabilities,
  lookup,
  PROVIDER_REGISTRY,
  type ProviderType,
  requiresApiKey,
} from "@rullama/provider";

async function main() {
  console.log("=== Provider Factory & Model Listing Example ===\n");

  // 1. Browse the provider registry
  console.log("--- Known Chat Providers ---");
  for (const entry of PROVIDER_REGISTRY) {
    console.log(
      `  ${entry.provider_type.padEnd(18)} | protocol: ${
        entry.chat_protocol.padEnd(28)
      } | default model: ${entry.default_model}`,
    );
  }
  console.log();

  // 2. Build configs with createProviderConfig
  console.log("--- Building Provider Configs ---");

  // Ollama does not require an API key
  const ollamaConfig = createProviderConfig("ollama", "llama3.3");
  ollamaConfig.base_url = "http://localhost:11434";
  console.log("Ollama config:", ollamaConfig);

  // OpenAI needs an API key
  const openaiConfig = createProviderConfig("openai", "gpt-5-mini");
  openaiConfig.api_key = "sk-demo-key-not-real";
  console.log("OpenAI config:", openaiConfig);

  // Groq uses the OpenAI-compatible protocol with a different base URL
  const groqConfig = createProviderConfig("groq", "llama-3.3-70b-versatile");
  groqConfig.api_key = "gsk-demo-key-not-real";
  console.log("Groq   config:", groqConfig);
  console.log();

  // 3. Create providers via the factory
  console.log("--- Creating Providers via ChatProviderFactory ---");

  // Ollama succeeds without a key
  try {
    const provider = ChatProviderFactory.create(ollamaConfig);
    console.log(`  Created '${provider.name}' provider successfully`);
  } catch (e) {
    console.log(`  Failed to create Ollama provider: ${(e as Error).message}`);
  }

  // OpenAI succeeds with a key (no network call at creation time)
  try {
    const provider = ChatProviderFactory.create(openaiConfig);
    console.log(`  Created '${provider.name}' provider successfully`);
  } catch (e) {
    console.log(`  Failed to create OpenAI provider: ${(e as Error).message}`);
  }

  // Groq is dispatched via OpenAI-compatible protocol
  try {
    const provider = ChatProviderFactory.create(groqConfig);
    console.log(`  Created '${provider.name}' provider successfully`);
  } catch (e) {
    console.log(`  Failed to create Groq provider: ${(e as Error).message}`);
  }

  // Unknown/unsupported provider types are rejected
  const customConfig = createProviderConfig("custom", "some-model");
  customConfig.api_key = "demo";
  try {
    ChatProviderFactory.create(customConfig);
    console.log("  Unexpected: custom should not have a registry entry");
  } catch (e) {
    console.log(`  Expected rejection for custom: ${(e as Error).message}`);
  }
  console.log();

  // 4. Default models per provider
  console.log("--- Default Models ---");
  const providerTypes: ProviderType[] = [
    "anthropic",
    "openai",
    "google",
    "groq",
    "ollama",
    "together",
  ];
  for (const pt of providerTypes) {
    console.log(`  ${pt.padEnd(12)} -> ${defaultModel(pt)}`);
  }
  console.log();

  // 5. Model lister creation (no actual API calls)
  console.log("--- Model Lister Availability ---");
  for (const pt of providerTypes) {
    const needsKey = requiresApiKey(pt);
    const key = needsKey ? "demo-key" : undefined;
    try {
      createModelLister(pt, key);
      console.log(
        `  ${pt.padEnd(12)} -> ModelLister created (would query API)`,
      );
    } catch (e) {
      console.log(`  ${pt.padEnd(12)} -> ${(e as Error).message}`);
    }
  }
  console.log();

  // 6. Inspect model capabilities helper
  console.log("--- Capability Inference (OpenAI-format IDs) ---");
  const testIds = [
    "gpt-4o",
    "gpt-3.5-turbo",
    "text-embedding-3-small",
    "dall-e-3",
    "whisper-1",
  ];
  for (const id of testIds) {
    const caps = inferOpenaiCapabilities(id);
    console.log(`  ${id.padEnd(30)} -> [${caps.join(", ")}]`);
  }

  // 7. Registry lookup for specific providers
  console.log("\n--- Registry Lookup ---");
  const anthropicEntry = lookup("anthropic");
  if (anthropicEntry) {
    console.log(`  Anthropic protocol: ${anthropicEntry.chat_protocol}`);
    console.log(`  Anthropic auth: ${JSON.stringify(anthropicEntry.auth)}`);
    console.log(
      `  Supports model listing: ${anthropicEntry.supports_model_listing}`,
    );
  }

  console.log(
    "\nDone! In a real application you would call provider.chat() or",
  );
  console.log("lister.listModels() to interact with the APIs.");
}

await main();
