import { assertEquals, assertNotEquals } from "@std/assert";
import { lookup, PROVIDER_REGISTRY } from "./registry.ts";

Deno.test("lookup - known providers return entries", () => {
  assertNotEquals(lookup("openai"), undefined);
  assertNotEquals(lookup("groq"), undefined);
  assertNotEquals(lookup("together"), undefined);
  assertNotEquals(lookup("fireworks"), undefined);
  assertNotEquals(lookup("anyscale"), undefined);
  assertNotEquals(lookup("anthropic"), undefined);
  assertNotEquals(lookup("google"), undefined);
  assertNotEquals(lookup("ollama"), undefined);
  assertNotEquals(lookup("openai-responses"), undefined);
  assertNotEquals(lookup("rullama"), undefined);
});

Deno.test("lookup - unknown provider returns undefined", () => {
  assertEquals(lookup("custom"), undefined);
});

Deno.test("OpenAI-compat providers share protocol", () => {
  const groq = lookup("groq")!;
  const together = lookup("together")!;
  const fireworks = lookup("fireworks")!;
  const anyscale = lookup("anyscale")!;

  assertEquals(groq.chat_protocol, "openai_chat_completions");
  assertEquals(together.chat_protocol, "openai_chat_completions");
  assertEquals(fireworks.chat_protocol, "openai_chat_completions");
  assertEquals(anyscale.chat_protocol, "openai_chat_completions");
});

Deno.test("Anthropic uses custom header auth", () => {
  const anthropic = lookup("anthropic")!;
  assertEquals(anthropic.chat_protocol, "anthropic_messages");
  assertEquals(anthropic.auth, { type: "custom_header", header: "x-api-key" });
});

Deno.test("Ollama uses no auth", () => {
  const ollama = lookup("ollama")!;
  assertEquals(ollama.auth, { type: "none" });
  assertEquals(ollama.chat_protocol, "ollama_chat");
});

Deno.test("Google uses Gemini protocol", () => {
  const google = lookup("google")!;
  assertEquals(google.chat_protocol, "gemini_generate_content");
});

Deno.test("PROVIDER_REGISTRY has all entries", () => {
  // We should have entries for the chat providers
  const types = PROVIDER_REGISTRY.map((e) => e.provider_type);
  assertEquals(types.includes("openai"), true);
  assertEquals(types.includes("anthropic"), true);
  assertEquals(types.includes("google"), true);
  assertEquals(types.includes("ollama"), true);
  assertEquals(types.includes("groq"), true);
  assertEquals(types.includes("together"), true);
  assertEquals(types.includes("fireworks"), true);
  assertEquals(types.includes("anyscale"), true);
});
