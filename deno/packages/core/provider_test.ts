import { assertEquals } from "@std/assert";
import { ChatOptions } from "./mod.ts";

Deno.test("ChatOptions default values", () => {
  const opts = ChatOptions.create();
  assertEquals(opts.temperature, 0.7);
  assertEquals(opts.max_tokens, 4096);
});

Deno.test("ChatOptions builder", () => {
  const opts = ChatOptions.create()
    .setTemperature(0.5)
    .setMaxTokens(2048)
    .setSystem("Test");
  assertEquals(opts.temperature, 0.5);
  assertEquals(opts.max_tokens, 2048);
  assertEquals(opts.system, "Test");
});

Deno.test("ChatOptions.deterministic", () => {
  const opts = ChatOptions.deterministic(50);
  assertEquals(opts.temperature, 0.0);
  assertEquals(opts.max_tokens, 50);
});

Deno.test("ChatOptions.factual", () => {
  const opts = ChatOptions.factual(200);
  assertEquals(opts.temperature, 0.1);
  assertEquals(opts.max_tokens, 200);
  assertEquals(opts.top_p, 0.9);
});

Deno.test("ChatOptions - the default temperature is marked, an explicit one is not", () => {
  const defaulted = new ChatOptions();
  assertEquals(defaulted.temperature, ChatOptions.DEFAULT_TEMPERATURE);
  assertEquals(defaulted.temperature_defaulted, true);
  assertEquals(ChatOptions.explicitTemperature(defaulted), undefined);

  const given = new ChatOptions({ temperature: 0.7 });
  assertEquals(given.temperature_defaulted, undefined);
  assertEquals(ChatOptions.explicitTemperature(given), 0.7);

  const built = ChatOptions.create().setTemperature(0.7);
  assertEquals(built.temperature_defaulted, false);
  assertEquals(ChatOptions.explicitTemperature(built), 0.7);

  assertEquals(ChatOptions.explicitTemperature({ temperature: 0 }), 0);
  assertEquals(ChatOptions.explicitTemperature({}), undefined);
  // Copies of defaulted options stay defaulted.
  assertEquals(new ChatOptions(defaulted).temperature_defaulted, true);
});

Deno.test("ChatOptions - extra_headers / extra_body: kept, copied, never serialized", () => {
  const opts = new ChatOptions({
    extra_headers: { "anthropic-beta": "x" },
    extra_body: { fallbacks: "default" },
  });
  assertEquals(opts.extra_headers, { "anthropic-beta": "x" });
  assertEquals(opts.extra_body, { fallbacks: "default" });
  assertEquals(new ChatOptions(opts).extra_body, { fallbacks: "default" });
  assertEquals(new ChatOptions().extra_headers, undefined);
  assertEquals(opts.toJSON(), { temperature: 0.7, max_tokens: 4096 });
});

Deno.test("ChatOptions - top_k: no default, builder, serialized; the default mark is not", () => {
  assertEquals(new ChatOptions().top_k, undefined);
  const opts = ChatOptions.create().setTopK(40);
  assertEquals(opts.top_k, 40);
  assertEquals(new ChatOptions({ top_k: 3 }).top_k, 3);
  assertEquals(opts.toJSON(), {
    temperature: 0.7,
    max_tokens: 4096,
    top_k: 40,
  });
});

Deno.test("ChatOptions.creative", () => {
  const opts = ChatOptions.creative(400);
  assertEquals(opts.temperature, 0.3);
  assertEquals(opts.max_tokens, 400);
});
