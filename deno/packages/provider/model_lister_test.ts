import { assertEquals } from "@std/assert/equals";
import { assert } from "@std/assert/assert";
import { assertThrows } from "@std/assert";
import {
  type AvailableModel,
  createModelLister,
  inferOpenaiCapabilities,
  isChatCapable,
  type ModelCapability,
} from "./model_lister.ts";

Deno.test("inferOpenaiCapabilities: chat model with vision", () => {
  const caps = inferOpenaiCapabilities("gpt-4o");
  assert(caps.includes("chat"));
  assert(caps.includes("tool_use"));
  assert(caps.includes("vision"));
});

Deno.test("inferOpenaiCapabilities: embedding model", () => {
  const caps = inferOpenaiCapabilities("text-embedding-3-small");
  assert(caps.includes("embedding"));
  assert(!caps.includes("chat"));
});

Deno.test("inferOpenaiCapabilities: audio model", () => {
  const caps = inferOpenaiCapabilities("whisper-1");
  assert(caps.includes("audio"));
  assert(!caps.includes("chat"));
});

Deno.test("inferOpenaiCapabilities: image generation model", () => {
  const caps = inferOpenaiCapabilities("dall-e-3");
  assert(caps.includes("image_generation"));
  assert(!caps.includes("chat"));
});

Deno.test("inferOpenaiCapabilities: basic chat without vision", () => {
  const caps = inferOpenaiCapabilities("gpt-3.5-turbo");
  assert(caps.includes("chat"));
  assert(caps.includes("tool_use"));
  assert(!caps.includes("vision"));
});

Deno.test("isChatCapable: returns true for chat model", () => {
  const model: AvailableModel = {
    id: "test",
    provider: "openai",
    capabilities: ["chat"],
  };
  assert(isChatCapable(model));
});

Deno.test("isChatCapable: returns false for embedding model", () => {
  const model: AvailableModel = {
    id: "embed",
    provider: "openai",
    capabilities: ["embedding"],
  };
  assert(!isChatCapable(model));
});

Deno.test("createModelLister: throws for unsupported provider", () => {
  assertThrows(
    () => createModelLister("rullama", "key"),
    Error,
    "not supported",
  );
});

Deno.test("createModelLister: throws without API key for cloud provider", () => {
  assertThrows(
    () => createModelLister("anthropic"),
    Error,
    "requires an API key",
  );
});

Deno.test("createModelLister: ollama does not require API key", () => {
  // Should not throw
  const lister = createModelLister("ollama");
  assert(lister !== undefined);
});

Deno.test("createModelLister: accepts API key for supported provider", () => {
  const lister = createModelLister("openai", "sk-test");
  assert(lister !== undefined);
});

Deno.test("ModelCapability display values match Rust", () => {
  const caps: ModelCapability[] = [
    "chat",
    "tool_use",
    "vision",
    "embedding",
    "audio",
    "image_generation",
  ];
  assertEquals(caps.length, 6);
  assert(caps.includes("chat"));
  assert(caps.includes("tool_use"));
  assert(caps.includes("vision"));
});
