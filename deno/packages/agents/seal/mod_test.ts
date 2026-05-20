import { assert, assertEquals } from "@std/assert";
import { defaultSealConfig, SealProcessor } from "./mod.ts";

Deno.test("SEAL processor creation", () => {
  const processor = SealProcessor.withDefaults();
  assert(processor.config.enable_coreference);
  assert(processor.config.enable_query_cores);
  assert(processor.config.enable_learning);
  assert(processor.config.enable_reflection);
});

Deno.test("SEAL config default", () => {
  const config = defaultSealConfig();
  assert(config.enable_coreference);
  assertEquals(config.max_reflection_retries, 2);
  assert(config.min_coreference_confidence > 0);
});

Deno.test("initConversation updates the learning coordinator", () => {
  const processor = SealProcessor.withDefaults();
  processor.initConversation("test-conv-123");
  assertEquals(processor.learning_coordinator.local.conversation_id, "test-conv-123");
});
