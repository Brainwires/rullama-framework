import { assert, type assertEquals } from "@std/assert";
import {
  allEnabled,
  defaultLocalInferenceConfig,
  InferenceTimer,
  tier1Enabled,
  tier2Enabled,
} from "./config.ts";

Deno.test("default config has everything off", () => {
  const c = defaultLocalInferenceConfig();
  assert(!c.routing_enabled);
  assert(!c.summarization_enabled);
});

Deno.test("tier1Enabled turns on tier-1 flags only", () => {
  const c = tier1Enabled();
  assert(c.routing_enabled);
  assert(c.validation_enabled);
  assert(c.complexity_enabled);
  assert(!c.summarization_enabled);
});

Deno.test("tier2Enabled turns on tier-2 flags only", () => {
  const c = tier2Enabled();
  assert(!c.routing_enabled);
  assert(c.summarization_enabled);
  assert(c.retrieval_gating_enabled);
  assert(c.relevance_scoring_enabled);
  assert(c.strategy_selection_enabled);
  assert(c.entity_enhancement_enabled);
});

Deno.test("allEnabled turns on all flags", () => {
  const c = allEnabled();
  assert(c.routing_enabled);
  assert(c.summarization_enabled);
});

Deno.test("timer measures elapsed", async () => {
  const t = new InferenceTimer("test", "m");
  await new Promise((r) => setTimeout(r, 10));
  assert(t.elapsedMs() >= 10);
});
