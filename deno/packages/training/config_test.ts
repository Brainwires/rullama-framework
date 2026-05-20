import { assert, assertEquals } from "@std/assert";
import {
  defaultHyperparams,
  defaultLoraConfig,
  dpoAlignment,
  isQuantized,
  orpoAlignment,
  quantizationBits,
} from "./config.ts";

Deno.test("hyperparams defaults", () => {
  const h = defaultHyperparams();
  assertEquals(h.epochs, 3);
  assertEquals(h.batch_size, 4);
  assertEquals(h.learning_rate, 2e-5);
});

Deno.test("lora config defaults", () => {
  const c = defaultLoraConfig();
  assertEquals(c.rank, 16);
  assertEquals(c.target_modules.length, 4);
  assertEquals(c.method.kind, "lora");
});

Deno.test("adapter method quantized", () => {
  assert(!isQuantized({ kind: "lora" }));
  assert(isQuantized({ kind: "qlora", bits: 4 }));
  assertEquals(quantizationBits({ kind: "qlora", bits: 4 }), 4);
  assertEquals(quantizationBits({ kind: "dora" }), null);
});

Deno.test("alignment helpers", () => {
  const dpo = dpoAlignment();
  assertEquals(dpo.kind, "dpo");
  if (dpo.kind === "dpo") assertEquals(dpo.beta, 0.1);
  const orpo = orpoAlignment();
  assertEquals(orpo.kind, "orpo");
  if (orpo.kind === "orpo") assertEquals(orpo.lambda, 0.5);
});
