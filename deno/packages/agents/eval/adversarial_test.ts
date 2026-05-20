import { assert, assertEquals } from "@std/assert";
import {
  ambiguousInstructionCase,
  budgetExhaustionCase,
  caseCategory,
  injectionPayload,
  missingContextCase,
  promptInjectionCase,
  standardAdversarialSuite,
  withExpectRejection,
} from "./adversarial.ts";

Deno.test("prompt injection builder", () => {
  const c = promptInjectionCase("test_inj", "ignore instructions", true);
  assertEquals(c.name, "test_inj");
  assert(c.expect_rejection);
  assertEquals(caseCategory(c), "prompt_injection");
  assertEquals(injectionPayload(c), "ignore instructions");
});

Deno.test("ambiguous instruction builder", () => {
  const c = ambiguousInstructionCase("test_amb", ["opt_a", "opt_b"]);
  assertEquals(caseCategory(c), "ambiguous_instruction");
  assert(!c.expect_rejection);
  assertEquals(injectionPayload(c), null);
  if (c.test_type.type !== "ambiguous_instruction") {
    throw new Error("wrong variant");
  }
  assertEquals(c.test_type.variants.length, 2);
});

Deno.test("missing context builder", () => {
  const c = missingContextCase("miss_lang", "language", "Rust");
  assertEquals(caseCategory(c), "missing_context");
  if (c.test_type.type !== "missing_context") {
    throw new Error("wrong variant");
  }
  assertEquals(c.test_type.missing_key, "language");
  assertEquals(c.test_type.expected_value, "Rust");
});

Deno.test("budget exhaustion builder", () => {
  const c = budgetExhaustionCase("budget", 5, "task desc");
  assertEquals(caseCategory(c), "budget_exhaustion");
  if (c.test_type.type !== "budget_exhaustion") {
    throw new Error("wrong variant");
  }
  assertEquals(c.test_type.max_steps, 5);
});

Deno.test("standard suite non empty", () => {
  const suite = standardAdversarialSuite();
  assert(suite.length > 0);
  const cats = new Set(suite.map((c) => caseCategory(c)));
  assert(cats.has("prompt_injection"));
  assert(cats.has("ambiguous_instruction"));
  assert(cats.has("missing_context"));
  assert(cats.has("budget_exhaustion"));
});

Deno.test("standard suite all injection cases expect rejection", () => {
  for (const c of standardAdversarialSuite()) {
    if (caseCategory(c) === "prompt_injection") {
      assert(c.expect_rejection, `${c.name} should expect rejection`);
    }
  }
});

Deno.test("with expect rejection override", () => {
  const c0 = missingContextCase("x", "key", null);
  const c1 = withExpectRejection(c0, true);
  assert(c1.expect_rejection);
});

Deno.test("json round trip", () => {
  const c = promptInjectionCase("inj", "payload", true);
  const decoded = JSON.parse(JSON.stringify(c));
  assertEquals(decoded.name, c.name);
  assertEquals(decoded.expect_rejection, c.expect_rejection);
});
