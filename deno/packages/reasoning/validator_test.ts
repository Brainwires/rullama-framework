import { assert, assertEquals } from "@std/assert";
import {
  isInvalid,
  isValid,
  parseValidation,
  validateHeuristic,
  type ValidationResult,
} from "./validator.ts";

Deno.test("valid / invalid tags", () => {
  const v: ValidationResult = { kind: "valid", confidence: 0.9 };
  assert(isValid(v));
  assert(!isInvalid(v));
  const i: ValidationResult = { kind: "invalid", reason: "x", severity: 0.5, confidence: 0.8 };
  assert(!isValid(i));
  assert(isInvalid(i));
});

Deno.test("heuristic catches refusal", () => {
  const r = validateHeuristic("Write a poem", "I'm sorry, I cannot write poems as an AI assistant.");
  assertEquals(r.kind, "invalid");
});

Deno.test("heuristic passes good response", () => {
  const r = validateHeuristic("Calculate 2+2", "The result of 2+2 is 4.");
  assertEquals(r.kind, "valid");
});

Deno.test("parseValidation reads VALID / INVALID / ambiguous", () => {
  assertEquals(parseValidation("VALID").kind, "valid");
  const inv = parseValidation("INVALID: Response is off-topic");
  assertEquals(inv.kind, "invalid");
  if (inv.kind === "invalid") assert(inv.reason.toLowerCase().includes("off-topic"));
  assertEquals(parseValidation("Maybe?").kind, "skipped");
});
