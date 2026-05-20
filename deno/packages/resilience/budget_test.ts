import { assert, assertEquals } from "@std/assert";
import { Message, createUsage } from "@brainwires/core";
import { approxInputTokens, BudgetGuard } from "./budget.ts";
import { ResilienceError } from "./error.ts";

Deno.test("guard tracks tokens", () => {
  const g = new BudgetGuard({ max_tokens: 100, max_usd_cents: null, max_rounds: null });
  g.recordUsage(createUsage(40, 40)); // total 80
  assertEquals(g.tokensConsumed(), 80);
  g.check(); // under budget
  g.recordUsage(createUsage(30, 0)); // total 110
  assertEquals(g.tokensConsumed(), 110);
  let thrown: unknown;
  try {
    g.check();
  } catch (e) {
    thrown = e;
  }
  assert(thrown instanceof ResilienceError);
  assertEquals((thrown as ResilienceError).kind, "budget_exceeded");
  assertEquals((thrown as ResilienceError).detail.kind, "tokens");
});

Deno.test("guard tracks rounds", () => {
  const g = new BudgetGuard({ max_tokens: null, max_usd_cents: null, max_rounds: 2 });
  g.checkAndTick();
  g.checkAndTick();
  let thrown: unknown;
  try {
    g.checkAndTick();
  } catch (e) {
    thrown = e;
  }
  assert(thrown instanceof ResilienceError);
  assertEquals((thrown as ResilienceError).detail.kind, "rounds");
});

Deno.test("guard reset zeros everything", () => {
  const g = new BudgetGuard({ max_tokens: 100, max_usd_cents: null, max_rounds: 5 });
  g.recordUsage(createUsage(5, 5));
  g.checkAndTick();
  g.reset();
  assertEquals(g.tokensConsumed(), 0);
  assertEquals(g.roundsConsumed(), 0);
});

Deno.test("approx tokens from text", () => {
  const text = "abcd".repeat(40); // 160 chars → 40 tokens
  const n = approxInputTokens([Message.user(text)]);
  assertEquals(n, 40);
});
