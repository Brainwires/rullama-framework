import { assert, assertEquals } from "@std/assert";
import { AlwaysFailCase, AlwaysPassCase, StochasticCase } from "./case.ts";

Deno.test("always pass case", async () => {
  const c = new AlwaysPassCase("test").with_duration(5);
  const r = await c.run(0);
  assert(r.success);
  assertEquals(r.trial_id, 0);
  assertEquals(r.duration_ms, 5);
});

Deno.test("always fail case", async () => {
  const c = new AlwaysFailCase("test", "oops");
  const r = await c.run(3);
  assert(!r.success);
  assertEquals(r.trial_id, 3);
  assertEquals(r.error, "oops");
});

Deno.test("stochastic case reproducible", async () => {
  const c = new StochasticCase("test", 0.7);
  const r1 = await c.run(42);
  const r2 = await c.run(42);
  assertEquals(r1.success, r2.success, "same trial_id must give same result");
});

Deno.test("stochastic case rate", async () => {
  const c = new StochasticCase("test", 0.6);
  let successes = 0;
  for (let i = 0; i < 200; i++) {
    if ((await c.run(i)).success) successes += 1;
  }
  assert(
    successes > 90 && successes < 170,
    `expected ~120 successes, got ${successes}`,
  );
});
