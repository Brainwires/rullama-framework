import { assert, assertEquals, assertRejects } from "@std/assert";
import { ChatOptions } from "@brainwires/core";
import { CircuitBreakerProvider } from "./circuit.ts";
import { ResilienceError } from "./error.ts";
import { EchoProvider, ToggleProvider } from "./test_util.ts";

Deno.test("state starts closed", () => {
  const cb = new CircuitBreakerProvider(EchoProvider.ok("p1"));
  assertEquals(cb.stateFor("any"), "closed");
});

Deno.test("opens after threshold", async () => {
  const cb = new CircuitBreakerProvider(
    EchoProvider.alwaysErr("p1", "500 internal server error"),
    { failure_threshold: 3, cooldown_ms: 50 },
  );
  const opts = new ChatOptions();
  for (let i = 0; i < 3; i++) {
    try {
      await cb.chat([], undefined, opts);
    } catch {
      // expected
    }
  }
  assertEquals(cb.stateFor("default"), "open");

  // Fast-fail while open.
  const e = await assertRejects(() => cb.chat([], undefined, opts), ResilienceError);
  assertEquals(e.kind, "circuit_open");
});

Deno.test("half open then closes on success", async () => {
  const prov = new ToggleProvider("p1");
  const cb = new CircuitBreakerProvider(prov, {
    failure_threshold: 2,
    cooldown_ms: 20,
  });
  const opts = new ChatOptions();

  prov.setFail(true);
  for (let i = 0; i < 2; i++) {
    try {
      await cb.chat([], undefined, opts);
    } catch {
      // expected
    }
  }

  await new Promise((r) => setTimeout(r, 30));
  prov.setFail(false);
  await cb.chat([], undefined, opts); // half-open probe succeeds
  assertEquals(cb.stateFor("default"), "closed");
});

Deno.test("half open reopens on failure", async () => {
  const prov = new ToggleProvider("p1");
  const cb = new CircuitBreakerProvider(prov, {
    failure_threshold: 2,
    cooldown_ms: 20,
  });
  const opts = new ChatOptions();

  prov.setFail(true);
  for (let i = 0; i < 2; i++) {
    try {
      await cb.chat([], undefined, opts);
    } catch {
      // expected
    }
  }
  assertEquals(cb.stateFor("default"), "open");

  await new Promise((r) => setTimeout(r, 30));
  // Provider still failing → half-open probe surfaces the provider error
  // and the breaker re-trips to Open.
  const e = await assertRejects(() => cb.chat([], undefined, opts));
  assert(e instanceof Error && e.message.includes("500"));
  assertEquals(cb.stateFor("default"), "open");

  // Next call fast-fails with CircuitOpen.
  const e2 = await assertRejects(() => cb.chat([], undefined, opts), ResilienceError);
  assertEquals(e2.kind, "circuit_open");
});
