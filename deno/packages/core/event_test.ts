import { assertEquals, assertNotEquals } from "@std/assert";
import { EventEnvelope, newTraceId } from "./event.ts";

Deno.test("EventEnvelope roundtrip", () => {
  const trace = newTraceId();
  const env = new EventEnvelope(trace, 1, "hello");
  assertEquals(env.trace_id, trace);
  assertEquals(env.sequence, 1);
  assertEquals(env.payload, "hello");
  assertEquals(env.event_type, "envelope");
});

Deno.test("EventEnvelope map preserves correlation", () => {
  const trace = newTraceId();
  const env = new EventEnvelope(trace, 42, 10);
  const mapped = env.map((v) => v.toString());
  assertEquals(mapped.trace_id, trace);
  assertEquals(mapped.sequence, 42);
  assertEquals(mapped.payload, "10");
  // Correlation fields must be preserved, including event_id and timestamp.
  assertEquals(mapped.event_id, env.event_id);
  assertEquals(mapped.occurred_at, env.occurred_at);
});

Deno.test("newTraceId generates unique UUIDs", () => {
  const a = newTraceId();
  const b = newTraceId();
  assertNotEquals(a, b);
});
