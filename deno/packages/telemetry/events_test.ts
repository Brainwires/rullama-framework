import { assertEquals } from "@std/assert";
import { type AnalyticsEvent, eventSessionId, eventTimestamp, eventType } from "./events.ts";

Deno.test("event type accessor", () => {
  const now = new Date().toISOString();
  const e: AnalyticsEvent = {
    event_type: "provider_call",
    session_id: "s1",
    provider: "anthropic",
    model: "claude-sonnet-4-6",
    prompt_tokens: 100,
    completion_tokens: 200,
    duration_ms: 500,
    cost_usd: 0.01,
    success: true,
    timestamp: now,
  };
  assertEquals(eventType(e), "provider_call");
  assertEquals(eventTimestamp(e), now);
  assertEquals(eventSessionId(e), "s1");
});

Deno.test("custom event without session_id returns null", () => {
  const e: AnalyticsEvent = {
    event_type: "custom",
    session_id: null,
    name: "x",
    payload: { a: 1 },
    timestamp: new Date().toISOString(),
  };
  assertEquals(eventSessionId(e), null);
});
