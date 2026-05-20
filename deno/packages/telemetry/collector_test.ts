import { assert, assertEquals } from "@std/assert";
import { AnalyticsCollector } from "./collector.ts";
import { MemoryAnalyticsSink } from "./sink.ts";
import type { AnalyticsEvent } from "./events.ts";

function customEvent(): AnalyticsEvent {
  return {
    event_type: "custom",
    session_id: null,
    name: "test",
    payload: null,
    timestamp: new Date().toISOString(),
  };
}

Deno.test("fan-out to sink delivers every event after flush", async () => {
  const sink = new MemoryAnalyticsSink(128);
  const collector = new AnalyticsCollector([sink]);

  for (let i = 0; i < 10; i++) collector.record(customEvent());
  await collector.flush();
  assertEquals(sink.len(), 10);
});

Deno.test("onEvent callback fires synchronously per record", () => {
  const seen: string[] = [];
  const collector = new AnalyticsCollector();
  collector.onEvent((e) => {
    if (e.event_type === "custom") seen.push(e.name);
  });
  collector.record(customEvent());
  collector.record(customEvent());
  assertEquals(seen.length, 2);
});

Deno.test("record never throws when a sink fails", async () => {
  const flaky = {
    record: () => Promise.reject(new Error("boom")),
    flush: () => Promise.resolve(),
  };
  const collector = new AnalyticsCollector([flaky]);
  collector.record(customEvent());
  await collector.flush();
  // If we got here without throwing, fail-open works.
  assert(true);
});
