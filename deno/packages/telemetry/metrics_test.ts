import { assert, assertEquals } from "@std/assert";
import {
  avgRunDurationMs,
  cacheHitRate,
  MetricsRegistry,
  successRate,
  toolErrorRate,
} from "./metrics.ts";
import type { AnalyticsEvent } from "./events.ts";

function agentRun(
  agent_id: string,
  overrides: Partial<Extract<AnalyticsEvent, { event_type: "agent_run" }>> = {},
): AnalyticsEvent {
  return {
    event_type: "agent_run",
    session_id: null,
    agent_id,
    task_id: "t",
    prompt_hash: "h",
    success: true,
    total_iterations: 1,
    total_tool_calls: 0,
    tool_error_count: 0,
    tools_used: [],
    total_prompt_tokens: 0,
    total_completion_tokens: 0,
    total_cost_usd: 0,
    duration_ms: 100,
    failure_category: null,
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

Deno.test("registry records runs and computes rates", async () => {
  const reg = new MetricsRegistry();
  await reg.record(agentRun("a", { success: true, duration_ms: 100, total_cost_usd: 0.02 }));
  await reg.record(agentRun("a", { success: false, duration_ms: 200, total_cost_usd: 0.03 }));
  await reg.record(agentRun("a", { success: true, duration_ms: 300, total_cost_usd: 0.01 }));

  const m = reg.get("a");
  assert(m);
  assertEquals(m.total_runs, 3);
  assertEquals(m.success_count, 2);
  assertEquals(m.failure_count, 1);
  assertEquals(successRate(m), 2 / 3);
  assertEquals(avgRunDurationMs(m), 200);
  // Float sum — round to cents before comparing.
  assertEquals(Math.round(m.total_cost_usd * 100) / 100, 0.06);
});

Deno.test("tool_error_rate is zero without calls", () => {
  const reg = new MetricsRegistry();
  assertEquals(toolErrorRate({
    agent_id: "x",
    total_runs: 0,
    success_count: 0,
    failure_count: 0,
    total_iterations: 0,
    total_tool_calls: 0,
    tool_error_count: 0,
    provider_call_count: 0,
    total_tokens_prompt: 0,
    total_tokens_completion: 0,
    total_cost_usd: 0,
    total_provider_duration_ms: 0,
    total_run_duration_ms: 0,
    total_cache_read_tokens: 0,
    total_cache_creation_tokens: 0,
  }), 0);
  assert(reg.all().length === 0);
});

Deno.test("provider_call credits the global bucket", async () => {
  const reg = new MetricsRegistry();
  await reg.record({
    event_type: "provider_call",
    session_id: null,
    provider: "anthropic",
    model: "claude-sonnet-4-6",
    prompt_tokens: 1000,
    completion_tokens: 500,
    duration_ms: 400,
    cost_usd: 0.02,
    success: true,
    timestamp: new Date().toISOString(),
    cache_read_input_tokens: 200,
    cache_creation_input_tokens: 50,
  });
  const g = reg.get("__global__");
  assert(g);
  assertEquals(g.provider_call_count, 1);
  assertEquals(g.total_tokens_prompt, 1000);
  assertEquals(g.total_cache_read_tokens, 200);
  assertEquals(g.total_cache_creation_tokens, 50);
  // cache hit rate against (prompt + cache_read).
  assertEquals(cacheHitRate(g), 200 / 1200);
});

Deno.test("prometheusText has expected TYPE lines", async () => {
  const reg = new MetricsRegistry();
  await reg.record(agentRun("a", { success: true }));
  const body = reg.prometheusText();
  assert(body.includes("# TYPE brainwires_agent_runs_total counter"));
  assert(body.includes('brainwires_agent_runs_total{agent_id="a"}'));
});

Deno.test("empty registry renders empty prometheus body", () => {
  const reg = new MetricsRegistry();
  assertEquals(reg.prometheusText(), "");
});
