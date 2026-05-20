/**
 * Outcome metrics — per-agent aggregates with Prometheus text exposition.
 *
 * Implements {@link AnalyticsSink} so it plugs directly into the existing
 * {@link AnalyticsCollector} pipeline and updates counters in real time as
 * events flow through.
 *
 * Equivalent to Rust's `brainwires_telemetry::metrics` module.
 */

import type { AnalyticsEvent } from "./events.ts";
import type { AnalyticsSink } from "./sink.ts";

/** Aggregated outcome metrics for a single agent. */
export interface OutcomeMetrics {
  agent_id: string;
  total_runs: number;
  success_count: number;
  failure_count: number;
  total_iterations: number;
  total_tool_calls: number;
  tool_error_count: number;
  provider_call_count: number;
  total_tokens_prompt: number;
  total_tokens_completion: number;
  total_cost_usd: number;
  total_provider_duration_ms: number;
  total_run_duration_ms: number;
  total_cache_read_tokens: number;
  total_cache_creation_tokens: number;
}

function freshMetrics(agent_id: string): OutcomeMetrics {
  return {
    agent_id,
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
  };
}

// Derived ratios / averages ---------------------------------------------------

export function successRate(m: OutcomeMetrics): number {
  return m.total_runs === 0 ? 0 : m.success_count / m.total_runs;
}

export function avgCostPerRunUsd(m: OutcomeMetrics): number {
  return m.total_runs === 0 ? 0 : m.total_cost_usd / m.total_runs;
}

export function avgRunDurationMs(m: OutcomeMetrics): number {
  return m.total_runs === 0 ? 0 : m.total_run_duration_ms / m.total_runs;
}

export function avgProviderLatencyMs(m: OutcomeMetrics): number {
  return m.provider_call_count === 0 ? 0 : m.total_provider_duration_ms / m.provider_call_count;
}

export function toolErrorRate(m: OutcomeMetrics): number {
  return m.total_tool_calls === 0 ? 0 : m.tool_error_count / m.total_tool_calls;
}

export function cacheHitRate(m: OutcomeMetrics): number {
  const denom = m.total_tokens_prompt + m.total_cache_read_tokens;
  return denom === 0 ? 0 : m.total_cache_read_tokens / denom;
}

// Registry --------------------------------------------------------------------

/** Thread-safe registry (JS isolates are single-threaded — just a Map). */
export class MetricsRegistry implements AnalyticsSink {
  private readonly entries = new Map<string, OutcomeMetrics>();

  /** Snapshot for a specific agent. */
  get(agent_id: string): OutcomeMetrics | null {
    const m = this.entries.get(agent_id);
    return m ? { ...m } : null;
  }

  /** Snapshots for all tracked agents. */
  all(): OutcomeMetrics[] {
    return Array.from(this.entries.values(), (m) => ({ ...m }));
  }

  /** Reset metrics for a single agent. */
  reset(agent_id: string): void {
    this.entries.delete(agent_id);
  }

  /** Reset all tracked metrics. */
  resetAll(): void {
    this.entries.clear();
  }

  private entry(agent_id: string): OutcomeMetrics {
    let m = this.entries.get(agent_id);
    if (!m) {
      m = freshMetrics(agent_id);
      this.entries.set(agent_id, m);
    }
    return m;
  }

  /** AnalyticsSink: update the counters from an event. */
  record(event: AnalyticsEvent): Promise<void> {
    switch (event.event_type) {
      case "agent_run": {
        const m = this.entry(event.agent_id);
        m.total_runs += 1;
        if (event.success) m.success_count += 1;
        else m.failure_count += 1;
        m.total_iterations += event.total_iterations;
        m.total_tool_calls += event.total_tool_calls;
        m.tool_error_count += event.tool_error_count;
        m.total_tokens_prompt += event.total_prompt_tokens;
        m.total_tokens_completion += event.total_completion_tokens;
        m.total_cost_usd += event.total_cost_usd;
        m.total_run_duration_ms += event.duration_ms;
        break;
      }
      case "provider_call": {
        // ProviderCall isn't keyed by agent — credit a global bucket so a
        // Prometheus scrape still picks up totals.
        const m = this.entry("__global__");
        m.provider_call_count += 1;
        m.total_tokens_prompt += event.prompt_tokens;
        m.total_tokens_completion += event.completion_tokens;
        m.total_cost_usd += event.cost_usd;
        m.total_provider_duration_ms += event.duration_ms;
        m.total_cache_read_tokens += event.cache_read_input_tokens ?? 0;
        m.total_cache_creation_tokens += event.cache_creation_input_tokens ?? 0;
        break;
      }
      case "tool_call": {
        if (event.agent_id) {
          const m = this.entry(event.agent_id);
          m.total_tool_calls += 1;
          if (event.is_error) m.tool_error_count += 1;
        }
        break;
      }
      default:
        // Other event types don't contribute to outcome metrics.
        break;
    }
    return Promise.resolve();
  }

  /** Render all tracked metrics as a Prometheus exposition text body. */
  prometheusText(): string {
    if (this.entries.size === 0) return "";
    const lines: string[] = [];

    const counter = (name: string, help: string) => {
      lines.push(`# HELP ${name} ${help}`);
      lines.push(`# TYPE ${name} counter`);
    };
    const gauge = (name: string, help: string) => {
      lines.push(`# HELP ${name} ${help}`);
      lines.push(`# TYPE ${name} gauge`);
    };
    const row = (name: string, agent_id: string, value: number) => {
      lines.push(`${name}{agent_id="${escape(agent_id)}"} ${fmt(value)}`);
    };

    counter("brainwires_agent_runs_total", "Total agent runs attempted");
    for (const m of this.entries.values()) row("brainwires_agent_runs_total", m.agent_id, m.total_runs);

    counter("brainwires_agent_runs_success_total", "Agent runs that succeeded");
    for (const m of this.entries.values()) row("brainwires_agent_runs_success_total", m.agent_id, m.success_count);

    counter("brainwires_agent_runs_failure_total", "Agent runs that failed");
    for (const m of this.entries.values()) row("brainwires_agent_runs_failure_total", m.agent_id, m.failure_count);

    gauge("brainwires_agent_success_rate", "Agent run success rate (0-1)");
    for (const m of this.entries.values()) row("brainwires_agent_success_rate", m.agent_id, successRate(m));

    counter("brainwires_agent_tool_calls_total", "Total tool calls made by agent");
    for (const m of this.entries.values()) row("brainwires_agent_tool_calls_total", m.agent_id, m.total_tool_calls);

    counter("brainwires_agent_tool_errors_total", "Tool calls that produced an error");
    for (const m of this.entries.values()) row("brainwires_agent_tool_errors_total", m.agent_id, m.tool_error_count);

    counter("brainwires_agent_provider_calls_total", "Total LLM provider calls");
    for (const m of this.entries.values()) row("brainwires_agent_provider_calls_total", m.agent_id, m.provider_call_count);

    counter("brainwires_agent_tokens_prompt_total", "Total prompt tokens consumed");
    for (const m of this.entries.values()) row("brainwires_agent_tokens_prompt_total", m.agent_id, m.total_tokens_prompt);

    counter("brainwires_agent_tokens_completion_total", "Total completion tokens generated");
    for (const m of this.entries.values()) row("brainwires_agent_tokens_completion_total", m.agent_id, m.total_tokens_completion);

    counter("brainwires_agent_cost_usd_total", "Cumulative LLM cost in USD");
    for (const m of this.entries.values()) row("brainwires_agent_cost_usd_total", m.agent_id, m.total_cost_usd);

    gauge("brainwires_agent_avg_run_duration_ms", "Average agent run duration in ms");
    for (const m of this.entries.values()) row("brainwires_agent_avg_run_duration_ms", m.agent_id, avgRunDurationMs(m));

    gauge("brainwires_agent_avg_provider_latency_ms", "Average LLM provider call latency in ms");
    for (const m of this.entries.values()) row("brainwires_agent_avg_provider_latency_ms", m.agent_id, avgProviderLatencyMs(m));

    counter("brainwires_agent_cache_read_tokens_total", "Prompt tokens served from the provider's cache");
    for (const m of this.entries.values()) row("brainwires_agent_cache_read_tokens_total", m.agent_id, m.total_cache_read_tokens);

    counter("brainwires_agent_cache_creation_tokens_total", "Prompt tokens charged to populate the provider's cache");
    for (const m of this.entries.values()) row("brainwires_agent_cache_creation_tokens_total", m.agent_id, m.total_cache_creation_tokens);

    gauge("brainwires_agent_cache_hit_rate", "Prompt cache hit rate (0-1)");
    for (const m of this.entries.values()) row("brainwires_agent_cache_hit_rate", m.agent_id, cacheHitRate(m));

    return lines.join("\n") + "\n";
  }
}

function escape(s: string): string {
  return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
}

function fmt(v: number): string {
  return Number.isInteger(v) ? v.toString() : v.toString();
}
