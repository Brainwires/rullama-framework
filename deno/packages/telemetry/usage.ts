/**
 * UsageEvent — the payload handed to a BillingHook.
 *
 * Equivalent to Rust's `brainwires_telemetry::usage::UsageEvent`.
 */

export type UsageEvent =
  | {
    kind: "tokens";
    agent_id: string;
    /** e.g. "anthropic/claude-sonnet-4-6". */
    model: string;
    total_tokens: number;
    cost_usd: number;
    /** ISO 8601. */
    timestamp: string;
  }
  | {
    kind: "tool_call";
    agent_id: string;
    tool_name: string;
    cost_usd: number;
    timestamp: string;
  }
  | {
    kind: "sandbox_seconds";
    agent_id: string;
    provider: string;
    seconds: number;
    cost_usd: number;
    timestamp: string;
  }
  | {
    kind: "api_call";
    agent_id: string;
    service: string;
    cost_usd: number;
    timestamp: string;
  }
  | {
    kind: "custom";
    agent_id: string;
    name: string;
    cost_usd: number;
    metadata: unknown;
    timestamp: string;
  };

function nowIso(): string {
  return new Date().toISOString();
}

/** Create a tokens usage event stamped with the current time. */
export function tokensEvent(
  agent_id: string,
  model: string,
  total_tokens: number,
  cost_usd: number,
): UsageEvent {
  return { kind: "tokens", agent_id, model, total_tokens, cost_usd, timestamp: nowIso() };
}

/** Create a tool_call event with zero cost (most built-ins). */
export function toolCallEvent(agent_id: string, tool_name: string): UsageEvent {
  return toolCallPaidEvent(agent_id, tool_name, 0);
}

/** Create a tool_call event with an explicit USD charge. */
export function toolCallPaidEvent(
  agent_id: string,
  tool_name: string,
  cost_usd: number,
): UsageEvent {
  return { kind: "tool_call", agent_id, tool_name, cost_usd, timestamp: nowIso() };
}

export function sandboxSecondsEvent(
  agent_id: string,
  provider: string,
  seconds: number,
  cost_usd: number,
): UsageEvent {
  return { kind: "sandbox_seconds", agent_id, provider, seconds, cost_usd, timestamp: nowIso() };
}

export function apiCallEvent(
  agent_id: string,
  service: string,
  cost_usd: number,
): UsageEvent {
  return { kind: "api_call", agent_id, service, cost_usd, timestamp: nowIso() };
}

export function agentIdOf(e: UsageEvent): string {
  return e.agent_id;
}

export function costUsdOf(e: UsageEvent): number {
  return e.cost_usd;
}

export function timestampOf(e: UsageEvent): string {
  return e.timestamp;
}

export function kindOf(e: UsageEvent): UsageEvent["kind"] {
  return e.kind;
}
