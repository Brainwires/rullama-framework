// deno-lint-ignore-file no-explicit-any

/** Events emitted during framework operation.
 * Equivalent to Rust's `LifecycleEvent` in rullama-core. */
export type LifecycleEvent =
  | { type: "agent_started"; agent_id: string; task_description: string }
  | {
    type: "agent_completed";
    agent_id: string;
    iterations: number;
    summary: string;
  }
  | {
    type: "agent_failed";
    agent_id: string;
    error: string;
    iterations: number;
  }
  | {
    type: "tool_before_execute";
    agent_id?: string;
    tool_name: string;
    args: any;
  }
  | {
    type: "tool_after_execute";
    agent_id?: string;
    tool_name: string;
    success: boolean;
    duration_ms: number;
  }
  | {
    type: "provider_request";
    agent_id?: string;
    provider: string;
    model: string;
  }
  | {
    type: "provider_response";
    agent_id?: string;
    provider: string;
    model: string;
    input_tokens: number;
    output_tokens: number;
    duration_ms: number;
  }
  | { type: "validation_started"; agent_id: string; checks: string[] }
  | {
    type: "validation_completed";
    agent_id: string;
    passed: boolean;
    issues: string[];
  };

/** Get the event type name. */
export function eventType(event: LifecycleEvent): string {
  return event.type;
}

/** Get the agent ID from an event, if any. */
export function eventAgentId(event: LifecycleEvent): string | undefined {
  return "agent_id" in event ? event.agent_id : undefined;
}

/** Get the tool name from an event, if any. */
export function eventToolName(event: LifecycleEvent): string | undefined {
  if (
    event.type === "tool_before_execute" || event.type === "tool_after_execute"
  ) {
    return event.tool_name;
  }
  return undefined;
}

/** Result of a hook invocation.
 * Equivalent to Rust's `HookResult` in rullama-core. */
export type HookResult =
  | { type: "continue" }
  | { type: "cancel"; reason: string }
  | { type: "modified"; data: any };

/** Filter to control which events a hook receives.
 * Equivalent to Rust's `EventFilter` in rullama-core. */
export interface EventFilter {
  agent_ids: Set<string>;
  event_types: Set<string>;
  tool_names: Set<string>;
}

/** Create a default EventFilter that matches everything. */
export function defaultEventFilter(): EventFilter {
  return {
    agent_ids: new Set(),
    event_types: new Set(),
    tool_names: new Set(),
  };
}

/** Check if a filter matches an event. */
export function filterMatches(
  filter: EventFilter,
  event: LifecycleEvent,
): boolean {
  if (filter.event_types.size > 0 && !filter.event_types.has(event.type)) {
    return false;
  }
  if (filter.agent_ids.size > 0) {
    const agentId = eventAgentId(event);
    if (!agentId || !filter.agent_ids.has(agentId)) return false;
  }
  if (filter.tool_names.size > 0) {
    const toolName = eventToolName(event);
    if (toolName && !filter.tool_names.has(toolName)) return false;
  }
  return true;
}

/** Interface for lifecycle hooks.
 * Equivalent to Rust's `LifecycleHook` trait in rullama-core. */
export interface LifecycleHook {
  readonly name: string;
  priority?(): number;
  filter?(): EventFilter;
  onEvent(event: LifecycleEvent): Promise<HookResult>;
}

/** Registry that manages and dispatches lifecycle hooks.
 * Equivalent to Rust's `HookRegistry` in rullama-core. */
export class HookRegistry {
  private hooks: LifecycleHook[] = [];

  /** Register a new hook. Hooks are sorted by priority after insertion. */
  register(hook: LifecycleHook): void {
    this.hooks.push(hook);
    this.hooks.sort((a, b) => (a.priority?.() ?? 0) - (b.priority?.() ?? 0));
  }

  /** Dispatch an event to all matching hooks. */
  async dispatch(event: LifecycleEvent): Promise<HookResult> {
    for (const hook of this.hooks) {
      const f = hook.filter?.();
      if (f && !filterMatches(f, event)) continue;
      const result = await hook.onEvent(event);
      if (result.type !== "continue") return result;
    }
    return { type: "continue" };
  }

  /** Number of registered hooks. */
  get length(): number {
    return this.hooks.length;
  }

  /** Whether the registry has no hooks. */
  isEmpty(): boolean {
    return this.hooks.length === 0;
  }
}
