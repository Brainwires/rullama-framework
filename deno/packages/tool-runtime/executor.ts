/**
 * The `ToolExecutor` interface — the seam through which agents run tools
 * without depending on a concrete implementation — plus the `ToolPreHook`
 * interface and its `allow()` / `reject()` decisions for gating a call before it
 * executes. `EnforcingExecutor` in `./enforcement.ts` wraps any executor with
 * permission checks; `@rullama/tool-builtins` provides the default executor.
 * Equivalent to Rust's `rullama_tool_runtime::executor`.
 *
 * @module
 */

import type { Tool, ToolContext, ToolUse } from "@rullama/core";
import type { ToolResult } from "@rullama/core";

/** Decision returned by a ToolPreHook before a tool call. */
export type PreHookDecision =
  | { type: "Allow" }
  | { type: "Reject"; reason: string };

/** Create an Allow decision. */
export function allow(): PreHookDecision {
  return { type: "Allow" };
}

/** Create a Reject decision. */
export function reject(reason: string): PreHookDecision {
  return { type: "Reject", reason };
}

/**
 * Pluggable pre-execution hook for semantic tool validation.
 *
 * Implement this to intercept tool calls before execution and validate
 * call intent against current agent state (not just JSON schema).
 */
export interface ToolPreHook {
  /** Decide whether `toolUse` may run; return `reject(reason)` to block it. */
  beforeExecute(
    toolUse: ToolUse,
    context: ToolContext,
  ): Promise<PreHookDecision>;
}

/**
 * Interface for executing tools in an agent context.
 *
 * Implement this on your tool executor to integrate with framework agents
 * like TaskAgent. The interface is designed for composition via dependency
 * injection.
 */
export interface ToolExecutor {
  /** Execute a tool and return its result. */
  execute(toolUse: ToolUse, context: ToolContext): Promise<ToolResult>;

  /** Return the list of tools available for the AI to invoke. */
  availableTools(): Tool[];
}
