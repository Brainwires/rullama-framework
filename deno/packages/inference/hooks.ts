/**
 * Agent Lifecycle Hooks - Granular control over the agent execution loop
 *
 * Provides {@link AgentLifecycleHooks} for intercepting every phase of the agent
 * loop: iteration boundaries, provider calls, tool execution, completion, and
 * context management.
 *
 * Unlike the observational lifecycle system in `@rullama/core`, these hooks
 * can **control** the loop -- skip iterations, override tool results, delegate
 * work to sub-agents, or compress conversation history.
 *
 * @module
 */

import type {
  ChatResponse,
  Message,
  ToolResult,
  ToolUse,
} from "@rullama/core";
import { estimateTokensFromSize } from "@rullama/core";

// Re-export TaskAgentConfig as an opaque type reference for DelegationRequest.
// The actual definition lives in task_agent.ts; import it there to avoid cycles.
import type { TaskAgentConfig } from "./task_agent.ts";
import type { TaskAgentResult } from "./task_agent.ts";

// ---------------------------------------------------------------------------
// Iteration context
// ---------------------------------------------------------------------------

/** Read-only snapshot of the current iteration state, passed to every hook. */
export interface IterationContext {
  /** The agent's unique identifier. */
  readonly agentId: string;
  /** Current iteration number (1-based). */
  readonly iteration: number;
  /** Maximum iterations allowed. */
  readonly maxIterations: number;
  /** Cumulative tokens consumed so far. */
  readonly totalTokensUsed: number;
  /** Cumulative estimated cost in USD. */
  readonly totalCostUsd: number;
  /** Wall-clock milliseconds since execution started. */
  readonly elapsedMs: number;
  /** Number of messages in the conversation history. */
  readonly conversationLen: number;
}

// ---------------------------------------------------------------------------
// Decision enums
// ---------------------------------------------------------------------------

/** Decision returned by {@link AgentLifecycleHooks.onBeforeIteration}. */
export type IterationDecision =
  | { kind: "continue" }
  | { kind: "skip" }
  | { kind: "abort"; reason: string };

/** Decision returned by {@link AgentLifecycleHooks.onBeforeToolExecution}. */
export type ToolDecision =
  | { kind: "execute" }
  | { kind: "override"; result: ToolResult }
  | { kind: "delegate"; request: DelegationRequest };

// ---------------------------------------------------------------------------
// Delegation types
// ---------------------------------------------------------------------------

/** A request to delegate work to a sub-agent. */
export interface DelegationRequest {
  /** Description of the sub-task for the spawned agent. */
  taskDescription: string;
  /** Optional config override for the sub-agent. */
  config?: TaskAgentConfig;
  /** Messages to seed the sub-agent's conversation with. */
  seedMessages: Message[];
  /** If `true`, block until the sub-agent completes and return its output. */
  blocking: boolean;
}

/** Create a default DelegationRequest. */
export function defaultDelegationRequest(): DelegationRequest {
  return {
    taskDescription: "",
    seedMessages: [],
    blocking: true,
  };
}

/** Result of a completed delegation. */
export interface DelegationResult {
  /** The sub-agent's unique ID. */
  agentId: string;
  /** Whether the sub-agent completed successfully. */
  success: boolean;
  /** Output summary from the sub-agent. */
  output: string;
  /** Iterations the sub-agent consumed. */
  iterationsUsed: number;
  /** Tokens the sub-agent consumed. */
  tokensUsed: number;
}

// ---------------------------------------------------------------------------
// Conversation view
// ---------------------------------------------------------------------------

/**
 * Controlled read/write handle to the conversation history.
 *
 * Passed to hooks that need to inspect or mutate messages (e.g., for
 * summarization or context injection).
 */
export class ConversationView {
  constructor(private messages: Message[]) {}

  /** Number of messages in the conversation. */
  get length(): number {
    return this.messages.length;
  }

  /** Whether the conversation is empty. */
  isEmpty(): boolean {
    return this.messages.length === 0;
  }

  /** Read-only access to all messages. */
  getMessages(): readonly Message[] {
    return this.messages;
  }

  /** The last `n` messages (or all if fewer exist). */
  lastN(n: number): readonly Message[] {
    const start = Math.max(0, this.messages.length - n);
    return this.messages.slice(start);
  }

  /** Append a message to the end of the conversation. */
  push(msg: Message): void {
    this.messages.push(msg);
  }

  /** Insert a message at a specific position. */
  insert(index: number, msg: Message): void {
    this.messages.splice(index, 0, msg);
  }

  /** Remove and return messages in the given range [start, end). */
  drain(start: number, end: number): Message[] {
    return this.messages.splice(start, end - start);
  }

  /**
   * Replace a range of messages with a single summary message.
   *
   * Useful for compressing old context to save tokens.
   */
  summarizeRange(start: number, end: number, summary: Message): void {
    this.messages.splice(start, end - start, summary);
  }

  /**
   * Estimate total tokens across all messages using byte-length heuristic.
   */
  estimatedTokens(): number {
    let total = 0;
    for (const m of this.messages) {
      const text = typeof m.content === "string"
        ? m.content
        : JSON.stringify(m.content);
      total += estimateTokensFromSize(new TextEncoder().encode(text).length);
    }
    return total;
  }

  /** Get the text of the most recent assistant message, if any. */
  lastAssistantText(): string | undefined {
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const m = this.messages[i];
      if (m.role === "assistant" && typeof m.content === "string") {
        return m.content;
      }
    }
    return undefined;
  }

  /**
   * Return the underlying mutable array.
   * Use with caution -- primarily for the runtime loop.
   */
  _rawMessages(): Message[] {
    return this.messages;
  }
}

// ---------------------------------------------------------------------------
// Main trait
// ---------------------------------------------------------------------------

/**
 * Granular lifecycle hooks for controlling an agent's execution loop.
 *
 * All methods have default no-op implementations -- consumers only override
 * the hooks they need.
 */
export interface AgentLifecycleHooks {
  // -- Iteration-level hooks --

  /** Called at the top of each iteration, before the provider call. */
  onBeforeIteration?(
    ctx: IterationContext,
    conversation: ConversationView,
  ): Promise<IterationDecision> | IterationDecision;

  /** Called after all tools have been executed, before the next iteration. */
  onAfterIteration?(
    ctx: IterationContext,
    conversation: ConversationView,
  ): Promise<void> | void;

  // -- Provider call hooks --

  /** Called immediately before the provider is called. */
  onBeforeProviderCall?(
    ctx: IterationContext,
    conversation: ConversationView,
  ): Promise<void> | void;

  /** Called immediately after the provider returns a response. */
  onAfterProviderCall?(
    ctx: IterationContext,
    response: ChatResponse,
  ): Promise<void> | void;

  // -- Tool execution hooks --

  /** Called before each tool is executed. */
  onBeforeToolExecution?(
    ctx: IterationContext,
    toolUse: ToolUse,
  ): Promise<ToolDecision> | ToolDecision;

  /** Called after each tool execution completes. */
  onAfterToolExecution?(
    ctx: IterationContext,
    toolUse: ToolUse,
    result: ToolResult,
    conversation: ConversationView,
  ): Promise<void> | void;

  // -- Completion hooks --

  /** Called when the agent signals completion, before validation. */
  onBeforeCompletion?(
    ctx: IterationContext,
    completionText: string,
  ): Promise<boolean> | boolean;

  /** Called after a successful completion (validation passed). */
  onAfterCompletion?(
    ctx: IterationContext,
    result: TaskAgentResult,
  ): Promise<void> | void;

  // -- Context management hooks --

  /** Called when estimated conversation token count exceeds the budget. */
  onContextPressure?(
    ctx: IterationContext,
    conversation: ConversationView,
    estimatedTokens: number,
    budgetTokens: number,
  ): Promise<void> | void;

  // -- Delegation hooks --

  /**
   * Called when a {@link ToolDecision} of kind "delegate" needs to be fulfilled.
   *
   * Consumers must override this if they use delegation.
   */
  executeDelegation?(
    request: DelegationRequest,
  ): Promise<DelegationResult>;
}
