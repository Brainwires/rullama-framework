/**
 * Agent Runtime - Generic execution loop for autonomous agents.
 *
 * Provides the {@link AgentRuntime} interface and {@link runAgentLoop} function
 * that implement the core agentic execution pattern:
 *
 * ```
 * Register -> Loop {
 *     Check iteration limit
 *     Call provider
 *     Check completion (finish_reason)
 *     Extract tool uses
 *     Execute tools (with optional file locking)
 *     Add results to conversation
 * } -> Complete & Unregister
 * ```
 *
 * @module
 */

import {
  type ChatResponse,
  type Message,
  ToolResult,
  type ToolUse,
} from "@rullama/core";

import type { CommunicationHub } from "@rullama/agent";
import type { FileLockManager, LockType } from "@rullama/agent";
import type { AgentLifecycleHooks, IterationContext } from "./hooks.ts";

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/** Result of an agent execution loop. */
export interface AgentExecutionResult {
  /** The agent's unique ID. */
  agentId: string;
  /** Whether the agent completed successfully. */
  success: boolean;
  /** Output message (completion summary or error description). */
  output: string;
  /** Number of iterations consumed. */
  iterations: number;
  /** Names of tools that were invoked. */
  toolsUsed: string[];
}

// ---------------------------------------------------------------------------
// Loop detector
// ---------------------------------------------------------------------------

class LoopDetector {
  private recent: string[] = [];

  constructor(
    private windowSize: number,
    private enabled: boolean,
  ) {}

  /** Record a tool call. Returns the stuck tool name if a loop is detected. */
  record(toolName: string): string | undefined {
    if (!this.enabled) return undefined;
    if (this.recent.length === this.windowSize) this.recent.shift();
    this.recent.push(toolName);
    if (
      this.recent.length === this.windowSize &&
      this.recent.every((n) => n === toolName)
    ) {
      return toolName;
    }
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// Agent runtime interface
// ---------------------------------------------------------------------------

/**
 * Trait that defines the core operations of an agentic execution loop.
 *
 * Implementors provide the provider interaction, tool execution, and
 * completion logic.
 */
export interface AgentRuntime {
  /** Get the agent's unique identifier. */
  agentId(): string;

  /** Maximum number of iterations before the loop terminates. */
  maxIterations(): number;

  /** Call the AI provider with the current conversation state. */
  callProvider(): Promise<ChatResponse>;

  /** Extract tool use requests from a provider response. */
  extractToolUses(response: ChatResponse): ToolUse[];

  /** Check if a response indicates the agent wants to complete. */
  isCompletion(response: ChatResponse): boolean;

  /** Execute a single tool and return the result. */
  executeTool(toolUse: ToolUse): Promise<ToolResult>;

  /**
   * Determine the file lock requirement for a tool invocation.
   * Returns `[path, lockType]` if a lock is needed, or `undefined`.
   */
  getLockRequirement(toolUse: ToolUse): [string, LockType] | undefined;

  /** Called when the provider returns a response containing tool uses. */
  onProviderResponse(response: ChatResponse): Promise<void> | void;

  /** Called when a tool produces a result. */
  onToolResult(toolUse: ToolUse, result: ToolResult): Promise<void> | void;

  /**
   * Called when the agent attempts to complete.
   * Return the output string if completion is accepted, or `undefined`
   * if validation failed and the loop should continue.
   */
  onCompletion(response: ChatResponse): Promise<string | undefined>;

  /** Called when the iteration limit is reached without completion. */
  onIterationLimit(iterations: number): Promise<string> | string;

  /** Optional lifecycle hooks for granular loop control. */
  lifecycleHooks?(): AgentLifecycleHooks | undefined;

  /** Context budget in tokens for pressure callbacks. */
  contextBudgetTokens?(): number | undefined;

  /** Access to the conversation history for hook-based mutation. */
  conversation?(): Message[] | undefined;
}

// ---------------------------------------------------------------------------
// Run agent loop
// ---------------------------------------------------------------------------

/**
 * Run the standard agent execution loop with communication hub and file
 * lock coordination.
 *
 * The loop terminates when:
 * - The agent signals completion and validation passes
 * - The iteration limit is reached
 * - An unrecoverable error occurs
 */
export async function runAgentLoop(
  agent: AgentRuntime,
  hub: CommunicationHub,
  lockManager: FileLockManager,
  signal?: AbortSignal,
): Promise<AgentExecutionResult> {
  const agentId = agent.agentId();
  let iterations = 0;
  const toolsUsed: string[] = [];
  const loopDetector = new LoopDetector(5, true);
  const startTime = Date.now();

  // Register with communication hub
  if (!hub.isRegistered(agentId)) {
    hub.registerAgent(agentId);
  }

  const hooks = agent.lifecycleHooks?.();

  const makeResult = (
    success: boolean,
    output: string,
  ): AgentExecutionResult => {
    try {
      hub.unregisterAgent(agentId);
    } catch { /* ignore */ }
    lockManager.releaseAllLocks(agentId);
    return { agentId, success, output, iterations, toolsUsed };
  };

  while (true) {
    // Abort signal check
    if (signal?.aborted) {
      return makeResult(false, "Aborted by signal");
    }

    // Iteration limit
    if (iterations >= agent.maxIterations()) {
      const output = await agent.onIterationLimit(iterations);
      return makeResult(false, output);
    }

    iterations++;

    // -- Hook A: onBeforeIteration --
    const conversation = agent.conversation?.();
    if (hooks?.onBeforeIteration && conversation) {
      const { ConversationView: CV } = await import("./hooks.ts");
      const view = new CV(conversation);
      const ctx = buildCtx(
        agentId,
        iterations,
        agent,
        startTime,
        conversation.length,
      );
      const decision = await hooks.onBeforeIteration(ctx, view);
      if (decision.kind === "skip") continue;
      if (decision.kind === "abort") {
        return makeResult(false, `Aborted by hook: ${decision.reason}`);
      }
    }

    // -- Hook B: onBeforeProviderCall --
    if (hooks?.onBeforeProviderCall && conversation) {
      const { ConversationView: CV } = await import("./hooks.ts");
      const view = new CV(conversation);
      const ctx = buildCtx(
        agentId,
        iterations,
        agent,
        startTime,
        conversation.length,
      );
      await hooks.onBeforeProviderCall(ctx, view);
    }

    // Call provider
    const response = await agent.callProvider();

    // -- Hook C: onAfterProviderCall --
    if (hooks?.onAfterProviderCall) {
      const convLen = conversation?.length ?? 0;
      const ctx = buildCtx(agentId, iterations, agent, startTime, convLen);
      await hooks.onAfterProviderCall(ctx, response);
    }

    // Check for completion
    if (agent.isCompletion(response)) {
      const output = await agent.onCompletion(response);
      if (output != null) {
        return makeResult(true, output);
      }
      // Validation failed -- loop continues
      continue;
    }

    // Extract tool uses
    const toolUseRequests = agent.extractToolUses(response);

    if (toolUseRequests.length === 0) {
      // No tools and no explicit completion -- try completion anyway
      const output = await agent.onCompletion(response);
      if (output != null) {
        return makeResult(true, output);
      }
      continue;
    }

    // Add assistant message
    await agent.onProviderResponse(response);

    // Execute each tool
    for (const toolUse of toolUseRequests) {
      // -- Hook D: onBeforeToolExecution --
      if (hooks?.onBeforeToolExecution) {
        const convLen = conversation?.length ?? 0;
        const ctx = buildCtx(agentId, iterations, agent, startTime, convLen);
        const decision = await hooks.onBeforeToolExecution(ctx, toolUse);
        if (decision.kind === "override") {
          await agent.onToolResult(toolUse, decision.result);
          toolsUsed.push(toolUse.name);
          continue;
        }
        if (decision.kind === "delegate" && hooks.executeDelegation) {
          try {
            const delegationResult = await hooks.executeDelegation(
              decision.request,
            );
            const result = new ToolResult(
              toolUse.id,
              `Delegated to sub-agent ${delegationResult.agentId}: ${delegationResult.output}`,
              false,
            );
            await agent.onToolResult(toolUse, result);
          } catch (e) {
            const result = new ToolResult(
              toolUse.id,
              `Delegation failed: ${e}`,
              true,
            );
            await agent.onToolResult(toolUse, result);
          }
          toolsUsed.push(toolUse.name);
          continue;
        }
      }

      toolsUsed.push(toolUse.name);

      let toolResult: ToolResult;
      const lockReq = agent.getLockRequirement(toolUse);

      if (lockReq) {
        const [path, lockType] = lockReq;
        try {
          const guard = lockManager.acquireLock(agentId, path, lockType);
          try {
            toolResult = await agent.executeTool(toolUse);
          } catch (e) {
            toolResult = new ToolResult(
              toolUse.id,
              `Tool execution failed: ${e}`,
              true,
            );
          } finally {
            guard.release();
          }
        } catch (e) {
          toolResult = new ToolResult(
            toolUse.id,
            `Failed to acquire lock on ${path}: ${e}`,
            true,
          );
        }
      } else {
        try {
          toolResult = await agent.executeTool(toolUse);
        } catch (e) {
          toolResult = new ToolResult(
            toolUse.id,
            `Tool execution failed: ${e}`,
            true,
          );
        }
      }

      await agent.onToolResult(toolUse, toolResult);

      // -- Hook E: onAfterToolExecution --
      if (hooks?.onAfterToolExecution && conversation) {
        const { ConversationView: CV } = await import("./hooks.ts");
        const view = new CV(conversation);
        const ctx = buildCtx(
          agentId,
          iterations,
          agent,
          startTime,
          conversation.length,
        );
        await hooks.onAfterToolExecution(ctx, toolUse, toolResult, view);
      }
    }

    // -- Loop detection --
    for (const toolUse of toolUseRequests) {
      const stuck = loopDetector.record(toolUse.name);
      if (stuck) {
        const output =
          `Loop detected: '${stuck}' called 5 times consecutively. Aborting.`;
        return makeResult(false, output);
      }
    }

    // -- Hook G: onAfterIteration + context pressure --
    if (conversation) {
      if (hooks?.onContextPressure) {
        const budgetTokens = agent.contextBudgetTokens?.();
        if (budgetTokens != null) {
          const { ConversationView: CV } = await import("./hooks.ts");
          const view = new CV(conversation);
          const estTokens = view.estimatedTokens();
          if (estTokens > budgetTokens) {
            const ctx = buildCtx(
              agentId,
              iterations,
              agent,
              startTime,
              conversation.length,
            );
            await hooks.onContextPressure(ctx, view, estTokens, budgetTokens);
          }
        }
      }

      if (hooks?.onAfterIteration) {
        const { ConversationView: CV } = await import("./hooks.ts");
        const view = new CV(conversation);
        const ctx = buildCtx(
          agentId,
          iterations,
          agent,
          startTime,
          conversation.length,
        );
        await hooks.onAfterIteration(ctx, view);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildCtx(
  agentId: string,
  iteration: number,
  agent: AgentRuntime,
  startTime: number,
  conversationLen: number,
): IterationContext {
  return {
    agentId,
    iteration,
    maxIterations: agent.maxIterations(),
    totalTokensUsed: 0,
    totalCostUsd: 0,
    elapsedMs: Date.now() - startTime,
    conversationLen,
  };
}
