/**
 * TaskAgent - Autonomous agent that executes a task in a loop using AI + tools.
 *
 * Each TaskAgent owns its conversation history and calls the AI provider
 * repeatedly, executing tool requests and running validation before it
 * signals completion.
 *
 * @module
 */

import {
  ChatOptions,
  type ChatResponse,
  Message,
  type Provider,
  type Task,
  type Tool,
  type ToolResult,
  type ToolUse,
} from "@rullama/core";

import type { AgentContext } from "./context.ts";
import type { AgentRuntime } from "./runtime.ts";
import { runAgentLoop } from "./runtime.ts";
import type { LockType } from "@rullama/agent";
import type { AgentLifecycleHooks } from "./hooks.ts";
import type { ValidationConfig } from "./validation_loop.ts";

// ---------------------------------------------------------------------------
// Loop detection config
// ---------------------------------------------------------------------------

/** Configuration for stuck-agent (loop) detection. */
export interface LoopDetectionConfig {
  /** Consecutive identical tool-name calls that trigger abort. Default: 5. */
  windowSize: number;
  /** Whether loop detection is active. Default: true. */
  enabled: boolean;
}

/** Default loop detection config. */
export function defaultLoopDetectionConfig(): LoopDetectionConfig {
  return { windowSize: 5, enabled: true };
}

// ---------------------------------------------------------------------------
// Task agent status
// ---------------------------------------------------------------------------

/** Runtime status of a task agent. */
export type TaskAgentStatus =
  | { kind: "idle" }
  | { kind: "working"; description: string }
  | { kind: "waiting_for_lock"; path: string }
  | { kind: "paused"; reason: string }
  | { kind: "replanning"; reason: string }
  | { kind: "completed"; summary: string }
  | { kind: "failed"; error: string };

/** Format a TaskAgentStatus as a display string. */
export function formatTaskAgentStatus(status: TaskAgentStatus): string {
  switch (status.kind) {
    case "idle":
      return "Idle";
    case "working":
      return `Working: ${status.description}`;
    case "waiting_for_lock":
      return `Waiting for lock: ${status.path}`;
    case "paused":
      return `Paused: ${status.reason}`;
    case "replanning":
      return `Replanning: ${status.reason}`;
    case "completed":
      return `Completed: ${status.summary}`;
    case "failed":
      return `Failed: ${status.error}`;
  }
}

// ---------------------------------------------------------------------------
// Failure category
// ---------------------------------------------------------------------------

/** Classification of why an agent run failed. */
export type FailureCategory =
  | "iteration_limit_exceeded"
  | "token_budget_exceeded"
  | "cost_budget_exceeded"
  | "wall_clock_timeout"
  | "loop_detected"
  | "max_replan_attempts_exceeded"
  | "file_scope_violation"
  | "validation_failed"
  | "tool_execution_error"
  | "unknown"
  | "plan_budget_exceeded";

// ---------------------------------------------------------------------------
// Task agent result
// ---------------------------------------------------------------------------

/** Result of a completed task agent execution. */
export interface TaskAgentResult {
  /** The agent's unique ID. */
  agentId: string;
  /** The task ID that was executed. */
  taskId: string;
  /** Whether the task completed successfully. */
  success: boolean;
  /** Completion summary or error description. */
  summary: string;
  /** Number of provider call iterations used. */
  iterations: number;
  /** Number of replan cycles during execution. */
  replanCount: number;
  /** True when any budget ceiling caused the stop. */
  budgetExhausted: boolean;
  /** Last meaningful assistant message when stopped early, if any. */
  partialOutput?: string;
  /** Cumulative tokens consumed across all provider calls. */
  totalTokensUsed: number;
  /** Estimated cost in USD. */
  totalCostUsd: number;
  /** True when wall-clock timeout caused the stop. */
  timedOut: boolean;
  /** Why the agent failed. undefined on success. */
  failureCategory?: FailureCategory;
}

// ---------------------------------------------------------------------------
// Task agent config
// ---------------------------------------------------------------------------

/** Configuration for a task agent. */
export interface TaskAgentConfig {
  /** Maximum provider call iterations. Default: 100. */
  maxIterations: number;
  /** Override the system prompt. */
  systemPrompt?: string;
  /** Temperature for AI calls (0.0-1.0). Default: 0.7. */
  temperature: number;
  /** Maximum tokens for a single AI response. Default: 4096. */
  maxTokens: number;
  /** Quality checks to run before accepting completion. */
  validationConfig?: ValidationConfig;
  /** Loop detection settings. */
  loopDetection?: LoopDetectionConfig;
  /** Inject goal-reminder every N iterations. */
  goalRevalidationInterval?: number;
  /** Abort after this many REPLAN cycles. Default: 3. */
  maxReplanAttempts: number;
  /** Abort when cumulative tokens reach this ceiling. */
  maxTotalTokens?: number;
  /** Abort when cumulative cost (USD) reaches this ceiling. */
  maxCostUsd?: number;
  /** Wall-clock timeout in seconds. */
  timeoutSecs?: number;
  /** Per-agent file scope whitelist. */
  allowedFiles?: string[];
  /** Context budget in tokens. */
  contextBudgetTokens?: number;
}

/** Create a default TaskAgentConfig. */
export function defaultTaskAgentConfig(): TaskAgentConfig {
  return {
    maxIterations: 100,
    temperature: 0.7,
    maxTokens: 4096,
    loopDetection: defaultLoopDetectionConfig(),
    goalRevalidationInterval: 10,
    maxReplanAttempts: 3,
  };
}

// ---------------------------------------------------------------------------
// Task agent class
// ---------------------------------------------------------------------------

/**
 * Autonomous task agent that runs a provider + tool loop until completion.
 *
 * Create with the constructor, then call `execute()` to run.
 */
export class TaskAgent {
  /** Unique agent ID. */
  readonly id: string;
  /** Task being executed. */
  task: Task;
  /** AI provider. */
  private provider: Provider;
  /** Shared environment context. */
  private context: AgentContext;
  /** Agent configuration. */
  private config: TaskAgentConfig;
  /** Conversation history. */
  private messages: Message[] = [];
  /** Current status. */
  private _status: TaskAgentStatus = { kind: "idle" };
  /** Total tokens used. */
  private totalTokensUsed = 0;
  /** Total cost USD. */
  private totalCostUsd = 0;

  constructor(
    id: string,
    task: Task,
    provider: Provider,
    context: AgentContext,
    config?: Partial<TaskAgentConfig>,
  ) {
    this.id = id;
    this.task = task;
    this.provider = provider;
    this.context = context;
    this.config = { ...defaultTaskAgentConfig(), ...config };
  }

  /** Get current status. */
  get status(): TaskAgentStatus {
    return this._status;
  }

  /** Execute the agent loop. */
  async execute(signal?: AbortSignal): Promise<TaskAgentResult> {
    this._status = { kind: "working", description: this.task.description };

    // Build an AgentRuntime adapter
    const runtime: AgentRuntime = {
      agentId: () => this.id,
      maxIterations: () => this.config.maxIterations,

      callProvider: async (): Promise<ChatResponse> => {
        const systemPrompt = this.config.systemPrompt ??
          `You are an autonomous agent working on a task. Task: ${this.task.description}`;

        const options = new ChatOptions({
          temperature: this.config.temperature,
          max_tokens: this.config.maxTokens,
          system: systemPrompt,
        });

        // Get available tools from the executor
        const tools: Tool[] = this.context.toolExecutor.availableTools();

        const response = await this.provider.chat(
          this.messages,
          tools.length > 0 ? tools : undefined,
          options,
        );

        // Track usage
        if (response.usage) {
          this.totalTokensUsed += (response.usage.prompt_tokens ?? 0) +
            (response.usage.completion_tokens ?? 0);
        }

        return response;
      },

      extractToolUses(response: ChatResponse): ToolUse[] {
        const content = response.message.content;
        if (typeof content === "string") return [];
        if (!Array.isArray(content)) return [];
        return content
          .filter(
            (block): block is Extract<typeof block, { type: "tool_use" }> =>
              typeof block === "object" &&
              block !== null &&
              "type" in block &&
              block.type === "tool_use",
          )
          .map((block) => ({
            id: block.id,
            name: block.name,
            input: block.input as Record<string, unknown>,
          }));
      },

      isCompletion(response: ChatResponse): boolean {
        return (
          response.finish_reason === "end_turn" ||
          response.finish_reason === "stop"
        );
      },

      executeTool: async (toolUse: ToolUse): Promise<ToolResult> => {
        const ctx = {
          agentId: this.id,
          workingDirectory: this.context.workingDirectory,
        };
        // deno-lint-ignore no-explicit-any
        return await this.context.toolExecutor.execute(toolUse, ctx as any);
      },

      getLockRequirement(
        toolUse: ToolUse,
      ): [string, LockType] | undefined {
        // Default lock inference based on tool name
        if (
          toolUse.name === "write_file" ||
          toolUse.name === "edit_file" ||
          toolUse.name === "create_file"
        ) {
          const path = (toolUse.input as Record<string, unknown>)
            .path as string | undefined;
          if (path) return [path, "write"];
        }
        if (toolUse.name === "read_file") {
          const path = (toolUse.input as Record<string, unknown>)
            .path as string | undefined;
          if (path) return [path, "read"];
        }
        return undefined;
      },

      onProviderResponse: (response: ChatResponse): void => {
        // Ensure it's a Message instance
        const msg = response.message instanceof Message
          ? response.message
          : new Message(response.message);
        this.messages.push(msg);
      },

      onToolResult: (_toolUse: ToolUse, result: ToolResult): void => {
        this.messages.push(
          new Message({
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: result.tool_use_id,
                content: result.content,
                is_error: result.is_error,
              },
            ],
          }),
        );
      },

      // deno-lint-ignore require-await
      async onCompletion(
        response: ChatResponse,
      ): Promise<string | undefined> {
        if (
          response.finish_reason === "end_turn" ||
          response.finish_reason === "stop"
        ) {
          const text = typeof response.message.content === "string"
            ? response.message.content
            : "completed";
          return text;
        }
        return undefined;
      },

      onIterationLimit(iterations: number): string {
        return `Hit iteration limit at ${iterations}`;
      },

      lifecycleHooks: (): AgentLifecycleHooks | undefined => {
        return this.context.lifecycleHooks;
      },

      contextBudgetTokens: (): number | undefined => {
        return this.config.contextBudgetTokens;
      },

      conversation: (): Message[] | undefined => {
        return this.messages;
      },
    };

    const executionResult = await runAgentLoop(
      runtime,
      this.context.communicationHub,
      this.context.fileLockManager,
      signal,
    );

    const result: TaskAgentResult = {
      agentId: this.id,
      taskId: this.task.id,
      success: executionResult.success,
      summary: executionResult.output,
      iterations: executionResult.iterations,
      replanCount: 0,
      budgetExhausted: false,
      totalTokensUsed: this.totalTokensUsed,
      totalCostUsd: this.totalCostUsd,
      timedOut: false,
      failureCategory: executionResult.success
        ? undefined
        : executionResult.output.includes("iteration limit")
        ? "iteration_limit_exceeded"
        : executionResult.output.includes("Loop detected")
        ? "loop_detected"
        : "unknown",
    };

    this._status = executionResult.success
      ? { kind: "completed", summary: executionResult.output }
      : { kind: "failed", error: executionResult.output };

    return result;
  }
}

/**
 * Spawn a task agent as a background async task.
 * Returns the task agent and a promise for its result.
 */
export function spawnTaskAgent(
  id: string,
  task: Task,
  provider: Provider,
  context: AgentContext,
  config?: Partial<TaskAgentConfig>,
  signal?: AbortSignal,
): { agent: TaskAgent; result: Promise<TaskAgentResult> } {
  const agent = new TaskAgent(id, task, provider, context, config);
  const result = agent.execute(signal);
  return { agent, result };
}
