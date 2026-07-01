/**
 * Agent Pool - Manages a pool of background task agents.
 *
 * Handles lifecycle of agents: spawning, monitoring, stopping, and
 * awaiting results. All agents in the pool share the same provider,
 * tool registry, communication hub, and file lock manager.
 *
 * This is a lightweight TypeScript port of the Rust AgentPool. In Deno
 * we model each agent as a `Promise<TaskAgentResult>` rather than a
 * Tokio JoinHandle.
 *
 * @module
 */

import type { Provider, Task } from "@rullama/core";
import type { AgentContext } from "./context.ts";
import {
  TaskAgent,
  type TaskAgentConfig,
  type TaskAgentResult,
  type TaskAgentStatus,
} from "./task_agent.ts";

// ---------------------------------------------------------------------------
// Handle type
// ---------------------------------------------------------------------------

interface AgentHandle {
  agent: TaskAgent;
  promise: Promise<TaskAgentResult>;
  finished: boolean;
  result?: TaskAgentResult;
  error?: Error;
  abortController: AbortController;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/** Statistics about the agent pool. */
export interface AgentPoolStats {
  /** Maximum concurrent agents allowed. */
  maxAgents: number;
  /** Total agents currently tracked (running + awaiting cleanup). */
  totalAgents: number;
  /** Agents that are currently running. */
  running: number;
  /** Agents that have finished but not yet cleaned up. */
  completed: number;
  /** Agents that failed. */
  failed: number;
}

// ---------------------------------------------------------------------------
// AgentPool
// ---------------------------------------------------------------------------

/** Manages a pool of background TaskAgents. */
export class AgentPool {
  private maxAgents: number;
  private agents = new Map<string, AgentHandle>();
  private provider: Provider;
  private defaultContext: AgentContext;
  private nextId = 0;

  /**
   * Create a new agent pool.
   *
   * @param maxAgents - Maximum number of concurrently running agents.
   * @param provider - AI provider shared by all agents.
   * @param context - Default agent context for spawned agents.
   */
  constructor(
    maxAgents: number,
    provider: Provider,
    context: AgentContext,
  ) {
    this.maxAgents = maxAgents;
    this.provider = provider;
    this.defaultContext = context;
  }

  /**
   * Spawn a new task agent and start it running.
   *
   * Returns the agent ID. Use {@link awaitCompletion} to wait for the result.
   * Throws if the pool is at capacity.
   */
  spawnAgent(
    task: Task,
    config?: Partial<TaskAgentConfig>,
    context?: AgentContext,
  ): string {
    if (this.agents.size >= this.maxAgents) {
      throw new Error(
        `Agent pool is full (${this.agents.size}/${this.maxAgents})`,
      );
    }

    const agentId = `agent-${this.nextId++}`;
    const ctx = context ?? this.defaultContext;

    const agent = new TaskAgent(
      agentId,
      task,
      this.provider,
      ctx,
      config,
    );

    const abortController = new AbortController();

    const promise = agent.execute().then(
      (result) => {
        const handle = this.agents.get(agentId);
        if (handle) {
          handle.finished = true;
          handle.result = result;
        }
        return result;
      },
      (err) => {
        const handle = this.agents.get(agentId);
        if (handle) {
          handle.finished = true;
          handle.error = err instanceof Error ? err : new Error(String(err));
        }
        throw err;
      },
    );

    this.agents.set(agentId, {
      agent,
      promise,
      finished: false,
      abortController,
    });

    return agentId;
  }

  /** Get the current status of an agent. */
  getStatus(agentId: string): TaskAgentStatus | undefined {
    return this.agents.get(agentId)?.agent.status;
  }

  /** Get the task assigned to an agent. */
  getTask(agentId: string): Task | undefined {
    return this.agents.get(agentId)?.agent.task;
  }

  /** Remove an agent from the pool. Aborts it if still running. */
  stopAgent(agentId: string): void {
    const handle = this.agents.get(agentId);
    if (!handle) throw new Error(`Agent ${agentId} not found`);
    handle.abortController.abort();
    this.agents.delete(agentId);
  }

  /** Wait for an agent to finish and return its result. Removes it from the pool. */
  async awaitCompletion(agentId: string): Promise<TaskAgentResult> {
    const handle = this.agents.get(agentId);
    if (!handle) throw new Error(`Agent ${agentId} not found`);
    try {
      const result = await handle.promise;
      return result;
    } finally {
      this.agents.delete(agentId);
    }
  }

  /** List all agents currently in the pool with their status. */
  listActive(): Array<{ id: string; status: TaskAgentStatus }> {
    const out: Array<{ id: string; status: TaskAgentStatus }> = [];
    for (const [id, handle] of this.agents) {
      out.push({ id, status: handle.agent.status });
    }
    return out;
  }

  /** Number of agents currently in the pool. */
  activeCount(): number {
    return this.agents.size;
  }

  /** Check if an agent is still running. */
  isRunning(agentId: string): boolean {
    const handle = this.agents.get(agentId);
    return handle != null && !handle.finished;
  }

  /** Remove all finished agents and return their results. */
  async cleanupCompleted(): Promise<
    Array<{ id: string; result?: TaskAgentResult; error?: Error }>
  > {
    const finished: string[] = [];
    for (const [id, handle] of this.agents) {
      if (handle.finished) finished.push(id);
    }

    const results: Array<{
      id: string;
      result?: TaskAgentResult;
      error?: Error;
    }> = [];
    for (const id of finished) {
      const handle = this.agents.get(id)!;
      this.agents.delete(id);
      try {
        const result = await handle.promise;
        results.push({ id, result });
      } catch (err) {
        results.push({
          id,
          error: err instanceof Error ? err : new Error(String(err)),
        });
      }
    }
    return results;
  }

  /** Wait for every agent in the pool to finish. */
  async awaitAll(): Promise<
    Array<{ id: string; result?: TaskAgentResult; error?: Error }>
  > {
    const ids = [...this.agents.keys()];
    const results: Array<{
      id: string;
      result?: TaskAgentResult;
      error?: Error;
    }> = [];
    for (const id of ids) {
      try {
        const result = await this.awaitCompletion(id);
        results.push({ id, result });
      } catch (err) {
        results.push({
          id,
          error: err instanceof Error ? err : new Error(String(err)),
        });
      }
    }
    return results;
  }

  /** Abort all agents and clear the pool. */
  shutdown(): void {
    for (const [, handle] of this.agents) {
      handle.abortController.abort();
    }
    this.agents.clear();
  }

  /** Get a statistical snapshot of the pool. */
  stats(): AgentPoolStats {
    let running = 0;
    let completed = 0;
    let failed = 0;
    for (const handle of this.agents.values()) {
      if (handle.finished) {
        if (handle.error) failed++;
        else completed++;
      } else {
        running++;
      }
    }
    return {
      maxAgents: this.maxAgents,
      totalAgents: this.agents.size,
      running,
      completed,
      failed,
    };
  }
}
