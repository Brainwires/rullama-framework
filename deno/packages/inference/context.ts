/**
 * Agent Context - environment for autonomous task execution.
 *
 * {@link AgentContext} bundles the stable environment that a TaskAgent
 * operates in: working directory, tool executor, inter-agent communication,
 * file lock coordination, and the working set of files currently in context.
 *
 * @module
 */

import { WorkingSet } from "@rullama/core";
import type { ToolExecutor, ToolPreHook } from "@rullama/tool-runtime";

import type { AgentLifecycleHooks } from "./hooks.ts";
import type { CommunicationHub } from "@rullama/agent";
import type { FileLockManager } from "@rullama/agent";

// Re-export for convenience
export type { ToolPreHook } from "@rullama/tool-runtime";

/** Environment context for a task agent. */
export class AgentContext {
  /** Working directory used for resolving relative file paths. */
  workingDirectory: string;

  /** Executes tools on behalf of the agent. */
  toolExecutor: ToolExecutor;

  /** Inter-agent message bus. */
  communicationHub: CommunicationHub;

  /** Coordinates exclusive/shared file access across concurrent agents. */
  fileLockManager: FileLockManager;

  /** Tracks files currently loaded into the agent's context window. */
  workingSet: WorkingSet;

  /** Application-specific metadata passed through to tools. */
  metadata: Map<string, string>;

  /** Optional pre-execution hook for semantic tool validation. */
  preExecuteHook?: ToolPreHook;

  /** Optional lifecycle hooks for granular loop control. */
  lifecycleHooks?: AgentLifecycleHooks;

  constructor(
    workingDirectory: string,
    toolExecutor: ToolExecutor,
    communicationHub: CommunicationHub,
    fileLockManager: FileLockManager,
    workingSet?: WorkingSet,
  ) {
    this.workingDirectory = workingDirectory;
    this.toolExecutor = toolExecutor;
    this.communicationHub = communicationHub;
    this.fileLockManager = fileLockManager;
    this.workingSet = workingSet ?? new WorkingSet();
    this.metadata = new Map();
  }

  /** Add application-specific metadata. Returns `this` for chaining. */
  withMetadata(key: string, value: string): this {
    this.metadata.set(key, value);
    return this;
  }

  /** Set a pre-execution hook. Returns `this` for chaining. */
  withPreExecuteHook(hook: ToolPreHook): this {
    this.preExecuteHook = hook;
    return this;
  }

  /** Set lifecycle hooks. Returns `this` for chaining. */
  withLifecycleHooks(hooks: AgentLifecycleHooks): this {
    this.lifecycleHooks = hooks;
    return this;
  }
}
