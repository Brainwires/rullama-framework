/**
 * @module @rullama/agent
 *
 * Agent coordination primitives. Equivalent to Rust's `rullama-agent` crate.
 *
 * Ships only:
 * - `CommunicationHub` — inter-agent messaging bus
 * - `FileLockManager` — file access coordination
 * - `TaskManager` / `TaskQueue` — hierarchical task decomposition + scheduling
 * - `ExecutionGraph` — telemetry-graph for runs
 * - Coordination patterns: ContractNet, Saga, OptimisticConcurrency,
 *   MarketAllocator, ThreeStateModel, WaitQueue
 *
 * LLM workhorses live in `@rullama/inference`. MDAP / SEAL / Skills / Eval
 * each have their own package. `AgentPool` is in `@rullama/inference` too.
 * Import from those directly — this package does not re-export them.
 */

// ── Communication ──────────────────────────────────────────────────────
export {
  type AgentMessage,
  CommunicationHub,
  type ConflictInfo,
  type ConflictType,
  type GitOperationType,
  type MessageEnvelope,
  type OperationType,
} from "./communication.ts";

// ── File locks ─────────────────────────────────────────────────────────
export {
  FileLockManager,
  isLockExpired,
  type LockGuard,
  type LockInfo,
  type LockStats,
  lockTimeRemaining,
  type LockType,
} from "./file_locks.ts";

// ── Task manager ───────────────────────────────────────────────────────
export {
  formatDurationSecs,
  TaskManager,
  type TaskStats,
  type TimeStats,
} from "./task_manager.ts";

// ── Task queue ─────────────────────────────────────────────────────────
export { type QueuedTask, TaskQueue } from "./task_queue.ts";

// ── Execution Graph ────────────────────────────────────────────────────
export {
  ExecutionGraph,
  type RunTelemetry,
  type StepNode,
  telemetryFromGraph,
  type ToolCallRecord,
} from "./execution_graph.ts";

// ── Coordination patterns ──────────────────────────────────────────────
export * from "./coordination/mod.ts";
