/**
 * Three-State Model for comprehensive state tracking.
 *
 * Based on SagaLLM's Three-State Architecture, this module maintains three
 * separate state domains:
 *
 * 1. **Application State** - Domain logic (what resources exist, their current values)
 * 2. **Operation State** - Execution logs, timing, agent actions
 * 3. **Dependency State** - Constraint graphs, resource relationships
 *
 * This separation enables:
 * - Better debugging through complete operation history
 * - Deadlock detection via dependency graph analysis
 * - Validation of operations against current state
 * - Saga-style compensation using operation logs
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Application state types
// ---------------------------------------------------------------------------

/** Status of a tracked file. */
export interface FileStatus {
  /** Whether the file currently exists. */
  exists: boolean;
  /** Hash of the last recorded content. */
  contentHash: string;
  /** When the status was last updated (epoch ms). */
  lastModified: number;
  /** Agent holding the file lock, or null if unlocked. */
  lockedBy: string | null;
  /** True after `updateFile` until `markFilesClean` is called. */
  dirty: boolean;
}

/** Git repository state snapshot. */
export interface GitState {
  /** Checked-out branch name. */
  currentBranch: string;
  /** Commit hash at HEAD. */
  headCommit: string;
  /** Paths staged in the index. */
  stagedFiles: string[];
  /** Paths modified in the working tree. */
  modifiedFiles: string[];
  /** Whether unresolved merge conflicts are present. */
  hasConflicts: boolean;
}

/** Create a default empty git state. */
export function defaultGitState(): GitState {
  return {
    currentBranch: "",
    headCommit: "",
    stagedFiles: [],
    modifiedFiles: [],
    hasConflicts: false,
  };
}

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------

/** Application State: Domain-level resource tracking. */
export class ApplicationState {
  private files = new Map<string, FileStatus>();
  private resources = new Set<string>();
  private gitState: GitState = defaultGitState();

  /** Check if a resource exists. */
  resourceExists(resourceId: string): boolean {
    return this.files.has(resourceId) || this.resources.has(resourceId);
  }

  /** Mark a resource as existing. */
  markResourceExists(resourceId: string): void {
    this.resources.add(resourceId);
  }

  /** Mark a resource as deleted. */
  markResourceDeleted(resourceId: string): void {
    this.resources.delete(resourceId);
  }

  /** Update file status. */
  updateFile(path: string, contentHash: string): void {
    const existing = this.files.get(path);
    if (existing) {
      existing.contentHash = contentHash;
      existing.lastModified = Date.now();
      existing.dirty = true;
      existing.exists = true;
    } else {
      this.files.set(path, {
        exists: true,
        contentHash,
        lastModified: Date.now(),
        lockedBy: null,
        dirty: true,
      });
    }
  }

  /** Get all file statuses. */
  getAllFiles(): Map<string, FileStatus> {
    return new Map(this.files);
  }

  /** Update git state. */
  updateGitState(state: GitState): void {
    this.gitState = { ...state };
  }

  /** Get current git state. */
  getGitState(): GitState {
    return { ...this.gitState };
  }

  /** Lock a file for an agent. */
  lockFile(path: string, agentId: string): void {
    const file = this.files.get(path);
    if (file) file.lockedBy = agentId;
  }

  /** Unlock a file. */
  unlockFile(path: string): void {
    const file = this.files.get(path);
    if (file) file.lockedBy = null;
  }

  /** Mark all source files as clean. */
  markFilesClean(): void {
    for (const file of this.files.values()) file.dirty = false;
  }
}

// ---------------------------------------------------------------------------
// Operation state types
// ---------------------------------------------------------------------------

/** Status of an operation in its lifecycle. */
export type OperationLogStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "compensated";

/** Log entry for a tracked operation. */
export interface OperationLog {
  /** Operation identifier (see `OperationState.generateId`). */
  id: string;
  /** Agent executing the operation. */
  agentId: string;
  /** Free-form operation category (e.g. `"file_write"`). */
  operationType: string;
  /** When the operation started (epoch ms). */
  startedAt: number;
  /** When it finished (epoch ms), or null while running. */
  completedAt: number | null;
  /** Lifecycle status. */
  status: OperationLogStatus;
  /** Caller-supplied inputs, kept for debugging and compensation. */
  // deno-lint-ignore no-explicit-any
  inputs: any;
  /** Outputs recorded by `completeOperation`, or null until then. */
  // deno-lint-ignore no-explicit-any
  outputs: any | null;
  /** Error text recorded on failure, or null. */
  error: string | null;
  /** IDs of operations nested under this one. */
  childOperations: string[];
  /** ID of the enclosing operation, or null at top level. */
  parentOperation: string | null;
  /** Resources the operation reads or requires. */
  resourcesNeeded: string[];
  /** Resources the operation creates or modifies. */
  resourcesProduced: string[];
}

/** Create a new operation log entry. */
export function createOperationLog(
  id: string,
  agentId: string,
  operationType: string,
  // deno-lint-ignore no-explicit-any
  inputs: any = {},
): OperationLog {
  return {
    id,
    agentId,
    operationType,
    startedAt: Date.now(),
    completedAt: null,
    status: "running",
    inputs,
    outputs: null,
    error: null,
    childOperations: [],
    parentOperation: null,
    resourcesNeeded: [],
    resourcesProduced: [],
  };
}

// ---------------------------------------------------------------------------
// Operation State
// ---------------------------------------------------------------------------

/** Operation State: Execution logging and history. */
export class OperationState {
  private operations = new Map<string, OperationLog>();
  private agentOperations = new Map<string, string[]>();
  private activeOperations = new Set<string>();
  private nextId = 1;

  /** Generate a new unique operation ID. */
  generateId(): string {
    return `op-${this.nextId++}`;
  }

  /** Start tracking a new operation. Returns the operation ID. */
  startOperation(log: OperationLog): string {
    this.operations.set(log.id, { ...log });
    const agentOps = this.agentOperations.get(log.agentId) ?? [];
    agentOps.push(log.id);
    this.agentOperations.set(log.agentId, agentOps);
    this.activeOperations.add(log.id);
    return log.id;
  }

  /** Complete an operation. */
  completeOperation(
    operationId: string,
    success: boolean,
    // deno-lint-ignore no-explicit-any
    outputs: any = null,
    error: string | null = null,
  ): void {
    this.activeOperations.delete(operationId);
    const op = this.operations.get(operationId);
    if (op) {
      op.completedAt = Date.now();
      op.status = success ? "completed" : "failed";
      op.outputs = outputs;
      op.error = error;
    }
  }

  /** Mark an operation as compensated. */
  markCompensated(operationId: string): void {
    const op = this.operations.get(operationId);
    if (op) op.status = "compensated";
  }

  /** Get active operations. */
  getActiveOperations(): OperationLog[] {
    return [...this.activeOperations]
      .map((id) => this.operations.get(id))
      .filter((o): o is OperationLog => o != null)
      .map((o) => ({ ...o }));
  }

  /** Get active operation IDs. */
  getActiveOperationIds(): string[] {
    return [...this.activeOperations];
  }

  /** Get operation by ID. */
  getOperation(operationId: string): OperationLog | undefined {
    const op = this.operations.get(operationId);
    return op ? { ...op } : undefined;
  }

  /** Get all operations for an agent. */
  getAgentOperations(agentId: string): OperationLog[] {
    const ids = this.agentOperations.get(agentId) ?? [];
    return ids
      .map((id) => this.operations.get(id))
      .filter((o): o is OperationLog => o != null)
      .map((o) => ({ ...o }));
  }

  /** Add child operation to parent. */
  addChildOperation(parentId: string, childId: string): void {
    const parent = this.operations.get(parentId);
    if (parent) parent.childOperations.push(childId);
    const child = this.operations.get(childId);
    if (child) child.parentOperation = parentId;
  }
}

// ---------------------------------------------------------------------------
// Dependency state types
// ---------------------------------------------------------------------------

/** Type of dependency relationship. */
export type DependencyType =
  | "blocked_by"
  | "produces"
  | "conflicts_with"
  | "reads"
  | "writes"
  | "waits_for";

/** How strictly a dependency must be respected. */
export type DependencyStrength = "hard" | "soft" | "advisory";

/** An edge in the dependency graph. */
export interface DependencyEdge {
  /** Kind of relationship the edge encodes. */
  dependencyType: DependencyType;
  /** How strictly the relationship must be honoured. */
  strength: DependencyStrength;
}

/** Type of resource node. */
export type ResourceNodeType =
  | "file"
  | "build_lock"
  | "test_lock"
  | "git_index"
  | "git_branch"
  | "agent"
  | "generic";

/** A node in the dependency graph. */
interface ResourceNode {
  resourceId: string;
  resourceType: ResourceNodeType;
  currentHolder: string | null;
}

// ---------------------------------------------------------------------------
// Dependency State (simplified graph using adjacency list)
// ---------------------------------------------------------------------------

/** Dependency State: Resource relationship graph. */
export class DependencyState {
  /** Nodes indexed by resource ID. */
  private nodes = new Map<string, ResourceNode>();
  /** Adjacency list: from -> [{to, edge}]. Edge from "to" to "from" in execution terms. */
  private edges = new Map<
    string,
    Array<{ target: string; edge: DependencyEdge }>
  >();

  /** Insert a node (with an empty adjacency list) for `resourceId` if none exists. */
  private ensureNode(
    resourceId: string,
    resourceType: ResourceNodeType = "generic",
  ): void {
    if (!this.nodes.has(resourceId)) {
      this.nodes.set(resourceId, {
        resourceId,
        resourceType,
        currentHolder: null,
      });
      this.edges.set(resourceId, []);
    }
  }

  /**
   * Add a dependency: `from` depends on `to`.
   * In graph terms: to -> from (to must be processed before from).
   */
  addDependency(from: string, to: string, edge: DependencyEdge): void {
    this.ensureNode(from);
    this.ensureNode(to);
    // Edge goes from `to` to `from` (to must execute before from)
    const adj = this.edges.get(to)!;
    adj.push({ target: from, edge });
  }

  /** Remove a dependency. */
  removeDependency(from: string, to: string): void {
    const adj = this.edges.get(to);
    if (adj) {
      const idx = adj.findIndex((e) => e.target === from);
      if (idx >= 0) adj.splice(idx, 1);
    }
  }

  /** Check if acquiring resources would create a cycle (deadlock). */
  wouldDeadlock(agentId: string, resources: string[]): boolean {
    const agentNodeId = `agent:${agentId}`;
    this.ensureNode(agentNodeId, "agent");

    // Temporarily add edges from agent to requested resources
    const tempEdges: Array<{ from: string; to: string }> = [];
    for (const resource of resources) {
      if (this.nodes.has(resource)) {
        const adj = this.edges.get(agentNodeId)!;
        adj.push({
          target: resource,
          edge: { dependencyType: "waits_for", strength: "hard" },
        });
        tempEdges.push({ from: agentNodeId, to: resource });
      }
    }

    const hasCycle = this.detectCycle();

    // Remove temporary edges
    for (const te of tempEdges) {
      const adj = this.edges.get(te.from);
      if (adj) {
        const idx = adj.findIndex((e) => e.target === te.to);
        if (idx >= 0) adj.splice(idx, 1);
      }
    }

    return hasCycle;
  }

  /** Set the current holder of a resource. */
  setHolder(resourceId: string, agentId: string | null): void {
    const node = this.nodes.get(resourceId);
    if (node) node.currentHolder = agentId;
  }

  /** Get current resource holders. */
  getCurrentHolders(): Map<string, string> {
    const result = new Map<string, string>();
    for (const node of this.nodes.values()) {
      if (node.currentHolder) {
        result.set(node.resourceId, node.currentHolder);
      }
    }
    return result;
  }

  /** Get resources held by an agent. */
  getAgentResources(agentId: string): string[] {
    const result: string[] = [];
    for (const node of this.nodes.values()) {
      if (node.currentHolder === agentId) result.push(node.resourceId);
    }
    return result;
  }

  /** Topological sort of operations respecting dependencies. */
  getExecutionOrder(operationIds: string[]): string[] {
    // Compute in-degree for the subset
    const remaining = new Set(operationIds);
    const ordered: string[] = [];

    while (remaining.size > 0) {
      let madeProgress = false;
      for (const opId of [...remaining]) {
        // Check if all dependencies are satisfied (not in remaining)
        const allDepsSatisfied = this.getIncomingDeps(opId).every(
          (dep) => !remaining.has(dep),
        );
        if (allDepsSatisfied || !this.nodes.has(opId)) {
          ordered.push(opId);
          remaining.delete(opId);
          madeProgress = true;
        }
      }
      if (!madeProgress) {
        // Cycle detected, add remaining in arbitrary order
        for (const id of remaining) ordered.push(id);
        break;
      }
    }
    return ordered;
  }

  // ── Private helpers ─────────────────────────────────────────────────────

  /** Get incoming dependencies for a node (nodes that must execute before it). */
  private getIncomingDeps(nodeId: string): string[] {
    const deps: string[] = [];
    for (const [sourceId, adj] of this.edges) {
      if (adj.some((e) => e.target === nodeId)) {
        deps.push(sourceId);
      }
    }
    return deps;
  }

  /** Detect if the graph has any cycle using DFS. */
  private detectCycle(): boolean {
    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = new Map<string, number>();
    for (const id of this.nodes.keys()) color.set(id, WHITE);

    const dfs = (u: string): boolean => {
      color.set(u, GRAY);
      for (const { target } of this.edges.get(u) ?? []) {
        const c = color.get(target);
        if (c === GRAY) return true; // back edge = cycle
        if (c === WHITE && dfs(target)) return true;
      }
      color.set(u, BLACK);
      return false;
    };

    for (const id of this.nodes.keys()) {
      if (color.get(id) === WHITE && dfs(id)) return true;
    }
    return false;
  }
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/** Result of validating an operation. */
export interface StateValidationResult {
  /** True when `errors` is empty. */
  valid: boolean;
  /** Blocking problems: resource overlap with a running operation, or a deadlock. */
  errors: string[];
  /** Non-blocking notes, e.g. a needed resource that does not exist yet. */
  warnings: string[];
}

/** A proposed operation to validate. */
export interface ProposedOperation {
  /** Agent that wants to run the operation. */
  agentId: string;
  /** Free-form operation category. */
  operationType: string;
  /** Resources the operation would read or require. */
  resourcesNeeded: string[];
  /** Resources the operation would create or modify. */
  resourcesProduced: string[];
}

/** Types of application state changes. */
export type ApplicationChange =
  | { kind: "file_modified"; path: string; newHash: string }
  | { kind: "artifact_invalidated"; artifactId: string }
  | { kind: "git_state_changed"; newState: GitState }
  | { kind: "resource_created"; resourceId: string }
  | { kind: "resource_deleted"; resourceId: string };

/** State change to record. */
export interface StateChange {
  /** Operation that produced the change (informational; not applied to operation state). */
  operationId: string;
  /** Application-state mutations to apply in order. */
  applicationChanges: ApplicationChange[];
  /** Dependency edges to add (`from` depends on `to`). */
  newDependencies: Array<{ from: string; to: string; edge: DependencyEdge }>;
}

/** Snapshot of current state. */
export interface StateSnapshot {
  /** Copy of every tracked file's status keyed by path. */
  files: Map<string, FileStatus>;
  /** Resource ID -> holding agent, from the dependency graph. */
  locks: Map<string, string>;
  /** Copy of the recorded git state. */
  gitState: GitState;
  /** IDs of operations still running. */
  activeOperations: string[];
}

// ---------------------------------------------------------------------------
// ThreeStateModel
// ---------------------------------------------------------------------------

/** Three-State Model for comprehensive state tracking. */
export class ThreeStateModel {
  /** Domain resources: files, generic resources, git state. */
  readonly applicationState: ApplicationState;
  /** Execution log of operations per agent. */
  readonly operationState: OperationState;
  /** Resource/agent dependency graph used for deadlock checks and ordering. */
  readonly dependencyState: DependencyState;

  constructor() {
    this.applicationState = new ApplicationState();
    this.operationState = new OperationState();
    this.dependencyState = new DependencyState();
  }

  /** Validate that a proposed operation is consistent with current state. */
  validateOperation(op: ProposedOperation): StateValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check application state - do required resources exist?
    for (const resource of op.resourcesNeeded) {
      if (!this.applicationState.resourceExists(resource)) {
        warnings.push(`Resource '${resource}' does not exist yet`);
      }
    }

    // Check operation state - any conflicting running operations?
    const activeOps = this.operationState.getActiveOperations();
    for (const activeOp of activeOps) {
      const activeResources = new Set([
        ...activeOp.resourcesNeeded,
        ...activeOp.resourcesProduced,
      ]);
      const proposedResources = new Set([
        ...op.resourcesNeeded,
        ...op.resourcesProduced,
      ]);
      const overlap: string[] = [];
      for (const r of proposedResources) {
        if (activeResources.has(r)) overlap.push(r);
      }
      if (overlap.length > 0) {
        errors.push(
          `Conflict with running operation '${activeOp.id}': shared resources [${
            overlap.join(", ")
          }]`,
        );
      }
    }

    // Check dependency state - would this create a deadlock?
    if (this.dependencyState.wouldDeadlock(op.agentId, op.resourcesNeeded)) {
      errors.push("Operation would create a deadlock");
    }

    return { valid: errors.length === 0, errors, warnings };
  }

  /** Record state change from an operation. */
  recordStateChange(change: StateChange): void {
    for (const appChange of change.applicationChanges) {
      switch (appChange.kind) {
        case "file_modified":
          this.applicationState.updateFile(appChange.path, appChange.newHash);
          break;
        case "git_state_changed":
          this.applicationState.updateGitState(appChange.newState);
          break;
        case "resource_created":
          this.applicationState.markResourceExists(appChange.resourceId);
          break;
        case "resource_deleted":
          this.applicationState.markResourceDeleted(appChange.resourceId);
          break;
        case "artifact_invalidated":
          // Simplified: no artifact tracking in TS port
          break;
      }
    }

    for (const dep of change.newDependencies) {
      this.dependencyState.addDependency(dep.from, dep.to, dep.edge);
    }
  }

  /** Get a snapshot of current state. */
  snapshot(): StateSnapshot {
    return {
      files: this.applicationState.getAllFiles(),
      locks: this.dependencyState.getCurrentHolders(),
      gitState: this.applicationState.getGitState(),
      activeOperations: this.operationState.getActiveOperationIds(),
    };
  }
}
