/**
 * Contract-Net Protocol for Task Allocation.
 *
 * Implements the Contract-Net Protocol where:
 * 1. Manager broadcasts task announcements
 * 2. Agents submit bids based on capability and availability
 * 3. Manager awards contract to best bidder
 * 4. Winner executes, others continue with other work
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Task requirements
// ---------------------------------------------------------------------------

/** Requirements for a task. */
export interface TaskRequirements {
  /** Required capabilities (e.g., "rust", "git", "testing"). */
  capabilities: string[];
  /** Resources needed. */
  resourcesNeeded: string[];
  /** Estimated complexity (1-10). */
  complexity: number;
  /** Priority level (higher = more important). */
  priority: number;
  /** Minimum capability score required (0.0-1.0). */
  minCapabilityScore: number;
}

/** Create default task requirements. */
export function defaultTaskRequirements(): TaskRequirements {
  return {
    capabilities: [],
    resourcesNeeded: [],
    complexity: 0,
    priority: 0,
    minCapabilityScore: 0,
  };
}

// ---------------------------------------------------------------------------
// Task announcement
// ---------------------------------------------------------------------------

/** Task announcement broadcast to all agents. */
export interface TaskAnnouncement {
  /** Unique task identifier. */
  taskId: string;
  /** Human-readable description. */
  description: string;
  /** Task requirements. */
  requirements: TaskRequirements;
  /** When the task should be completed by (epoch ms, optional). */
  deadline?: number;
  /** When bidding closes (epoch ms). */
  bidDeadline: number;
  /** Who announced the task. */
  announcer: string;
  /** When announced (epoch ms). */
  announcedAt: number;
}

/** Check if bidding is still open for an announcement. */
export function isBiddingOpen(a: TaskAnnouncement): boolean {
  return Date.now() < a.bidDeadline;
}

/** Time remaining to bid (ms). */
export function bidTimeRemaining(a: TaskAnnouncement): number {
  return Math.max(0, a.bidDeadline - Date.now());
}

// ---------------------------------------------------------------------------
// Task bid
// ---------------------------------------------------------------------------

/** Bid submitted by an agent. */
export interface TaskBid {
  /** Agent submitting the bid. */
  agentId: string;
  /** Task being bid on. */
  taskId: string;
  /** Agent's capability match score (0.0-1.0). */
  capabilityScore: number;
  /** Agent's current load (0.0 = idle, 1.0 = fully busy). */
  currentLoad: number;
  /** Estimated completion time (ms). */
  estimatedDurationMs: number;
  /** Any constraints or conditions. */
  conditions: string[];
  /** When the bid was submitted (epoch ms). */
  submittedAt: number;
}

/** Calculate overall bid score (higher is better). */
export function bidScore(bid: TaskBid): number {
  const availability = 1.0 - bid.currentLoad;
  const speed = 1.0 / (1.0 + bid.estimatedDurationMs / 60_000);
  return 0.4 * bid.capabilityScore + 0.3 * availability + 0.3 * speed;
}

/** Calculate score with custom weights. */
export function bidScoreWeighted(
  bid: TaskBid,
  capabilityWeight: number,
  availabilityWeight: number,
  speedWeight: number,
): number {
  const total = capabilityWeight + availabilityWeight + speedWeight;
  if (total === 0) return 0;
  const availability = 1.0 - bid.currentLoad;
  const speed = 1.0 / (1.0 + bid.estimatedDurationMs / 60_000);
  return (
    (capabilityWeight * bid.capabilityScore +
      availabilityWeight * availability +
      speedWeight * speed) /
    total
  );
}

// ---------------------------------------------------------------------------
// Bid evaluation strategy
// ---------------------------------------------------------------------------

/** Strategy for evaluating bids. */
export type BidEvaluationStrategy =
  | { kind: "highest_score" }
  | { kind: "fastest_completion" }
  | { kind: "load_balancing" }
  | { kind: "best_capability" }
  | {
    kind: "custom_weights";
    capability: number;
    availability: number;
    speed: number;
  };

// ---------------------------------------------------------------------------
// Contract message
// ---------------------------------------------------------------------------

/** Protocol messages. */
export type ContractMessage =
  | { kind: "announce"; announcement: TaskAnnouncement }
  | { kind: "bid"; bid: TaskBid }
  | { kind: "award"; taskId: string; winner: string; score: number }
  | { kind: "no_award"; taskId: string; reason: string }
  | { kind: "accept"; taskId: string; agentId: string }
  | { kind: "decline"; taskId: string; agentId: string; reason: string }
  | {
    kind: "complete";
    taskId: string;
    agentId: string;
    success: boolean;
    result?: string;
  }
  | { kind: "cancel"; taskId: string; reason: string };

// ---------------------------------------------------------------------------
// Awarded contract
// ---------------------------------------------------------------------------

/** Information about an awarded contract. */
export interface AwardedContract {
  /** Task the contract was awarded for. */
  taskId: string;
  /** Agent ID of the winning bidder. */
  winner: string;
  /** The bid selected by the evaluation strategy. */
  winningBid: TaskBid;
  /** When the award was made (epoch ms). */
  awardedAt: number;
  /** Whether the winner has called `acceptAward`. */
  accepted: boolean;
  /** Set by `completeTask` to the reported success flag; undefined until then. */
  completed?: boolean;
}

// ---------------------------------------------------------------------------
// Task status (contract-net specific)
// ---------------------------------------------------------------------------

/** Task status in the contract-net protocol. */
export type ContractTaskStatus =
  | "open_for_bids"
  | "awarded"
  | "in_progress"
  | "completed";

// ---------------------------------------------------------------------------
// Contract-net manager
// ---------------------------------------------------------------------------

/** Contract-Net Protocol manager. */
export class ContractNetManager {
  private announcements = new Map<string, TaskAnnouncement>();
  private bids = new Map<string, TaskBid[]>();
  private awarded = new Map<string, AwardedContract>();
  private evaluationStrategy: BidEvaluationStrategy;
  private nextTaskId = 1;
  private listeners: Array<(msg: ContractMessage) => void> = [];

  /**
   * Create a manager.
   * @param strategy How `awardTask` picks a winner; defaults to `{ kind: "highest_score" }` (composite `bidScore`).
   */
  constructor(strategy?: BidEvaluationStrategy) {
    this.evaluationStrategy = strategy ?? { kind: "highest_score" };
  }

  /** Subscribe to protocol messages. Returns an unsubscribe function. */
  subscribe(callback: (msg: ContractMessage) => void): () => void {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== callback);
    };
  }

  /** Deliver `msg` to every subscriber, ignoring exceptions thrown by listeners. */
  private emit(msg: ContractMessage): void {
    for (const l of this.listeners) {
      try {
        l(msg);
      } catch { /* ignore listener errors */ }
    }
  }

  /** Generate a unique task ID. */
  generateTaskId(): string {
    return `task-${this.nextTaskId++}`;
  }

  /** Announce a task for bidding. Returns the task ID. */
  announceTask(announcement: TaskAnnouncement): string {
    const taskId = announcement.taskId || this.generateTaskId();
    const a = { ...announcement, taskId, announcedAt: Date.now() };
    this.announcements.set(taskId, a);
    this.bids.set(taskId, []);
    this.emit({ kind: "announce", announcement: a });
    return taskId;
  }

  /** Process a bid from an agent. */
  receiveBid(bid: TaskBid): void {
    const announcement = this.announcements.get(bid.taskId);
    if (!announcement) throw new Error(`Unknown task: ${bid.taskId}`);
    if (!isBiddingOpen(announcement)) {
      throw new Error("Bid deadline passed");
    }
    if (bid.capabilityScore < announcement.requirements.minCapabilityScore) {
      throw new Error(
        `Capability score ${bid.capabilityScore} below minimum ${announcement.requirements.minCapabilityScore}`,
      );
    }

    const taskBids = this.bids.get(bid.taskId)!;
    const existing = taskBids.findIndex((b) => b.agentId === bid.agentId);
    if (existing !== -1) taskBids.splice(existing, 1);
    taskBids.push(bid);

    this.emit({ kind: "bid", bid });
  }

  /** Evaluate bids and award task to winner. Returns winner agent ID. */
  awardTask(taskId: string): string | undefined {
    const taskBids = this.bids.get(taskId);
    if (!taskBids || taskBids.length === 0) {
      this.emit({
        kind: "no_award",
        taskId,
        reason: "No bids received",
      });
      return undefined;
    }

    const winner = this.evaluateBids(taskBids);
    if (!winner) return undefined;

    const score = bidScore(winner);

    this.awarded.set(taskId, {
      taskId,
      winner: winner.agentId,
      winningBid: winner,
      awardedAt: Date.now(),
      accepted: false,
    });

    this.emit({ kind: "award", taskId, winner: winner.agentId, score });
    return winner.agentId;
  }

  /**
   * Pick the winning bid per the configured strategy: composite score, lowest
   * `estimatedDurationMs`, lowest `currentLoad`, highest `capabilityScore`, or
   * a weighted score. Ties keep the earlier bid.
   */
  private evaluateBids(bids: TaskBid[]): TaskBid | undefined {
    if (bids.length === 0) return undefined;

    switch (this.evaluationStrategy.kind) {
      case "highest_score":
        return bids.reduce((best, b) =>
          bidScore(b) > bidScore(best) ? b : best
        );
      case "fastest_completion":
        return bids.reduce((best, b) =>
          b.estimatedDurationMs < best.estimatedDurationMs ? b : best
        );
      case "load_balancing":
        return bids.reduce((best, b) =>
          b.currentLoad < best.currentLoad ? b : best
        );
      case "best_capability":
        return bids.reduce((best, b) =>
          b.capabilityScore > best.capabilityScore ? b : best
        );
      case "custom_weights": {
        const { capability, availability, speed } = this.evaluationStrategy;
        return bids.reduce((best, b) =>
          bidScoreWeighted(b, capability, availability, speed) >
              bidScoreWeighted(best, capability, availability, speed)
            ? b
            : best
        );
      }
    }
  }

  /** Record acceptance of an award. */
  acceptAward(taskId: string, agentId: string): void {
    const contract = this.awarded.get(taskId);
    if (!contract) throw new Error(`No award found for task: ${taskId}`);
    if (contract.winner !== agentId) {
      throw new Error(`Agent ${agentId} is not the winner of task ${taskId}`);
    }
    contract.accepted = true;
    this.emit({ kind: "accept", taskId, agentId });
  }

  /** Record decline of an award. */
  declineAward(taskId: string, agentId: string, reason: string): void {
    this.awarded.delete(taskId);
    this.emit({ kind: "decline", taskId, agentId, reason });
  }

  /** Record task completion. */
  completeTask(
    taskId: string,
    agentId: string,
    success: boolean,
    result?: string,
  ): void {
    const contract = this.awarded.get(taskId);
    if (!contract) throw new Error(`No contract found for task: ${taskId}`);
    if (contract.winner !== agentId) {
      throw new Error(
        `Agent ${agentId} is not the contractor for task ${taskId}`,
      );
    }
    contract.completed = success;
    this.emit({ kind: "complete", taskId, agentId, success, result });
    this.announcements.delete(taskId);
    this.bids.delete(taskId);
  }

  /** Get task status. */
  getTaskStatus(taskId: string): ContractTaskStatus | undefined {
    const contract = this.awarded.get(taskId);
    if (contract) {
      if (contract.completed != null) return "completed";
      if (contract.accepted) return "in_progress";
      return "awarded";
    }
    if (this.announcements.has(taskId)) return "open_for_bids";
    return undefined;
  }

  /** Get all pending tasks. */
  getPendingTasks(): TaskAnnouncement[] {
    return [...this.announcements.values()];
  }

  /** Get bids for a task. */
  getBids(taskId: string): TaskBid[] {
    return this.bids.get(taskId) ?? [];
  }
}

// ---------------------------------------------------------------------------
// Contract participant (agent-side)
// ---------------------------------------------------------------------------

/** Agent-side contract participant. */
export class ContractParticipant {
  /** ID this participant bids under. */
  readonly agentId: string;
  /** Capability tags matched against `TaskRequirements.capabilities`. */
  readonly capabilities: string[];
  private currentTasks: string[] = [];
  private maxConcurrent: number;

  /**
   * Create a participant.
   * @param agentId Agent identifier placed on generated bids.
   * @param capabilities Capability tags the agent offers.
   * @param maxConcurrent Task count at which `shouldBid` returns false (default 3).
   */
  constructor(
    agentId: string,
    capabilities: string[],
    maxConcurrent = 3,
  ) {
    this.agentId = agentId;
    this.capabilities = capabilities;
    this.maxConcurrent = maxConcurrent;
  }

  /** Check if agent should bid on a task. */
  shouldBid(announcement: TaskAnnouncement): boolean {
    const hasCapabilities = announcement.requirements.capabilities.every(
      (c) => this.capabilities.includes(c),
    );
    if (!hasCapabilities) return false;
    if (this.currentTasks.length >= this.maxConcurrent) return false;
    return isBiddingOpen(announcement);
  }

  /** Generate a bid for a task. */
  generateBid(announcement: TaskAnnouncement): TaskBid {
    return {
      agentId: this.agentId,
      taskId: announcement.taskId,
      capabilityScore: this.calculateCapabilityScore(
        announcement.requirements,
      ),
      currentLoad: this.currentTasks.length / this.maxConcurrent,
      estimatedDurationMs: (announcement.requirements.complexity + 1) * 60_000,
      conditions: [],
      submittedAt: Date.now(),
    };
  }

  /** Fraction of the required capabilities this agent has (1.0 when nothing is required). */
  private calculateCapabilityScore(requirements: TaskRequirements): number {
    if (requirements.capabilities.length === 0) return 1.0;
    const matched =
      requirements.capabilities.filter((c) => this.capabilities.includes(c))
        .length;
    return matched / requirements.capabilities.length;
  }

  /** Accept a task. */
  acceptTask(taskId: string): void {
    this.currentTasks.push(taskId);
  }

  /** Complete a task. */
  completeTask(taskId: string): void {
    this.currentTasks = this.currentTasks.filter((t) => t !== taskId);
  }

  /** Get current task count. */
  currentTaskCount(): number {
    return this.currentTasks.length;
  }

  /** Get available capacity. */
  availableCapacity(): number {
    return this.maxConcurrent - this.currentTasks.length;
  }
}
