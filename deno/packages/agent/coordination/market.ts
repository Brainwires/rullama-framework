/**
 * Market-Based Resource Allocation with Priority Bidding.
 *
 * Implements market-based allocation where agents bid for resources with
 * dynamic urgency scores. Higher urgency = higher chance of getting
 * the resource.
 *
 * Key concepts:
 * - **ResourceBid**: Agent's bid with base priority and urgency multiplier
 * - **AgentBudget**: Budget management for fair allocation
 * - **MarketAllocator**: Manages auctions and allocations
 * - **PricingStrategy**: How prices are calculated (first-price, second-price, etc.)
 * - **UrgencyCalculator**: Dynamic priority based on context
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Bid submitted by an agent for a resource. */
export interface ResourceBid {
  /** Agent submitting the bid. */
  agentId: string;
  /** Resource being bid on. */
  resourceId: string;
  /** Base priority (0-10, static). */
  basePriority: number;
  /** Urgency multiplier (1.0 = normal, 2.0 = double urgency). */
  urgencyMultiplier: number;
  /** Maximum bid amount from budget. */
  maxBid: number;
  /** Reason for urgency (for logging/debugging). */
  urgencyReason: string;
  /** Estimated hold duration in milliseconds. */
  estimatedDurationMs: number;
  /** When the bid was submitted. */
  submittedAt: number;
}

/** Create a new bid with sensible defaults. */
export function createBid(
  agentId: string,
  resourceId: string,
): ResourceBid {
  return {
    agentId,
    resourceId,
    basePriority: 5,
    urgencyMultiplier: 1.0,
    maxBid: 10,
    urgencyReason: "",
    estimatedDurationMs: 60_000,
    submittedAt: Date.now(),
  };
}

/** Calculate the effective priority for a bid. */
export function effectivePriority(bid: ResourceBid): number {
  return bid.basePriority * bid.urgencyMultiplier;
}

/** Calculate a composite score for ranking bids. */
export function bidScore(bid: ResourceBid): number {
  const priorityFactor = effectivePriority(bid) / 10.0;
  const bidFactor = Math.min(bid.maxBid / 100.0, 1.0);
  return 0.7 * priorityFactor + 0.3 * bidFactor;
}

/** Agent's budget for bidding. */
export interface AgentBudget {
  /** Agent identifier. */
  agentId: string;
  /** Total budget points. */
  totalBudget: number;
  /** Currently available points. */
  available: number;
  /** Budget replenishment rate (points per second). */
  replenishRate: number;
  /** Last replenishment time (epoch ms). */
  lastReplenish: number;
}

/** Create a new budget. */
export function createBudget(
  agentId: string,
  totalBudget: number,
  replenishRate = 1.0,
): AgentBudget {
  return {
    agentId,
    totalBudget,
    available: totalBudget,
    replenishRate,
    lastReplenish: Date.now(),
  };
}

/** Replenish budget based on elapsed time. */
export function replenishBudget(budget: AgentBudget): void {
  const elapsedSecs = (Date.now() - budget.lastReplenish) / 1000;
  const replenished = Math.floor(elapsedSecs * budget.replenishRate);
  budget.available = Math.min(
    budget.available + replenished,
    budget.totalBudget,
  );
  budget.lastReplenish = Date.now();
}

/** Information about the current holder of a resource. */
export interface CurrentHolder {
  /** Agent that won the allocation. */
  agentId: string;
  /** When the resource was allocated (epoch ms). */
  acquiredAt: number;
  /** `acquiredAt + estimatedDurationMs` of the winning bid (epoch ms). */
  expectedRelease?: number;
}

/** A resource auction. */
interface ResourceAuction {
  resourceId: string;
  bids: ResourceBid[];
  currentHolder: CurrentHolder | null;
  auctionStart: number;
}

/** Strategy for calculating prices. */
export type PricingStrategy =
  | { kind: "first_price" }
  | { kind: "second_price" }
  | { kind: "fixed_price"; prices: Map<string, number> }
  | { kind: "dynamic"; basePrice: number; demandMultiplier: number }
  | { kind: "free" };

/** Default pricing strategy (second price). */
export function defaultPricingStrategy(): PricingStrategy {
  return { kind: "second_price" };
}

/** Result of an allocation attempt. */
export type AllocationResult =
  | { kind: "allocated"; agentId: string; price: number; position: number }
  | { kind: "no_bids" }
  | { kind: "still_held"; holder: string; remainingMs?: number }
  | {
    kind: "insufficient_budget";
    agentId: string;
    required: number;
    available: number;
  }
  | {
    kind: "outbid";
    agentId: string;
    winningAgent: string;
    winningScore: number;
  };

/** Check if an allocation was successful. */
export function isAllocated(
  r: AllocationResult,
): r is AllocationResult & { kind: "allocated" } {
  return r.kind === "allocated";
}

/** Record of an allocation for history. */
export interface AllocationRecord {
  /** Resource that was allocated. */
  resourceId: string;
  /** Agent that received it. */
  winner: string;
  /** Budget points charged to the winner. */
  price: number;
  /** Number of bids in the auction when it was decided. */
  competingBids: number;
  /** When the allocation happened (epoch ms). */
  allocatedAt: number;
}

/** Status of a specific resource's market. */
export interface MarketStatus {
  /** Resource the auction is for. */
  resourceId: string;
  /** Agent currently holding the resource, or null if free. */
  currentHolder: string | null;
  /** Bids waiting for the next `allocate` call. */
  pendingBids: number;
  /** `bidScore` of the first pending bid (sorted highest-first after an allocation), or null with no bids. */
  highestScore: number | null;
  /** Milliseconds since the auction was opened by its first bid. */
  auctionAgeMs: number;
}

/** Overall market statistics. */
export interface MarketStats {
  /** Auctions that have been opened and not garbage-collected (they are never removed). */
  activeAuctions: number;
  /** Sum of pending bids across all auctions. */
  totalPendingBids: number;
  /** Agents with a registered budget. */
  registeredAgents: number;
  /** Allocation records retained in history (bounded by `maxHistory`). */
  totalAllocations: number;
  /** Sum of prices over retained allocation records. */
  totalRevenue: number;
  /** `totalRevenue / totalAllocations`, or 0 with no history. */
  avgPrice: number;
  /** Mean `competingBids` over retained allocation records, or 0 with no history. */
  avgCompetition: number;
}

// ---------------------------------------------------------------------------
// Urgency calculator
// ---------------------------------------------------------------------------

/** Context for calculating urgency. */
export interface UrgencyContext {
  /** User is actively waiting for the result. */
  userWaiting: boolean;
  /** Deadline in epoch ms (if any). */
  deadline?: number;
  /** Operation is on the critical path. */
  criticalPath: boolean;
  /** Number of other resources currently held. */
  resourcesHeld: number;
  /** How long the agent has been waiting (ms). */
  waitTimeMs?: number;
}

/** Create a default urgency context. */
export function defaultUrgencyContext(): UrgencyContext {
  return { userWaiting: false, criticalPath: false, resourcesHeld: 0 };
}

/** Calculate urgency multiplier based on context. */
export function calculateUrgency(ctx: UrgencyContext): number {
  let multiplier = 1.0;

  if (ctx.userWaiting) multiplier *= 2.0;

  if (ctx.deadline != null) {
    const remainingMs = ctx.deadline - Date.now();
    if (remainingMs < 60_000) multiplier *= 3.0;
    else if (remainingMs < 300_000) multiplier *= 2.0;
    else if (remainingMs < 600_000) multiplier *= 1.5;
  }

  if (ctx.criticalPath) multiplier *= 1.5;

  multiplier *= 1.0 + ctx.resourcesHeld * 0.2;

  if (ctx.waitTimeMs != null && ctx.waitTimeMs > 60_000) {
    multiplier *= 1.0 + Math.min((ctx.waitTimeMs / 1000) / 120.0, 2.0);
  }

  return Math.min(multiplier, 10.0);
}

// ---------------------------------------------------------------------------
// MarketAllocator
// ---------------------------------------------------------------------------

/** Market-based resource allocator. */
export class MarketAllocator {
  private auctions = new Map<string, ResourceAuction>();
  private budgets = new Map<string, AgentBudget>();
  private allocationHistory: AllocationRecord[] = [];
  private pricing: PricingStrategy;
  private maxHistory: number;

  /**
   * Create an allocator with no agents or auctions.
   * @param pricing How the winner's price is computed; defaults to second-price.
   * @param maxHistory Maximum allocation records retained (oldest dropped first; default 1000).
   */
  constructor(pricing?: PricingStrategy, maxHistory = 1000) {
    this.pricing = pricing ?? defaultPricingStrategy();
    this.maxHistory = maxHistory;
  }

  /** Register an agent with a budget. */
  registerAgent(
    agentId: string,
    totalBudget: number,
    replenishRate = 1.0,
  ): void {
    this.budgets.set(
      agentId,
      createBudget(agentId, totalBudget, replenishRate),
    );
  }

  /** Get an agent's current budget (replenishes first). */
  getBudget(agentId: string): AgentBudget | undefined {
    const budget = this.budgets.get(agentId);
    if (budget) replenishBudget(budget);
    return budget ? { ...budget } : undefined;
  }

  /** Submit a bid for a resource. */
  submitBid(bid: ResourceBid): void {
    const budget = this.budgets.get(bid.agentId);
    if (!budget) throw new Error("Agent not registered");

    replenishBudget(budget);
    if (budget.available < bid.maxBid) {
      throw new Error(
        `Insufficient budget: have ${budget.available}, need ${bid.maxBid}`,
      );
    }

    let auction = this.auctions.get(bid.resourceId);
    if (!auction) {
      auction = {
        resourceId: bid.resourceId,
        bids: [],
        currentHolder: null,
        auctionStart: Date.now(),
      };
      this.auctions.set(bid.resourceId, auction);
    }

    // Remove existing bid from same agent
    auction.bids = auction.bids.filter((b) => b.agentId !== bid.agentId);
    auction.bids.push(bid);
  }

  /** Cancel a bid. Returns true if a bid was removed. */
  cancelBid(agentId: string, resourceId: string): boolean {
    const auction = this.auctions.get(resourceId);
    if (!auction) return false;
    const before = auction.bids.length;
    auction.bids = auction.bids.filter((b) => b.agentId !== agentId);
    return auction.bids.length < before;
  }

  /** Allocate resource to highest bidder. */
  allocate(resourceId: string): AllocationResult {
    const auction = this.auctions.get(resourceId);
    if (!auction) return { kind: "no_bids" };

    if (auction.currentHolder) {
      const remaining = auction.currentHolder.expectedRelease
        ? Math.max(0, auction.currentHolder.expectedRelease - Date.now())
        : undefined;
      return {
        kind: "still_held",
        holder: auction.currentHolder.agentId,
        remainingMs: remaining,
      };
    }

    if (auction.bids.length === 0) return { kind: "no_bids" };

    // Sort by score (highest first)
    auction.bids.sort((a, b) => bidScore(b) - bidScore(a));

    const price = this.calculatePrice(auction.bids);

    // Try to charge the winner
    for (let position = 0; position < auction.bids.length; position++) {
      const bid = auction.bids[position];
      const budget = this.budgets.get(bid.agentId);
      if (budget) {
        replenishBudget(budget);
        if (budget.available >= price) {
          budget.available -= price;

          auction.currentHolder = {
            agentId: bid.agentId,
            acquiredAt: Date.now(),
            expectedRelease: Date.now() + bid.estimatedDurationMs,
          };

          const winnerId = bid.agentId;
          const competingBids = auction.bids.length;
          auction.bids = [];

          this.recordAllocation(resourceId, winnerId, price, competingBids);

          return { kind: "allocated", agentId: winnerId, price, position };
        }
      }
    }

    // No one could afford the price
    const firstBid = auction.bids[0];
    return {
      kind: "insufficient_budget",
      agentId: firstBid.agentId,
      required: price,
      available: this.budgets.get(firstBid.agentId)?.available ?? 0,
    };
  }

  /** Release a resource (current holder done). */
  release(resourceId: string, agentId: string): boolean {
    const auction = this.auctions.get(resourceId);
    if (
      auction?.currentHolder &&
      auction.currentHolder.agentId === agentId
    ) {
      auction.currentHolder = null;
      return true;
    }
    return false;
  }

  /** Get current market status for a resource. */
  marketStatus(resourceId: string): MarketStatus | undefined {
    const auction = this.auctions.get(resourceId);
    if (!auction) return undefined;
    return {
      resourceId,
      currentHolder: auction.currentHolder?.agentId ?? null,
      pendingBids: auction.bids.length,
      highestScore: auction.bids.length > 0 ? bidScore(auction.bids[0]) : null,
      auctionAgeMs: Date.now() - auction.auctionStart,
    };
  }

  /** List all active auctions. */
  listAuctions(): MarketStatus[] {
    const out: MarketStatus[] = [];
    for (const [resourceId, auction] of this.auctions) {
      out.push({
        resourceId,
        currentHolder: auction.currentHolder?.agentId ?? null,
        pendingBids: auction.bids.length,
        highestScore: auction.bids.length > 0
          ? bidScore(auction.bids[0])
          : null,
        auctionAgeMs: Date.now() - auction.auctionStart,
      });
    }
    return out;
  }

  /** Get allocation history. */
  getHistory(): AllocationRecord[] {
    return [...this.allocationHistory];
  }

  /** Get market statistics. */
  getStats(): MarketStats {
    const totalAllocations = this.allocationHistory.length;
    const totalRevenue = this.allocationHistory.reduce(
      (s, r) => s + r.price,
      0,
    );
    const avgPrice = totalAllocations > 0 ? totalRevenue / totalAllocations : 0;
    const avgCompetition = totalAllocations > 0
      ? this.allocationHistory.reduce((s, r) => s + r.competingBids, 0) /
        totalAllocations
      : 0;

    let totalPendingBids = 0;
    for (const a of this.auctions.values()) totalPendingBids += a.bids.length;

    return {
      activeAuctions: this.auctions.size,
      totalPendingBids,
      registeredAgents: this.budgets.size,
      totalAllocations,
      totalRevenue,
      avgPrice,
      avgCompetition,
    };
  }

  // ── Private ─────────────────────────────────────────────────────────────

  /**
   * Price for the (already score-sorted) bid list under the configured
   * strategy: top bid's `maxBid`; the lower of the top two (or 1 with a single
   * bid); a fixed per-resource price (1 if unlisted); `basePrice * (1 +
   * bids * demandMultiplier)` floored; or 0 when free.
   */
  private calculatePrice(bids: ResourceBid[]): number {
    switch (this.pricing.kind) {
      case "first_price":
        return bids[0]?.maxBid ?? 0;
      case "second_price":
        if (bids.length >= 2) {
          return Math.min(bids[1].maxBid, bids[0].maxBid);
        }
        return 1; // Minimum price
      case "fixed_price": {
        const resourceId = bids[0]?.resourceId;
        return this.pricing.prices.get(resourceId ?? "") ?? 1;
      }
      case "dynamic": {
        const demand = bids.length;
        return Math.floor(
          this.pricing.basePrice *
            (1.0 + demand * this.pricing.demandMultiplier),
        );
      }
      case "free":
        return 0;
    }
  }

  /** Append an `AllocationRecord`, trimming the oldest entries beyond `maxHistory`. */
  private recordAllocation(
    resourceId: string,
    winner: string,
    price: number,
    competingBids: number,
  ): void {
    this.allocationHistory.push({
      resourceId,
      winner,
      price,
      competingBids,
      allocatedAt: Date.now(),
    });
    while (this.allocationHistory.length > this.maxHistory) {
      this.allocationHistory.shift();
    }
  }
}
