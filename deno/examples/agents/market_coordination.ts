// Example: Market-Based Resource Allocation
// Demonstrates the MarketAllocator for resource bidding with dynamic urgency,
// agent budgets, pricing strategies, and allocation tracking.
// Run: deno run deno/examples/agents/market_coordination.ts

import {
  calculateUrgency,
  createBid,
  defaultUrgencyContext,
  effectivePriority,
  isAllocated,
  MarketAllocator,
  marketBidScore,
  type ResourceBid,
  type UrgencyContext,
} from "@rullama/agent";

async function main() {
  console.log("=== Market-Based Resource Allocation ===\n");

  // 1. Create a market allocator with second-price auction
  const market = new MarketAllocator({ kind: "second_price" });

  // 2. Register agents with budgets
  console.log("--- 1. Register Agents ---");
  market.registerAgent("agent-alpha", 100, 1.0);
  market.registerAgent("agent-beta", 80, 0.5);
  market.registerAgent("agent-gamma", 50, 2.0);

  for (const agentId of ["agent-alpha", "agent-beta", "agent-gamma"]) {
    const budget = market.getBudget(agentId)!;
    console.log(
      `  ${agentId}: budget=${budget.totalBudget}, available=${budget.available}, replenish=${budget.replenishRate}/s`,
    );
  }

  // 3. Calculate urgency for different contexts
  console.log("\n--- 2. Urgency Calculation ---");

  const contexts: Array<[string, UrgencyContext]> = [
    ["Normal", defaultUrgencyContext()],
    ["User waiting", { ...defaultUrgencyContext(), userWaiting: true }],
    [
      "Critical path + deadline",
      {
        ...defaultUrgencyContext(),
        criticalPath: true,
        deadline: Date.now() + 120_000,
      },
    ],
    [
      "Holding resources + long wait",
      {
        ...defaultUrgencyContext(),
        resourcesHeld: 3,
        waitTimeMs: 90_000,
      },
    ],
  ];

  for (const [label, ctx] of contexts) {
    const urgency = calculateUrgency(ctx);
    console.log(`  ${label}: urgency multiplier = ${urgency.toFixed(2)}`);
  }

  // 4. Submit bids for a resource
  console.log("\n--- 3. Submit Bids ---");

  const bids: ResourceBid[] = [
    {
      ...createBid("agent-alpha", "src/main.rs"),
      basePriority: 8,
      urgencyMultiplier: 1.5,
      maxBid: 20,
      urgencyReason: "Critical fix",
    },
    {
      ...createBid("agent-beta", "src/main.rs"),
      basePriority: 5,
      urgencyMultiplier: 1.0,
      maxBid: 15,
      urgencyReason: "Routine update",
    },
    {
      ...createBid("agent-gamma", "src/main.rs"),
      basePriority: 7,
      urgencyMultiplier: 2.0,
      maxBid: 25,
      urgencyReason: "User waiting",
    },
  ];

  for (const bid of bids) {
    console.log(
      `  ${bid.agentId}: priority=${bid.basePriority}, urgency=${bid.urgencyMultiplier}x, effective=${
        effectivePriority(bid).toFixed(1)
      }, score=${marketBidScore(bid).toFixed(3)}, maxBid=${bid.maxBid}`,
    );
    market.submitBid(bid);
  }

  // 5. Run the auction
  console.log("\n--- 4. Allocate Resource ---");

  const allocationResult = market.allocate("src/main.rs");
  if (isAllocated(allocationResult)) {
    console.log(`  Winner: ${allocationResult.agentId}`);
    console.log(`  Price paid: ${allocationResult.price}`);
    console.log(`  Position: ${allocationResult.position}`);
  } else {
    console.log(`  Allocation result: ${allocationResult.kind}`);
  }

  // 6. Check market status
  console.log("\n--- 5. Market Status ---");

  const status = market.marketStatus("src/main.rs");
  if (status) {
    console.log(`  Resource: ${status.resourceId}`);
    console.log(`  Current holder: ${status.currentHolder ?? "none"}`);
    console.log(`  Pending bids: ${status.pendingBids}`);
    console.log(`  Auction age: ${status.auctionAgeMs}ms`);
  }

  // 7. Second resource with competing bids
  console.log("\n--- 6. Second Auction (config.json) ---");

  market.submitBid({
    ...createBid("agent-alpha", "config.json"),
    basePriority: 3,
    maxBid: 10,
    urgencyReason: "Config update",
  });
  market.submitBid({
    ...createBid("agent-beta", "config.json"),
    basePriority: 6,
    maxBid: 12,
    urgencyReason: "Schema migration",
  });

  const configResult = market.allocate("config.json");
  if (isAllocated(configResult)) {
    console.log(
      `  Winner: ${configResult.agentId}, price: ${configResult.price}`,
    );
  }

  // 8. Release and re-allocate
  console.log("\n--- 7. Release and Stats ---");

  if (isAllocated(allocationResult)) {
    market.release("src/main.rs", allocationResult.agentId);
    console.log(`  Released src/main.rs by ${allocationResult.agentId}`);
  }

  const stats = market.getStats();
  console.log(`\n  Active auctions: ${stats.activeAuctions}`);
  console.log(`  Total pending bids: ${stats.totalPendingBids}`);
  console.log(`  Registered agents: ${stats.registeredAgents}`);
  console.log(`  Total allocations: ${stats.totalAllocations}`);
  console.log(`  Total revenue: ${stats.totalRevenue}`);
  console.log(`  Avg price: ${stats.avgPrice.toFixed(2)}`);
  console.log(`  Avg competition: ${stats.avgCompetition.toFixed(2)}`);

  // 9. Check remaining budgets
  console.log("\n--- 8. Remaining Budgets ---");
  for (const agentId of ["agent-alpha", "agent-beta", "agent-gamma"]) {
    const budget = market.getBudget(agentId)!;
    console.log(
      `  ${agentId}: available=${budget.available}/${budget.totalBudget}`,
    );
  }

  console.log("\nMarket coordination demo complete.");
}

await main();
