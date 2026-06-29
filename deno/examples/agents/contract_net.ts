// Example: Contract-Net Bidding Protocol
// Shows how to create task announcements with requirements, simulate agents
// submitting bids with different profiles, score bids, and evaluate winners
// under different BidEvaluationStrategy variants.
// Run: deno run deno/examples/agents/contract_net.ts

import {
  type BidEvaluationStrategy,
  bidScore,
  ContractNetManager,
  defaultTaskRequirements,
  isBiddingOpen,
  type TaskAnnouncement,
  type TaskBid,
} from "@rullama/agent";

async function main() {
  console.log("=== Contract-Net Bidding Protocol ===\n");

  // 1. Create a task announcement with requirements
  const requirements = {
    ...defaultTaskRequirements(),
    capabilities: ["rust", "testing"],
    complexity: 5,
    priority: 3,
  };

  const announcement: TaskAnnouncement = {
    taskId: "task-refactor",
    description: "Refactor error handling module with full test coverage",
    requirements,
    bidDeadline: Date.now() + 30_000,
    announcer: "orchestrator",
    announcedAt: Date.now(),
  };

  console.log(`Task announced: ${announcement.description}`);
  console.log(
    `  Requirements: [${
      announcement.requirements.capabilities.join(", ")
    }], complexity=${announcement.requirements.complexity}`,
  );
  console.log(`  Bidding open: ${isBiddingOpen(announcement)}\n`);

  // 2. Simulate three agents submitting bids
  const bidAlpha: TaskBid = {
    agentId: "agent-alpha",
    taskId: "task-refactor",
    capabilityScore: 0.95,
    currentLoad: 0.6,
    estimatedDurationMs: 300_000,
    conditions: [],
    submittedAt: Date.now(),
  };

  const bidBeta: TaskBid = {
    agentId: "agent-beta",
    taskId: "task-refactor",
    capabilityScore: 0.70,
    currentLoad: 0.1,
    estimatedDurationMs: 180_000,
    conditions: [],
    submittedAt: Date.now(),
  };

  const bidGamma: TaskBid = {
    agentId: "agent-gamma",
    taskId: "task-refactor",
    capabilityScore: 0.85,
    currentLoad: 0.3,
    estimatedDurationMs: 240_000,
    conditions: [],
    submittedAt: Date.now(),
  };

  const bids = [bidAlpha, bidBeta, bidGamma];

  // 3. Show bid scores (default weighted: 40% capability, 30% availability, 30% speed)
  console.log("--- Bid Scores (default weights) ---");
  for (const bid of bids) {
    console.log(
      `  ${bid.agentId}: score=${bidScore(bid).toFixed(3)}  (capability=${
        bid.capabilityScore.toFixed(2)
      }, load=${bid.currentLoad.toFixed(2)}, duration=${
        bid.estimatedDurationMs / 1000
      }s)`,
    );
  }
  console.log();

  // 4. Evaluate winners under different strategies
  const strategies: Array<[string, BidEvaluationStrategy]> = [
    ["HighestScore", { kind: "highest_score" }],
    ["FastestCompletion", { kind: "fastest_completion" }],
    ["LoadBalancing", { kind: "load_balancing" }],
    ["BestCapability", { kind: "best_capability" }],
  ];

  console.log("--- Winners by Strategy ---");
  for (const [label, strategy] of strategies) {
    const manager = new ContractNetManager(strategy);

    const taskAnnouncement: TaskAnnouncement = {
      taskId: "",
      description: "Refactor error handling",
      requirements: defaultTaskRequirements(),
      bidDeadline: Date.now() + 30_000,
      announcer: "orchestrator",
      announcedAt: Date.now(),
    };

    const taskId = manager.announceTask(taskAnnouncement);

    for (const bid of bids) {
      manager.receiveBid({
        ...bid,
        taskId,
        submittedAt: Date.now(),
      });
    }

    const winner = manager.awardTask(taskId);
    console.log(
      `  ${label.padEnd(20)} => winner: ${winner ?? "none"}`,
    );
  }
  console.log();

  // 5. Full lifecycle with the default manager
  console.log("--- Full Lifecycle Demo ---");
  const manager = new ContractNetManager();

  const lifecycleAnnouncement: TaskAnnouncement = {
    taskId: "lifecycle-task",
    description: "Build auth module",
    requirements: defaultTaskRequirements(),
    bidDeadline: Date.now() + 30_000,
    announcer: "orchestrator",
    announcedAt: Date.now(),
  };

  const taskId = manager.announceTask(lifecycleAnnouncement);

  manager.receiveBid({
    agentId: "agent-alpha",
    taskId,
    capabilityScore: 0.9,
    currentLoad: 0.2,
    estimatedDurationMs: 60_000,
    conditions: [],
    submittedAt: Date.now(),
  });

  const winner = manager.awardTask(taskId)!;
  console.log(`  Awarded to: ${winner}`);

  manager.acceptAward(taskId, winner);
  console.log(`  Status after accept: awarded (accepted)`);

  manager.completeTask(taskId, winner, true, "Auth module ready");
  console.log(`  Status after complete: completed`);

  console.log("\nContract-Net demo complete.");
}

await main();
