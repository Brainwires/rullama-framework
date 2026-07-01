// Example: Agent Pool Management
// Demonstrates AgentPool for managing multiple concurrent TaskAgents,
// monitoring status, and collecting results. Uses AgentPoolStats
// to track pool health. Also shows CommunicationHub and FileLockManager.
// Run: deno run deno/examples/agents/agent_pool.ts

import {
  type AgentPoolStats,
  CommunicationHub,
  FileLockManager,
} from "@rullama/agent";
import { Task } from "@rullama/core";

async function main() {
  console.log("=== Agent Pool Demo ===\n");

  // NOTE: AgentPool requires a Provider and AgentContext to actually spawn
  // agents. This example demonstrates the pool stats, communication hub,
  // and file lock manager APIs that don't require a live provider.

  // 1. Demonstrate CommunicationHub
  console.log("--- 1. Communication Hub ---");

  const hub = new CommunicationHub();
  hub.registerAgent("agent-1");
  hub.registerAgent("agent-2");
  hub.registerAgent("listener-1");

  console.log(`  Registered agents: ${hub.agentCount()}`);

  // Send a direct message
  hub.sendMessage("agent-1", "agent-2", {
    type: "status_update",
    agentId: "agent-1",
    status: "working",
    details: "Implementing auth module",
  });

  // Broadcast to all
  hub.broadcast("orchestrator", {
    type: "status_update",
    agentId: "orchestrator",
    status: "planning",
    details: "Starting cycle 1",
  });

  // Check messages (receiveMessage returns a Promise)
  const msgPromise = hub.receiveMessage("agent-2");
  if (msgPromise) {
    const msg = await msgPromise;
    console.log(`  agent-2 received: ${msg.message.type} from ${msg.from}`);
  }

  const broadcastPromise = hub.receiveMessage("listener-1");
  if (broadcastPromise) {
    const broadcastMsg = await broadcastPromise;
    console.log(
      `  listener-1 received broadcast: ${broadcastMsg.message.type} from ${broadcastMsg.from}`,
    );
  }

  // 2. Demonstrate FileLockManager
  console.log("\n--- 2. File Lock Manager ---");

  const lockManager = new FileLockManager();

  // Multiple concurrent reads are allowed
  const read1 = lockManager.acquireLock("agent-1", "src/lib.rs", "read");
  const read2 = lockManager.acquireLock("agent-2", "src/lib.rs", "read");
  console.log("  Two agents reading src/lib.rs concurrently - OK");

  // Release reads via the guard
  read1.release();
  read2.release();

  // Exclusive write lock
  const writeGuard = lockManager.acquireLock("agent-1", "src/lib.rs", "write");
  console.log("  agent-1 has exclusive write access to src/lib.rs - OK");

  // Attempt conflicting read while write is held
  let conflictDetected = false;
  try {
    lockManager.acquireLock("agent-2", "src/lib.rs", "read");
  } catch {
    conflictDetected = true;
  }
  console.log(
    `  agent-2 read while write held: ${
      conflictDetected ? "BLOCKED (expected)" : "OK"
    }`,
  );

  writeGuard.release();

  // Lock stats
  const lockStats = lockManager.stats();
  console.log("\n  Lock stats:");
  console.log(`    Files with locks: ${lockStats.totalFiles}`);
  console.log(`    Write locks: ${lockStats.totalWriteLocks}`);
  console.log(`    Read locks: ${lockStats.totalReadLocks}`);

  // 3. Demonstrate Task creation for pool usage
  console.log("\n--- 3. Task Creation ---");

  const tasks = [
    new Task("task-1", "Implement authentication module"),
    new Task("task-2", "Add unit tests for parser"),
    new Task("task-3", "Refactor error handling"),
  ];

  for (const task of tasks) {
    console.log(
      `  Created: ${task.id} - "${task.description}" [${task.status}]`,
    );
  }

  // 4. Show what AgentPoolStats looks like
  console.log("\n--- 4. Pool Stats Structure ---");

  const mockStats: AgentPoolStats = {
    maxAgents: 5,
    totalAgents: 3,
    running: 2,
    completed: 1,
    failed: 0,
  };

  console.log(`  Max agents:   ${mockStats.maxAgents}`);
  console.log(`  Total agents: ${mockStats.totalAgents}`);
  console.log(`  Running:      ${mockStats.running}`);
  console.log(`  Completed:    ${mockStats.completed}`);
  console.log(`  Failed:       ${mockStats.failed}`);

  // 5. Hub message types showcase
  console.log("\n--- 5. Hub Message Types ---");

  hub.registerAgent("worker-1");

  hub.sendMessage("orchestrator", "worker-1", {
    type: "task_request",
    taskId: "task-1",
    description: "Implement authentication module",
    priority: 3,
  });

  hub.sendMessage("worker-1", "agent-1", {
    type: "task_result",
    taskId: "task-1",
    success: true,
    result: "Auth module implemented with JWT support",
  });

  hub.sendMessage("agent-1", "agent-2", {
    type: "help_request",
    requestId: "help-1",
    topic: "error handling",
    details: "Need advice on error propagation pattern",
  });

  const workerMsg = hub.tryReceiveMessage("worker-1");
  if (workerMsg) {
    console.log(`  worker-1 received: ${workerMsg.message.type}`);
  }

  const agent1Msg = hub.tryReceiveMessage("agent-1");
  if (agent1Msg) {
    console.log(`  agent-1 received: ${agent1Msg.message.type}`);
  }

  const agent2Msg = hub.tryReceiveMessage("agent-2");
  if (agent2Msg) {
    console.log(`  agent-2 received: ${agent2Msg.message.type}`);
  }

  console.log(`\n  Registered agents: [${hub.listAgents().join(", ")}]`);

  console.log("\nAgent pool demo complete.");
}

await main();
