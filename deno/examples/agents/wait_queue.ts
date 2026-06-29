// Example: Wait Queue for Resource Coordination
// Demonstrates the WaitQueue for managing agents waiting for locked resources
// with priority ordering, event subscriptions, and estimated wait times.
// Run: deno run deno/examples/agents/wait_queue.ts

import {
  fileResourceKey,
  resourceKey,
  WaitQueue,
  type WaitQueueEvent,
} from "@rullama/agent";

async function main() {
  console.log("=== Wait Queue Demo ===\n");

  // 1. Create a wait queue
  const queue = new WaitQueue();

  // 2. Subscribe to events
  console.log("--- 1. Event Subscription ---");
  const events: WaitQueueEvent[] = [];
  const unsub = queue.subscribe((event: WaitQueueEvent) => {
    events.push(event);
  });

  // 3. Register agents waiting for the same file
  console.log("\n--- 2. Register Waiters ---");

  const fileKey = fileResourceKey("src/main.rs");
  console.log(`Resource key: ${fileKey}`);

  const handle1 = queue.register(fileKey, "agent-1", 5, true);
  console.log(
    `  agent-1 registered at position ${handle1.initialPosition} (priority=5)`,
  );

  const handle2 = queue.register(fileKey, "agent-2", 3, false);
  console.log(
    `  agent-2 registered at position ${handle2.initialPosition} (priority=3, higher)`,
  );

  const handle3 = queue.register(fileKey, "agent-3", 7, true);
  console.log(
    `  agent-3 registered at position ${handle3.initialPosition} (priority=7, lower)`,
  );

  // 4. Check queue status
  console.log("\n--- 3. Queue Status ---");
  console.log(`  Queue length: ${queue.queueLength(fileKey)}`);
  console.log(`  agent-1 position: ${queue.position(fileKey, "agent-1")}`);
  console.log(`  agent-2 position: ${queue.position(fileKey, "agent-2")}`);
  console.log(`  agent-3 position: ${queue.position(fileKey, "agent-3")}`);

  const status = queue.getQueueStatus(fileKey)!;
  console.log(`\n  Detailed status:`);
  console.log(`    Resource: ${status.resourceKey}`);
  console.log(`    Length: ${status.queueLength}`);
  for (const waiter of status.waiters) {
    console.log(
      `    [${waiter.position}] ${waiter.agentId} (priority=${waiter.priority}, autoAcquire=${waiter.autoAcquire})`,
    );
  }

  // 5. Peek at the next waiter
  console.log("\n--- 4. Peek Next ---");
  const next = queue.peekNext(fileKey);
  if (next) {
    console.log(
      `  Next: ${next.agentId} (priority=${next.priority}, autoAcquire=${next.autoAcquire})`,
    );
  }

  // 6. Simulate resource release -- notify front of queue
  console.log("\n--- 5. Resource Release ---");

  // Seed some history for wait estimation
  queue.recordWaitTime(fileKey, 1500);
  queue.recordWaitTime(fileKey, 2000);
  queue.recordWaitTime(fileKey, 1800);

  const estimatedWait = queue.estimateWait(fileKey);
  console.log(
    `  Estimated wait time: ${
      estimatedWait != null ? estimatedWait.toFixed(0) + "ms" : "unknown"
    }`,
  );

  const released = queue.notifyReleased(fileKey);
  console.log(`  Resource released, next waiter: ${released}`);
  console.log(`  Queue length after release: ${queue.queueLength(fileKey)}`);

  // 7. Cancel a waiter
  console.log("\n--- 6. Cancel Waiter ---");
  const cancelled = handle3.cancel();
  console.log(`  agent-3 cancelled: ${cancelled}`);
  console.log(`  Queue length after cancel: ${queue.queueLength(fileKey)}`);
  console.log(`  agent-3 is waiting: ${queue.isWaiting("agent-3")}`);

  // 8. Show different resource key helpers
  console.log("\n--- 7. Resource Key Helpers ---");
  console.log(`  File key: ${fileResourceKey("src/lib.rs")}`);
  console.log(`  Build key: ${resourceKey("build", "release")}`);
  console.log(`  Test key: ${resourceKey("test", "unit")}`);

  // 9. Multi-resource waiting
  console.log("\n--- 8. Multi-Resource Waiting ---");
  const buildKey = resourceKey("build", "project");
  queue.register(buildKey, "agent-1", 5, true);
  const waitingFor = queue.waitingFor("agent-1");
  console.log(`  agent-1 is waiting for: [${waitingFor.join(", ")}]`);

  // 10. List all active queues
  console.log("\n--- 9. Active Queues ---");
  const activeQueues = queue.listQueues();
  for (const q of activeQueues) {
    console.log(`  ${q}: ${queue.queueLength(q)} waiters`);
  }

  // 11. Summary of events collected
  console.log("\n--- 10. Events Summary ---");
  const eventTypes = new Map<string, number>();
  for (const event of events) {
    const count = eventTypes.get(event.type) ?? 0;
    eventTypes.set(event.type, count + 1);
  }
  for (const [type, count] of eventTypes) {
    console.log(`  ${type}: ${count}`);
  }

  unsub();

  console.log("\nWait queue demo complete.");
}

await main();
