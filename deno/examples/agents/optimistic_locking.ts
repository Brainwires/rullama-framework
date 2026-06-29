// Example: Optimistic Concurrency with Conflict Detection
// Demonstrates how OptimisticController allows agents to proceed without
// upfront locks, detecting conflicts at commit time and resolving them
// via configurable strategies (FirstWriterWins, LastWriterWins, Retry).
// Run: deno run deno/examples/agents/optimistic_locking.ts

import {
  type CommitResult,
  isCommitSuccess,
  type OptimisticConflict,
  OptimisticController,
  type ResolutionStrategy,
} from "@rullama/agent";

async function main() {
  console.log("=== Optimistic Concurrency Demo ===\n");

  // 1. Create controller (default: FirstWriterWins)
  const controller = new OptimisticController();

  // 2. Agent-A begins an optimistic operation
  const tokenA = controller.beginOptimistic("agent-a", "src/main.rs");
  console.log(
    `Agent-A began optimistic operation on src/main.rs (base_version=${tokenA.baseVersion})`,
  );

  // 3. Agent-B begins on the same resource
  const tokenB = controller.beginOptimistic("agent-b", "src/main.rs");
  console.log(
    `Agent-B began optimistic operation on src/main.rs (base_version=${tokenB.baseVersion})`,
  );

  // 4. Agent-A commits successfully
  const versionA = controller.commitOptimistic(tokenA, "hash-after-a");
  console.log(`\nAgent-A committed successfully (new version=${versionA})`);

  // 5. Agent-B tries to commit -- conflict detected
  try {
    controller.commitOptimistic(tokenB, "hash-after-b");
    console.log("Unexpected: Agent-B committed without conflict");
  } catch (e) {
    const conflict = e as OptimisticConflict;
    console.log("\nAgent-B conflict detected:");
    console.log(`  Resource:          ${conflict.resourceId}`);
    console.log(`  Conflicting agent: ${conflict.conflictingAgent}`);
    console.log(`  Expected version:  ${conflict.expectedVersion}`);
    console.log(`  Actual version:    ${conflict.actualVersion}`);
    console.log(`  Holder agent:      ${conflict.holderAgent}`);
    console.log(
      `  Version diff:      ${
        conflict.actualVersion - conflict.expectedVersion
      }`,
    );
  }

  // 6. Demonstrate commitOrResolve with LastWriterWins strategy
  console.log("\n--- LastWriterWins strategy ---");

  const lwwStrategy: ResolutionStrategy = { kind: "last_writer_wins" };
  const lwwController = new OptimisticController(lwwStrategy);

  const tok1 = lwwController.beginOptimistic("agent-x", "config.json");
  const tok2 = lwwController.beginOptimistic("agent-y", "config.json");

  // Agent-X commits first
  lwwController.commitOptimistic(tok1, "hash-x");

  // Agent-Y uses commitOrResolve -- LastWriterWins lets it succeed
  const result: CommitResult = lwwController.commitOrResolve(tok2, "hash-y");

  if (result.kind === "committed") {
    console.log(
      `Agent-Y committed (version=${result.version}) -- last writer won`,
    );
  } else {
    console.log(`Unexpected result: ${result.kind}`);
  }
  console.log(`  is_success: ${isCommitSuccess(result)}`);

  // 7. Retry strategy
  console.log("\n--- Retry strategy ---");

  const retryStrategy: ResolutionStrategy = {
    kind: "retry",
    maxAttempts: 3,
  };
  const retryController = new OptimisticController(retryStrategy);

  const tokR1 = retryController.beginOptimistic("agent-r1", "data.db");
  const tokR2 = retryController.beginOptimistic("agent-r2", "data.db");

  retryController.commitOptimistic(tokR1, "hash-r1");

  const retryResult = retryController.commitOrResolve(tokR2, "hash-r2");

  if (retryResult.kind === "retry_needed") {
    console.log(
      `Agent-R2 told to retry (current_version=${retryResult.currentVersion})`,
    );
  } else {
    console.log(`Result: ${retryResult.kind}`);
  }

  // 8. Stats
  console.log("\n--- Controller stats ---");

  const stats = lwwController.getStats();
  console.log(`  Total resources tracked: ${stats.totalResources}`);
  console.log(`  Total conflicts:         ${stats.totalConflicts}`);
  console.log(`  Resolved by retry:       ${stats.resolvedByRetry}`);
  console.log(`  Escalated:               ${stats.escalated}`);

  console.log("\nOptimistic concurrency demo complete.");
}

await main();
