// Example: Three-State Model for Comprehensive State Tracking
// Demonstrates how ThreeStateModel separates application, operation, and
// dependency state to enable conflict detection, operation validation,
// and state snapshots for rollback support.
// Run: deno run deno/examples/agents/three_state.ts

import {
  type ApplicationChange,
  createOperationLog,
  type OperationLog,
  type ProposedOperation,
  type StateChange,
  ThreeStateModel,
} from "@rullama/agent";

async function main() {
  console.log("=== Three-State Model Demo ===\n");

  // 1. Create model
  const model = new ThreeStateModel();

  // 2. Register resources in application state
  console.log("--- Updating application state ---");

  model.applicationState.updateFile("src/main.rs", "abc123");
  model.applicationState.updateFile("src/lib.rs", "def456");
  model.applicationState.markResourceExists("build-cache");

  const files = model.applicationState.getAllFiles();
  console.log(`Tracked files: ${files.size}`);
  for (const [path, status] of files) {
    console.log(
      `  ${path} (hash=${status.contentHash}, dirty=${status.dirty})`,
    );
  }

  // 3. Start and complete an operation
  console.log("\n--- Operation lifecycle ---");

  const opId = model.operationState.generateId();
  const log: OperationLog = {
    ...createOperationLog(opId, "agent-1", "build", { target: "release" }),
    resourcesNeeded: ["src/main.rs"],
    resourcesProduced: ["target/release/app"],
  };

  model.operationState.startOperation(log);
  console.log(`Started operation: ${opId}`);

  const active = model.operationState.getActiveOperations();
  console.log(`Active operations: ${active.length}`);

  model.operationState.completeOperation(opId, true);

  const completed = model.operationState.getOperation(opId)!;
  console.log(`Operation ${opId} status: ${completed.status}`);

  // 4. Validate a proposed operation -- should pass
  console.log("\n--- Operation validation (no conflict) ---");

  const proposed: ProposedOperation = {
    agentId: "agent-2",
    operationType: "test",
    resourcesNeeded: ["src/lib.rs"],
    resourcesProduced: ["test-report.xml"],
  };

  const result = model.validateOperation(proposed);
  console.log(
    `Proposed test on src/lib.rs: valid=${result.valid}, errors=${result.errors.length}, warnings=${result.warnings.length}`,
  );

  // 5. Show a conflicting scenario
  console.log("\n--- Operation validation (conflict) ---");

  // Start a long-running operation that holds src/main.rs
  const conflictOpId = model.operationState.generateId();
  const conflictLog: OperationLog = {
    ...createOperationLog(conflictOpId, "agent-3", "refactor", {}),
    resourcesNeeded: ["src/main.rs"],
    resourcesProduced: [],
  };
  model.operationState.startOperation(conflictLog);

  // Another agent tries to use the same resource
  const conflicting: ProposedOperation = {
    agentId: "agent-4",
    operationType: "format",
    resourcesNeeded: ["src/main.rs"],
    resourcesProduced: [],
  };

  const conflictResult = model.validateOperation(conflicting);
  console.log(
    `Proposed format on src/main.rs while refactor is running: valid=${conflictResult.valid}`,
  );
  for (const err of conflictResult.errors) {
    console.log(`  Error: ${err}`);
  }

  // Clean up the running operation
  model.operationState.completeOperation(conflictOpId, true);

  // 6. Record a state change and take snapshot
  console.log("\n--- State change + snapshot ---");

  const change: StateChange = {
    operationId: "op-change-1",
    applicationChanges: [
      {
        kind: "file_modified",
        path: "src/main.rs",
        newHash: "updated-hash-789",
      },
      { kind: "resource_created", resourceId: "deploy-artifact" },
    ] as ApplicationChange[],
    newDependencies: [],
  };
  model.recordStateChange(change);

  const snapshot = model.snapshot();
  console.log("Snapshot summary:");
  console.log(`  Files tracked:      ${snapshot.files.size}`);
  console.log(`  Resource locks:     ${snapshot.locks.size}`);
  console.log(`  Active operations:  ${snapshot.activeOperations.length}`);
  console.log(
    `  Git branch:         ${snapshot.gitState.currentBranch || "(none)"}`,
  );

  // Verify the updated hash
  const mainRs = snapshot.files.get("src/main.rs")!;
  console.log(
    `  src/main.rs hash:   ${mainRs.contentHash} (dirty=${mainRs.dirty})`,
  );

  console.log("\nThree-state model demo complete.");
}

await main();
