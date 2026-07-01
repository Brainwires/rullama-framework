// Example: Saga Compensating Transactions
// Shows how to use SagaExecutor with CompensableOperation implementations
// to execute a sequence of operations where a mid-sequence failure triggers
// automatic compensation (rollback) of all previously completed steps.
// Run: deno run deno/examples/agents/saga_compensation.ts

import {
  type CompensableOperation,
  failureResult,
  type OperationResult,
  SagaExecutor,
  type SagaOperationType,
  successResult,
} from "@rullama/agent";

// -- Mock Operations --

/** A file-write operation that always succeeds. */
const mockFileWrite: CompensableOperation = {
  async execute(): Promise<OperationResult> {
    console.log("    [execute] Writing config.toml ...");
    return {
      ...successResult("mock-file-write"),
      output: "wrote 42 bytes to config.toml",
    };
  },
  async compensate(_result: OperationResult): Promise<void> {
    console.log("    [compensate] Restoring original config.toml");
  },
  description(): string {
    return "Write file: config.toml";
  },
  operationType(): SagaOperationType {
    return "file_write";
  },
};

/** A git-stage operation that always succeeds. */
const mockGitStage: CompensableOperation = {
  async execute(): Promise<OperationResult> {
    console.log("    [execute] Staging config.toml ...");
    return {
      ...successResult("mock-git-stage"),
      output: "staged 1 file",
    };
  },
  async compensate(_result: OperationResult): Promise<void> {
    console.log("    [compensate] Unstaging config.toml (git reset HEAD)");
  },
  description(): string {
    return "Git stage: config.toml";
  },
  operationType(): SagaOperationType {
    return "git_stage";
  },
};

/** A build operation that always fails. */
const mockBuild: CompensableOperation = {
  async execute(): Promise<OperationResult> {
    console.log("    [execute] Running cargo build ... FAILED");
    return failureResult("mock-build");
  },
  async compensate(_result: OperationResult): Promise<void> {
    // Build is non-compensable, but we implement it anyway (will be skipped).
  },
  description(): string {
    return "Build project";
  },
  operationType(): SagaOperationType {
    return "build";
  },
};

// -- Main --

async function main() {
  console.log("=== Saga Compensating Transactions ===\n");

  // 1. Create a saga executor
  const saga = new SagaExecutor("agent-1", "edit-stage-build pipeline");
  console.log(`Saga created: ${saga.sagaId}`);
  console.log(`  Agent:       ${saga.agentId}`);
  console.log(`  Description: ${saga.description}`);
  console.log(`  Status:      ${saga.status}\n`);

  // 2. Execute step 1: file write (succeeds)
  console.log("Step 1: File write");
  const result1 = await saga.executeStep(mockFileWrite);
  console.log(`  Success: ${result1.success}`);
  console.log(`  Operations so far: ${saga.operationCount()}\n`);

  // 3. Execute step 2: git stage (succeeds)
  console.log("Step 2: Git stage");
  const result2 = await saga.executeStep(mockGitStage);
  console.log(`  Success: ${result2.success}`);
  console.log(`  Operations so far: ${saga.operationCount()}\n`);

  // 4. Execute step 3: build (fails)
  console.log("Step 3: Build");
  const result3 = await saga.executeStep(mockBuild);
  console.log(`  Success: ${result3.success}`);
  console.log(`  Status:  ${saga.status}\n`);

  // 5. The build returned a failure result -- trigger compensation
  if (!result3.success) {
    console.log("Build failed! Compensating all completed operations...\n");
    const report = await saga.compensateAll();

    // 6. Print the compensation report
    console.log("\n--- Compensation Report ---");
    console.log(`  Saga: ${report.sagaId}`);
    for (const entry of report.operations) {
      const icon = entry.status === "success"
        ? "OK"
        : entry.status === "failed"
        ? "FAIL"
        : "SKIP";
      console.log(`  [${icon}] ${entry.description}`);
      if (entry.error) {
        console.log(`        reason: ${entry.error}`);
      }
    }
    console.log(`\n  Summary:        ${report.summary()}`);
    console.log(`  All successful: ${report.allSuccessful()}`);
    console.log(`  Final status:   ${saga.status}`);
  }

  console.log("\nSaga compensation demo complete.");
}

await main();
