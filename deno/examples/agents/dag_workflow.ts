// Example: DAG Workflow with ExecutionGraph
// Demonstrates the ExecutionGraph for tracking DAG-based execution with
// step nodes, tool call records, and telemetry generation.
// Run: deno run deno/examples/agents/dag_workflow.ts

import {
  ExecutionGraph,
  telemetryFromGraph,
  type ToolCallRecord,
} from "@rullama/agent";

async function main() {
  console.log("=== DAG Workflow (ExecutionGraph) ===\n");

  // 1. Create an execution graph for a multi-step agent run
  console.log("--- 1. Create Execution Graph ---");

  const graph = new ExecutionGraph("sha256-prompt-hash-abc");
  console.log(`  Prompt hash: ${graph.promptHash}`);
  console.log(`  Run started: ${graph.runStartedAt}`);

  // 2. Simulate iteration 1: planning step with tool calls
  console.log("\n--- 2. Iteration 1: Planning ---");

  const step1 = graph.pushStep(1);
  console.log(`  Step index: ${step1}`);

  const toolCall1: ToolCallRecord = {
    toolUseId: "tc-001",
    toolName: "list_directory",
    isError: false,
    executedAt: new Date().toISOString(),
  };
  graph.recordToolCall(step1, toolCall1);

  const toolCall2: ToolCallRecord = {
    toolUseId: "tc-002",
    toolName: "read_file",
    isError: false,
    executedAt: new Date().toISOString(),
  };
  graph.recordToolCall(step1, toolCall2);

  graph.finalizeStep(
    step1,
    new Date().toISOString(),
    1200,
    350,
    "stop",
  );

  console.log(`  Tool calls: ${graph.steps[step1].toolCalls.length}`);
  console.log(`  Prompt tokens: ${graph.steps[step1].promptTokens}`);
  console.log(`  Completion tokens: ${graph.steps[step1].completionTokens}`);

  // 3. Simulate iteration 2: implementation with a tool error
  console.log("\n--- 3. Iteration 2: Implementation ---");

  const step2 = graph.pushStep(2);

  graph.recordToolCall(step2, {
    toolUseId: "tc-003",
    toolName: "write_file",
    isError: false,
    executedAt: new Date().toISOString(),
  });

  graph.recordToolCall(step2, {
    toolUseId: "tc-004",
    toolName: "bash",
    isError: true,
    executedAt: new Date().toISOString(),
  });

  graph.recordToolCall(step2, {
    toolUseId: "tc-005",
    toolName: "write_file",
    isError: false,
    executedAt: new Date().toISOString(),
  });

  graph.finalizeStep(
    step2,
    new Date().toISOString(),
    2500,
    800,
    "stop",
  );

  console.log(`  Tool calls: ${graph.steps[step2].toolCalls.length}`);
  console.log(
    `  Errors: ${
      graph.steps[step2].toolCalls.filter((tc: ToolCallRecord) => tc.isError)
        .length
    }`,
  );

  // 4. Simulate iteration 3: verification
  console.log("\n--- 4. Iteration 3: Verification ---");

  const step3 = graph.pushStep(3);

  graph.recordToolCall(step3, {
    toolUseId: "tc-006",
    toolName: "bash",
    isError: false,
    executedAt: new Date().toISOString(),
  });

  graph.finalizeStep(
    step3,
    new Date().toISOString(),
    800,
    200,
    "stop",
  );

  console.log(`  Tool calls: ${graph.steps[step3].toolCalls.length}`);

  // 5. Show the full tool sequence
  console.log("\n--- 5. Tool Sequence ---");
  console.log(`  Flat sequence: [${graph.toolSequence.join(", ")}]`);
  console.log(`  Total steps: ${graph.steps.length}`);

  // 6. Generate telemetry
  console.log("\n--- 6. Run Telemetry ---");

  const telemetry = telemetryFromGraph(
    graph,
    new Date().toISOString(),
    true,
    0.0045,
  );

  console.log(`  Prompt hash:        ${telemetry.promptHash}`);
  console.log(`  Duration:           ${telemetry.durationMs}ms`);
  console.log(`  Total iterations:   ${telemetry.totalIterations}`);
  console.log(`  Total tool calls:   ${telemetry.totalToolCalls}`);
  console.log(`  Tool errors:        ${telemetry.toolErrorCount}`);
  console.log(`  Tools used:         [${telemetry.toolsUsed.join(", ")}]`);
  console.log(`  Total prompt tokens: ${telemetry.totalPromptTokens}`);
  console.log(`  Total compl tokens: ${telemetry.totalCompletionTokens}`);
  console.log(`  Estimated cost:     $${telemetry.totalCostUsd.toFixed(4)}`);
  console.log(`  Success:            ${telemetry.success}`);

  // 7. Inspect individual steps
  console.log("\n--- 7. Step Details ---");
  for (const step of graph.steps) {
    const tools = step.toolCalls.map((tc: ToolCallRecord) => tc.toolName).join(
      ", ",
    );
    const errors = step.toolCalls.filter((tc: ToolCallRecord) =>
      tc.isError
    ).length;
    console.log(
      `  Iteration ${step.iteration}: ${step.toolCalls.length} tools [${tools}], ${errors} errors, ${step.promptTokens}+${step.completionTokens} tokens`,
    );
  }

  console.log("\nDAG workflow demo complete.");
}

await main();
