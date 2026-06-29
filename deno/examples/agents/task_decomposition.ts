// Example: Task Decomposition & MDAP Cost Estimation
// Demonstrates task decomposition strategies and MDAP cost estimation
// using the scaling laws from the MAKER paper.
// Run: deno run deno/examples/agents/task_decomposition.ts

import {
  atomicDecomposition,
  compositeDecomposition,
  createAtomicSubtask,
  defaultDecomposeContext,
  estimateCallCost,
  estimateMdap,
  type MdapSubtask,
  MODEL_COSTS,
  type ModelCosts,
} from "@rullama/agent";

async function main() {
  console.log("=== Task Decomposition & MDAP Cost Estimation ===\n");

  // 1. Build a decomposition context and create subtasks manually
  console.log("--- 1. Sequential Task Decomposition ---\n");

  const _context = {
    ...defaultDecomposeContext("/home/user/project"),
    availableTools: ["read_file", "write_file", "bash"],
    additionalContext: "Rust web server project",
  };

  // Create subtasks with full control over fields
  const subtasks: MdapSubtask[] = [
    {
      id: "step-1",
      description: "Read the current handler code in src/handlers.rs",
      inputState: null,
      dependsOn: [],
      complexityEstimate: 0.2,
    },
    {
      id: "step-2",
      description: "Add a new GET /health endpoint that returns 200 OK",
      inputState: null,
      dependsOn: ["step-1"],
      complexityEstimate: 0.4,
    },
    {
      id: "step-3",
      description: "Register the route in src/router.rs",
      inputState: null,
      dependsOn: ["step-2"],
      complexityEstimate: 0.3,
    },
    {
      id: "step-4",
      description: "Write a test for the health endpoint in tests/api.rs",
      inputState: null,
      dependsOn: ["step-3"],
      complexityEstimate: 0.5,
    },
    {
      id: "step-5",
      description: "Run cargo test to verify everything compiles",
      inputState: null,
      dependsOn: ["step-4"],
      complexityEstimate: 0.3,
    },
  ];

  const result = compositeDecomposition(subtasks, { kind: "sequence" });

  console.log(`  Task decomposed into ${result.subtasks.length} subtasks:`);
  console.log(`  Is minimal: ${result.isMinimal}`);
  console.log(`  Total complexity: ${result.totalComplexity.toFixed(2)}`);
  console.log(`  Composition: ${result.compositionFunction.kind}`);
  console.log();

  for (const subtask of result.subtasks) {
    const deps = subtask.dependsOn.length === 0
      ? "none"
      : subtask.dependsOn.join(", ");
    console.log(
      `  [${subtask.id}] ${subtask.description} (complexity: ${
        subtask.complexityEstimate.toFixed(2)
      }, depends on: ${deps})`,
    );
  }

  // 2. Check if simple tasks are considered minimal (atomic)
  console.log("\n--- 2. Minimality Check ---\n");

  const simpleSubtask = createAtomicSubtask("Return the sum of two numbers");
  const atomicResult = atomicDecomposition(simpleSubtask);
  console.log(
    `  'Return the sum of two numbers' is minimal: ${atomicResult.isMinimal}`,
  );
  console.log(
    `  Multi-step task is minimal: ${result.isMinimal}`,
  );

  // 3. MDAP cost estimation with estimateMdap
  console.log("\n--- 3. MDAP Cost Estimation ---\n");

  const scenarios: Array<[string, number, number, number, number, number]> = [
    ["Simple (5 steps, p=0.95)", 5, 0.95, 0.90, 0.003, 0.95],
    ["Moderate (10 steps, p=0.85)", 10, 0.85, 0.85, 0.003, 0.95],
    ["Complex (20 steps, p=0.75)", 20, 0.75, 0.80, 0.003, 0.95],
    [
      "High-reliability (10 steps, p=0.90, t=0.99)",
      10,
      0.90,
      0.90,
      0.003,
      0.99,
    ],
  ];

  console.log(
    `  ${"Scenario".padEnd(45)} ${"k".padStart(5)} ${"Calls".padStart(8)} ${
      "Cost ($)".padStart(10)
    } ${"P(success)".padStart(8)}`,
  );
  console.log(`  ${"-".repeat(80)}`);

  for (const [name, steps, p, v, cost, target] of scenarios) {
    const estimate = estimateMdap(steps, p, v, cost, target);
    console.log(
      `  ${name.padEnd(45)} ${String(estimate.recommendedK).padStart(5)} ${
        String(estimate.expectedApiCalls).padStart(8)
      } ${estimate.expectedCostUsd.toFixed(4).padStart(10)} ${
        (estimate.successProbability * 100).toFixed(1).padStart(7)
      }%`,
    );
  }

  // 4. Compare model costs
  console.log("\n--- 4. Model Cost Comparison ---\n");

  const models: Array<[string, ModelCosts]> = [
    ["Claude Sonnet", MODEL_COSTS.claudeSonnet],
    ["Claude Haiku", MODEL_COSTS.claudeHaiku],
    ["GPT-4o", MODEL_COSTS.gpt4o],
    ["GPT-4o Mini", MODEL_COSTS.gpt4oMini],
  ];

  const inputTokens = 500;
  const outputTokens = 200;

  console.log(
    `  ${"Model".padEnd(16)} ${"Input/1K".padStart(12)} ${
      "Output/1K".padStart(12)
    } ${"Per Call Cost".padStart(14)}`,
  );
  console.log(`  ${"-".repeat(58)}`);

  for (const [name, costs] of models) {
    const callCost = estimateCallCost(costs, inputTokens, outputTokens);
    console.log(
      `  ${name.padEnd(16)} ${costs.inputPer1k.toFixed(5).padStart(11)}$ ${
        costs.outputPer1k.toFixed(5).padStart(11)
      }$ ${callCost.toFixed(6).padStart(13)}$`,
    );
  }

  // 5. Full cost projection: model costs * MDAP scaling
  console.log(
    "\n--- 5. Full MDAP Cost Projection (10 steps, p=0.85, target=0.95) ---\n",
  );

  const projSteps = 10;
  const projP = 0.85;
  const projV = 0.90;
  const projTarget = 0.95;

  console.log(
    `  ${"Model".padEnd(16)} ${"Per Call ($)".padStart(14)} ${
      "Est. Calls".padStart(10)
    } ${"Total ($)".padStart(12)}`,
  );
  console.log(`  ${"-".repeat(56)}`);

  for (const [name, costs] of models) {
    const callCost = estimateCallCost(costs, inputTokens, outputTokens);
    const estimate = estimateMdap(
      projSteps,
      projP,
      projV,
      callCost,
      projTarget,
    );
    console.log(
      `  ${name.padEnd(16)} ${callCost.toFixed(6).padStart(14)} ${
        String(estimate.expectedApiCalls).padStart(10)
      } ${estimate.expectedCostUsd.toFixed(4).padStart(12)}`,
    );
  }

  console.log("\n=== Done ===");
}

await main();
