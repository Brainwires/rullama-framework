// Example: Planner & Judge Parsing API
// Demonstrates the static parsing methods for PlannerOutput and JudgeVerdict
// without any async runtime, mock providers, or network calls. This is the
// simplest entry point for understanding the structured output formats
// used in the Plan->Work->Judge cycle.
// Run: deno run deno/examples/agents/planner_agent.ts

import {
  defaultPlannerAgentConfig,
  type JudgeVerdict,
  parsePlannerOutput,
  parseVerdict,
  type PlannerAgentConfig,
  validateTaskGraph,
  verdictHints,
  verdictType,
} from "@rullama/agent";

async function main() {
  // 1. Parse a planner output from a fenced JSON block
  const plannerText = `
I've analyzed the codebase. Here is the plan:

\`\`\`json
{
  "tasks": [
    {
      "id": "task-1",
      "description": "Add error handling to the parser module",
      "files_involved": ["src/parser.rs", "src/error.rs"],
      "depends_on": [],
      "priority": "high",
      "estimated_iterations": 10
    },
    {
      "id": "task-2",
      "description": "Write unit tests for the parser",
      "files_involved": ["tests/parser_test.rs"],
      "depends_on": ["task-1"],
      "priority": "normal",
      "estimated_iterations": 5
    },
    {
      "id": "task-3",
      "description": "Update documentation with new error types",
      "files_involved": ["docs/errors.md"],
      "depends_on": ["task-1"],
      "priority": "low"
    }
  ],
  "sub_planners": [
    {
      "focus_area": "Integration tests",
      "context": "Need end-to-end tests for the parser pipeline",
      "max_depth": 1
    }
  ],
  "rationale": "Parser needs robust error handling before tests can be meaningful"
}
\`\`\`

This plan prioritizes correctness before documentation.
`;

  const config = defaultPlannerAgentConfig();
  const output = parsePlannerOutput(plannerText, config);

  console.log("=== Planner Output ===");
  console.log(`Rationale: ${output.rationale}`);
  console.log(`Tasks (${output.tasks.length}):`);
  for (const task of output.tasks) {
    console.log(
      `  [${task.priority}] ${task.id} - ${task.description} (depends on: [${
        task.dependsOn.join(", ")
      }])`,
    );
    if (task.filesInvolved.length > 0) {
      console.log(`         files: [${task.filesInvolved.join(", ")}]`);
    }
  }
  console.log(`Sub-planners (${output.subPlanners.length}):`);
  for (const sp of output.subPlanners) {
    console.log(`  focus: ${sp.focusArea}, depth: ${sp.maxDepth}`);
  }

  // 2. Demonstrate task limit enforcement
  const strictConfig: PlannerAgentConfig = {
    ...defaultPlannerAgentConfig(),
    maxTasks: 2,
    maxSubPlanners: 0,
  };
  const limited = parsePlannerOutput(plannerText, strictConfig);
  console.log("\n=== With limits (max_tasks=2, max_sub_planners=0) ===");
  console.log(`Tasks: ${limited.tasks.length} (truncated from 3)`);
  console.log(`Sub-planners: ${limited.subPlanners.length}`);

  // 3. Demonstrate cycle detection
  const cyclicPlan = `\`\`\`json
{
  "tasks": [
    {"id": "a", "description": "Step A", "depends_on": ["b"]},
    {"id": "b", "description": "Step B", "depends_on": ["a"]}
  ],
  "rationale": "This plan has a circular dependency"
}
\`\`\``;

  try {
    parsePlannerOutput(cyclicPlan, config);
    console.log("\nUnexpected: cyclic plan was accepted");
  } catch (e) {
    console.log(
      `\n=== Cycle Detection ===\nCorrectly rejected: ${(e as Error).message}`,
    );
  }

  // 4. Parse all four judge verdict types
  console.log("\n=== Judge Verdicts ===");

  // Complete
  const completeText = `\`\`\`json
{"verdict": "complete", "summary": "All tasks finished, tests pass, code reviewed"}
\`\`\``;
  const completeVerdict = parseVerdict(completeText);
  printVerdict(completeVerdict);

  // Continue
  const continueText = `\`\`\`json
{
  "verdict": "continue",
  "summary": "Two of three tasks done, error handling still missing",
  "additional_tasks": [
    {"id": "fix-1", "description": "Add missing error variants", "priority": "high"}
  ],
  "retry_tasks": ["task-2"],
  "hints": ["Focus on the From<io::Error> impl", "Check edge cases in parse_header"]
}
\`\`\``;
  const continueVerdict = parseVerdict(continueText);
  printVerdict(continueVerdict);

  // FreshRestart
  const restartText = `\`\`\`json
{
  "verdict": "fresh_restart",
  "reason": "Workers modified the wrong module entirely",
  "hints": ["Target src/parser.rs not src/lexer.rs"],
  "summary": "Completely off track, need to start over"
}
\`\`\``;
  const restartVerdict = parseVerdict(restartText);
  printVerdict(restartVerdict);

  // Abort
  const abortText = `\`\`\`json
{
  "verdict": "abort",
  "reason": "Goal requires access to a private API we cannot reach",
  "summary": "Impossible to proceed without API credentials"
}
\`\`\``;
  const abortVerdict = parseVerdict(abortText);
  printVerdict(abortVerdict);

  console.log("\nAll parsing demonstrations complete.");
}

function printVerdict(verdict: JudgeVerdict): void {
  console.log(`\n  Verdict type: ${verdictType(verdict)}`);
  switch (verdict.verdict) {
    case "complete":
      console.log(`  Summary: ${verdict.summary}`);
      break;
    case "continue":
      console.log(`  Summary: ${verdict.summary}`);
      console.log(`  Additional tasks: ${verdict.additionalTasks.length}`);
      console.log(`  Retry tasks: [${verdict.retryTasks.join(", ")}]`);
      console.log(`  Hints: [${verdictHints(verdict).join(", ")}]`);
      break;
    case "fresh_restart":
      console.log(`  Reason: ${verdict.reason}`);
      console.log(`  Summary: ${verdict.summary}`);
      console.log(`  Hints: [${verdictHints(verdict).join(", ")}]`);
      break;
    case "abort":
      console.log(`  Reason: ${verdict.reason}`);
      console.log(`  Summary: ${verdict.summary}`);
      break;
  }
}

await main();
