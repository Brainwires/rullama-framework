/**
 * Tests for JudgeAgent, PlannerAgent, and ValidatorAgent prompt generation
 * and output parsing.
 */

import { assertEquals, assertThrows } from "@std/assert";

import {
  buildJudgeTaskDescription,
  extractJsonBlock,
  formatMergeStatus,
  judgeAgentPrompt,
  type JudgeContext,
  type JudgeVerdict,
  parseVerdict,
  verdictHints,
  verdictType,
} from "./judge_agent.ts";

import {
  defaultPlannerAgentConfig,
  type DynamicTaskSpec,
  parsePlannerOutput,
  plannerAgentPrompt,
  validateTaskGraph,
} from "./planner_agent.ts";

import { formatValidatorStatus } from "./validator_agent.ts";

import { defaultCycleOrchestratorConfig } from "./cycle_orchestrator.ts";

import { ExecutionGraph, telemetryFromGraph } from "@rullama/agent";

// ===========================================================================
// Judge Agent tests
// ===========================================================================

Deno.test("JudgeAgent: parse complete verdict", () => {
  const text =
    '```json\n{"verdict": "complete", "summary": "All tasks completed successfully"}\n```';
  const verdict = parseVerdict(text);
  assertEquals(verdict.verdict, "complete");
  assertEquals(verdictType(verdict), "complete");
});

Deno.test("JudgeAgent: parse continue verdict", () => {
  const text = `\`\`\`json
{
  "verdict": "continue",
  "summary": "Two tasks still need work",
  "additional_tasks": [
    {
      "id": "fix-1",
      "description": "Fix the remaining bug",
      "files_involved": ["src/bug.rs"],
      "depends_on": [],
      "priority": "high"
    }
  ],
  "retry_tasks": ["task-3"],
  "hints": ["Focus on error handling"]
}
\`\`\``;
  const verdict = parseVerdict(text);
  assertEquals(verdict.verdict, "continue");
  if (verdict.verdict === "continue") {
    assertEquals(verdict.additionalTasks.length, 1);
    assertEquals(verdict.retryTasks, ["task-3"]);
    assertEquals(verdict.hints, ["Focus on error handling"]);
  }
});

Deno.test("JudgeAgent: parse fresh_restart verdict", () => {
  const text = `\`\`\`json
{
  "verdict": "fresh_restart",
  "reason": "Agents went down the wrong path",
  "hints": ["Try a different approach", "Focus on the API first"],
  "summary": "Need to restart"
}
\`\`\``;
  const verdict = parseVerdict(text);
  assertEquals(verdict.verdict, "fresh_restart");
  if (verdict.verdict === "fresh_restart") {
    assertEquals(verdict.reason.includes("wrong path"), true);
    assertEquals(verdict.hints.length, 2);
  }
});

Deno.test("JudgeAgent: parse abort verdict", () => {
  const text =
    '```json\n{"verdict": "abort", "reason": "The goal requires external API access", "summary": "Cannot proceed"}\n```';
  const verdict = parseVerdict(text);
  assertEquals(verdict.verdict, "abort");
  assertEquals(verdictType(verdict), "abort");
});

Deno.test("JudgeAgent: verdict hints", () => {
  const complete: JudgeVerdict = { verdict: "complete", summary: "done" };
  assertEquals(verdictHints(complete).length, 0);

  const cont: JudgeVerdict = {
    verdict: "continue",
    summary: "partial",
    additionalTasks: [],
    retryTasks: [],
    hints: ["hint1"],
  };
  assertEquals(verdictHints(cont).length, 1);
});

Deno.test("JudgeAgent: merge status display", () => {
  assertEquals(formatMergeStatus({ kind: "merged" }), "merged");
  assertEquals(formatMergeStatus({ kind: "not_attempted" }), "not_attempted");
  assertEquals(
    formatMergeStatus({ kind: "conflict_failed", message: "oops" }).includes(
      "oops",
    ),
    true,
  );
});

Deno.test("JudgeAgent: extractJsonBlock from fenced code", () => {
  const text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.';
  const json = extractJsonBlock(text);
  assertEquals(json, '{"key": "value"}');
});

Deno.test("JudgeAgent: extractJsonBlock from raw JSON", () => {
  const text =
    'I think the result is {"tasks": [], "rationale": "test"} and that\'s it.';
  const json = extractJsonBlock(text);
  assertEquals(json !== null, true);
  const parsed = JSON.parse(json!);
  assertEquals(Array.isArray(parsed.tasks), true);
});

Deno.test("JudgeAgent: system prompt generation", () => {
  const prompt = judgeAgentPrompt("judge-1", "/home/dev/project");
  assertEquals(prompt.includes("judge-1"), true);
  assertEquals(prompt.includes("/home/dev/project"), true);
  assertEquals(prompt.includes("VERDICT TYPES"), true);
});

Deno.test("JudgeAgent: build task description", () => {
  const ctx: JudgeContext = {
    originalGoal: "Implement feature X",
    cycleNumber: 1,
    workerResults: [
      {
        taskId: "task-1",
        taskDescription: "Add the API endpoint",
        success: true,
        summary: "Done",
        iterations: 5,
        branchName: "cycle-1-task-1",
        mergeStatus: { kind: "merged" },
      },
    ],
    plannerRationale: "Focus on the API first",
    previousVerdicts: [],
  };

  const desc = buildJudgeTaskDescription(ctx);
  assertEquals(desc.includes("Implement feature X"), true);
  assertEquals(desc.includes("task-1"), true);
  assertEquals(desc.includes("Focus on the API first"), true);
});

// ===========================================================================
// Planner Agent tests
// ===========================================================================

Deno.test("PlannerAgent: system prompt generation", () => {
  const prompt = plannerAgentPrompt(
    "planner-1",
    "/home/dev/project",
    "Build a REST API",
    ["Focus on error handling"],
  );
  assertEquals(prompt.includes("planner-1"), true);
  assertEquals(prompt.includes("Build a REST API"), true);
  assertEquals(prompt.includes("Focus on error handling"), true);
  assertEquals(prompt.includes("HINTS FROM PREVIOUS CYCLES"), true);
});

Deno.test("PlannerAgent: system prompt without hints", () => {
  const prompt = plannerAgentPrompt("planner-1", "/dev", "Goal", []);
  assertEquals(prompt.includes("HINTS FROM PREVIOUS CYCLES"), false);
});

Deno.test("PlannerAgent: parse output", () => {
  const text = `\`\`\`json
{
  "tasks": [
    {
      "id": "task-1",
      "description": "Add error handling to parser",
      "files_involved": ["src/parser.rs"],
      "depends_on": [],
      "priority": "high",
      "estimated_iterations": 10
    },
    {
      "id": "task-2",
      "description": "Add tests for parser",
      "files_involved": ["tests/parser_test.rs"],
      "depends_on": ["task-1"],
      "priority": "normal",
      "estimated_iterations": 5
    }
  ],
  "sub_planners": [],
  "rationale": "Parser needs error handling before tests can be written"
}
\`\`\``;

  const config = defaultPlannerAgentConfig();
  const output = parsePlannerOutput(text, config);
  assertEquals(output.tasks.length, 2);
  assertEquals(output.tasks[0].id, "task-1");
  assertEquals(output.tasks[1].dependsOn, ["task-1"]);
  assertEquals(
    output.rationale,
    "Parser needs error handling before tests can be written",
  );
});

Deno.test("PlannerAgent: validate task graph - no cycle", () => {
  const tasks: DynamicTaskSpec[] = [
    {
      id: "a",
      description: "A",
      filesInvolved: [],
      dependsOn: [],
      priority: "normal",
      estimatedIterations: null,
    },
    {
      id: "b",
      description: "B",
      filesInvolved: [],
      dependsOn: ["a"],
      priority: "normal",
      estimatedIterations: null,
    },
  ];
  // Should not throw
  validateTaskGraph(tasks);
});

Deno.test("PlannerAgent: validate task graph - cycle detected", () => {
  const tasks: DynamicTaskSpec[] = [
    {
      id: "a",
      description: "A",
      filesInvolved: [],
      dependsOn: ["b"],
      priority: "normal",
      estimatedIterations: null,
    },
    {
      id: "b",
      description: "B",
      filesInvolved: [],
      dependsOn: ["a"],
      priority: "normal",
      estimatedIterations: null,
    },
  ];
  assertThrows(() => validateTaskGraph(tasks), Error, "Circular dependency");
});

Deno.test("PlannerAgent: truncate limits", () => {
  const text = `\`\`\`json
{
  "tasks": [
    {"id": "1", "description": "t1"},
    {"id": "2", "description": "t2"},
    {"id": "3", "description": "t3"}
  ],
  "sub_planners": [
    {"focus_area": "a", "context": "c", "max_depth": 1},
    {"focus_area": "b", "context": "c", "max_depth": 1}
  ],
  "rationale": "test"
}
\`\`\``;

  const config = {
    ...defaultPlannerAgentConfig(),
    maxTasks: 2,
    maxSubPlanners: 1,
  };
  const output = parsePlannerOutput(text, config);
  assertEquals(output.tasks.length, 2);
  assertEquals(output.subPlanners.length, 1);
});

// ===========================================================================
// Validator Agent tests
// ===========================================================================

Deno.test("ValidatorAgent: status formatting", () => {
  assertEquals(formatValidatorStatus({ kind: "idle" }), "Idle");
  assertEquals(formatValidatorStatus({ kind: "validating" }), "Validating");
  assertEquals(formatValidatorStatus({ kind: "passed" }), "Passed");
  assertEquals(
    formatValidatorStatus({ kind: "failed", issueCount: 3 }),
    "Failed (3 issues)",
  );
  assertEquals(
    formatValidatorStatus({ kind: "error", message: "oops" }),
    "Error: oops",
  );
});

// ===========================================================================
// Cycle Orchestrator tests
// ===========================================================================

Deno.test("CycleOrchestrator: config defaults", () => {
  const config = defaultCycleOrchestratorConfig();
  assertEquals(config.maxCycles, 5);
  assertEquals(config.maxWorkers, 5);
  assertEquals(config.autoMerge, true);
  assertEquals(config.mergeStrategy, "sequential");
  assertEquals(config.failurePolicy, "continue_on_failure");
});

// ===========================================================================
// ExecutionGraph tests
// ===========================================================================

Deno.test("ExecutionGraph: push step returns index", () => {
  const graph = new ExecutionGraph("abc123");
  const idx0 = graph.pushStep(1);
  const idx1 = graph.pushStep(2);
  assertEquals(idx0, 0);
  assertEquals(idx1, 1);
  assertEquals(graph.steps.length, 2);
});

Deno.test("ExecutionGraph: finalize step sets tokens", () => {
  const graph = new ExecutionGraph("abc123");
  const idx = graph.pushStep(1);
  const end = new Date().toISOString();
  graph.finalizeStep(idx, end, 100, 50, "stop");
  assertEquals(graph.steps[idx].promptTokens, 100);
  assertEquals(graph.steps[idx].completionTokens, 50);
  assertEquals(graph.steps[idx].finishReason, "stop");
});

Deno.test("ExecutionGraph: record tool call appends sequence", () => {
  const graph = new ExecutionGraph("abc123");
  const idx = graph.pushStep(1);
  graph.recordToolCall(idx, {
    toolUseId: "u1",
    toolName: "read_file",
    isError: false,
    executedAt: new Date().toISOString(),
  });
  graph.recordToolCall(idx, {
    toolUseId: "u2",
    toolName: "write_file",
    isError: false,
    executedAt: new Date().toISOString(),
  });
  assertEquals(graph.toolSequence, ["read_file", "write_file"]);
  assertEquals(graph.steps[idx].toolCalls.length, 2);
});

Deno.test("ExecutionGraph: telemetry from graph", () => {
  const start = new Date().toISOString();
  const graph = new ExecutionGraph("hash", start);
  const idx = graph.pushStep(1, start);
  graph.finalizeStep(idx, new Date().toISOString(), 100, 50, null);
  graph.recordToolCall(idx, {
    toolUseId: "u1",
    toolName: "bash",
    isError: false,
    executedAt: new Date().toISOString(),
  });
  graph.recordToolCall(idx, {
    toolUseId: "u2",
    toolName: "bash",
    isError: true,
    executedAt: new Date().toISOString(),
  });

  const telem = telemetryFromGraph(graph, new Date().toISOString(), true, 0.01);
  assertEquals(telem.totalIterations, 1);
  assertEquals(telem.totalToolCalls, 2);
  assertEquals(telem.toolErrorCount, 1);
  // "bash" appears twice but toolsUsed should deduplicate
  assertEquals(telem.toolsUsed, ["bash"]);
  assertEquals(telem.totalPromptTokens, 100);
  assertEquals(telem.totalCompletionTokens, 50);
  assertEquals(telem.success, true);
});

Deno.test("ExecutionGraph: tool sequence preserves order", () => {
  const graph = new ExecutionGraph("abc");
  const idx = graph.pushStep(1);
  for (const name of ["a", "b", "c", "b", "a"]) {
    graph.recordToolCall(idx, {
      toolUseId: "x",
      toolName: name,
      isError: false,
      executedAt: new Date().toISOString(),
    });
  }
  assertEquals(graph.toolSequence, ["a", "b", "c", "b", "a"]);
});
