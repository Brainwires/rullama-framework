/**
 * Tests for the MDAP module.
 */

import { assert, assertEquals, assertThrows } from "@std/assert";

import {
  AcceptAllValidator,
  aggressiveEarlyStopping,
  atomicDecomposition,
  // Borda count
  bordaCountWinner,
  calculateExpectedCost,
  calculateExpectedVotes,
  // Scaling
  calculateKMin,
  calculatePFull,
  childContext,
  // Composer
  Composer,
  compositeDecomposition,
  conservativeEarlyStopping,
  // Subtask helpers
  createAtomicSubtask,
  createSubtask,
  createSubtaskOutput,
  type DecompositionResult,
  // Decomposition
  defaultDecomposeContext,
  // Early stopping
  defaultEarlyStopping,
  // Microagent config
  defaultMicroagentConfig,
  disabledEarlyStopping,
  estimateCallCost,
  estimateMdap,
  estimatePerStepSuccess,
  estimateValidResponseRate,
  // Confidence
  extractResponseConfidence,
  // Voter
  FirstToAheadByKVoter,
  // Types
  MdapError,
  // Metrics
  MdapMetrics,
  MODEL_COSTS,
  outputFormatDescription,
  outputFormatMatches,
  parseToolIntent,
  readOnlyCategories,
  type RedFlagResult,
  relaxedRedFlagConfig,
  type ResponseMetadata,
  type SampledResponse,
  sideEffectCategories,
  StandardRedFlagValidator,
  // Red-flag
  strictRedFlagConfig,
  type SubtaskOutput,
  suggestKForBudget,
  // Tool intent
  toolCategoryContains,
  toolSchemaToPrompt,
  topologicalSort,
  validateDecomposition,
  VoterBuilder,
} from "./mod.ts";

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

function makeMetadata(tokenCount = 50): ResponseMetadata {
  return {
    tokenCount,
    responseTimeMs: 100,
    formatValid: true,
  };
}

function makeResponse<T>(
  value: T,
  raw: string,
  confidence = 0.75,
): SampledResponse<T> {
  return {
    value,
    metadata: makeMetadata(),
    rawResponse: raw,
    confidence,
  };
}

// ---------------------------------------------------------------------------
// MdapError tests
// ---------------------------------------------------------------------------

Deno.test("MdapError - static constructors", () => {
  const err = MdapError.other("test message");
  assertEquals(err.kind.type, "other");
  assertEquals(err.message, "test message");

  const provErr = MdapError.provider("connection failed");
  assertEquals(provErr.kind.type, "provider");
  assert(!provErr.isUserError());
  assert(provErr.isRetryable());
  assert(!provErr.isRedFlag());
  assert(!provErr.isToolError());
});

Deno.test("MdapError - isUserError", () => {
  const configErr = new MdapError({ type: "config", message: "bad config" });
  assert(configErr.isUserError());

  const scalingErr = new MdapError({ type: "scaling", message: "bad scale" });
  assert(scalingErr.isUserError());
});

Deno.test("MdapError - isToolError", () => {
  const toolErr = new MdapError({
    type: "tool_execution_failed",
    tool: "read_file",
    reason: "not found",
  });
  assert(toolErr.isToolError());
});

// ---------------------------------------------------------------------------
// Early stopping config tests
// ---------------------------------------------------------------------------

Deno.test("EarlyStopping - presets", () => {
  const def = defaultEarlyStopping();
  assert(def.enabled);
  assertEquals(def.minVotes, 3);

  const disabled = disabledEarlyStopping();
  assert(!disabled.enabled);
  assert(!disabled.lossOfHopeEnabled);

  const aggressive = aggressiveEarlyStopping();
  assertEquals(aggressive.minVotes, 2);
  assertEquals(aggressive.minConfidence, 0.75);

  const conservative = conservativeEarlyStopping();
  assertEquals(conservative.minVotes, 5);
  assertEquals(conservative.minConfidence, 0.9);
});

// ---------------------------------------------------------------------------
// Red-flag config and validator tests
// ---------------------------------------------------------------------------

Deno.test("RedFlagConfig - presets", () => {
  const strict = strictRedFlagConfig();
  assertEquals(strict.maxResponseTokens, 750);
  assert(strict.requireExactFormat);
  assert(strict.flagSelfCorrection);
  assert(strict.confusionPatterns.length > 0);

  const relaxed = relaxedRedFlagConfig();
  assertEquals(relaxed.maxResponseTokens, 1500);
  assert(!relaxed.requireExactFormat);
  assert(!relaxed.flagSelfCorrection);
});

Deno.test("StandardRedFlagValidator - valid response", () => {
  const validator = StandardRedFlagValidator.strict();
  const result = validator.validate("This is valid.", makeMetadata(50));
  assert(result.valid);
});

Deno.test("StandardRedFlagValidator - empty response", () => {
  const validator = StandardRedFlagValidator.strict();
  const result = validator.validate("", makeMetadata(0));
  assert(!result.valid);
  if (!result.valid) {
    assertEquals(result.reason.kind, "empty_response");
  }
});

Deno.test("StandardRedFlagValidator - response too long", () => {
  const validator = StandardRedFlagValidator.strict();
  const result = validator.validate("Some response", makeMetadata(800));
  assert(!result.valid);
  if (!result.valid) {
    assertEquals(result.reason.kind, "response_too_long");
  }
});

Deno.test("StandardRedFlagValidator - self-correction detected", () => {
  const validator = StandardRedFlagValidator.strict();
  const result = validator.validate(
    "Wait, I think I made an error.",
    makeMetadata(50),
  );
  assert(!result.valid);
  if (!result.valid) {
    assertEquals(result.reason.kind, "self_correction_detected");
  }
});

Deno.test("StandardRedFlagValidator - truncation detection", () => {
  const validator = StandardRedFlagValidator.strict();
  const metadata: ResponseMetadata = {
    ...makeMetadata(50),
    finishReason: "length",
  };
  const result = validator.validate("Truncated", metadata);
  assert(!result.valid);
  if (!result.valid) {
    assertEquals(result.reason.kind, "truncated");
  }
});

Deno.test("StandardRedFlagValidator - relaxed allows self-correction", () => {
  const validator = new StandardRedFlagValidator(relaxedRedFlagConfig());
  const result = validator.validate(
    "Wait, let me reconsider this.",
    makeMetadata(50),
  );
  assert(result.valid);
});

Deno.test("StandardRedFlagValidator - format validation JSON", () => {
  const validator = StandardRedFlagValidator.withFormat({ kind: "json" });
  assert(
    validator.validate('{"key": "value"}', makeMetadata(20)).valid,
  );
  assert(
    !validator.validate("not json", makeMetadata(10)).valid,
  );
});

Deno.test("StandardRedFlagValidator - format validation one_of", () => {
  const validator = StandardRedFlagValidator.withFormat({
    kind: "one_of",
    options: ["yes", "no", "maybe"],
  });
  assert(validator.validate("yes", makeMetadata(5)).valid);
  assert(!validator.validate("perhaps", makeMetadata(10)).valid);
});

Deno.test("AcceptAllValidator - accepts everything", () => {
  const validator = new AcceptAllValidator();
  assert(validator.validate("", makeMetadata(0)).valid);
  assert(validator.validate("anything", makeMetadata(10000)).valid);
});

// ---------------------------------------------------------------------------
// Output format tests
// ---------------------------------------------------------------------------

Deno.test("outputFormatMatches - exact", () => {
  assert(outputFormatMatches({ kind: "exact", value: "hello" }, "hello"));
  assert(outputFormatMatches({ kind: "exact", value: "hello" }, "  hello  "));
  assert(!outputFormatMatches({ kind: "exact", value: "hello" }, "world"));
});

Deno.test("outputFormatMatches - pattern", () => {
  assert(outputFormatMatches({ kind: "pattern", regex: "^\\d+$" }, "123"));
  assert(!outputFormatMatches({ kind: "pattern", regex: "^\\d+$" }, "abc"));
});

Deno.test("outputFormatMatches - json_with_fields", () => {
  assert(
    outputFormatMatches(
      { kind: "json_with_fields", fields: ["name", "value"] },
      '{"name": "test", "value": 42}',
    ),
  );
  assert(
    !outputFormatMatches(
      { kind: "json_with_fields", fields: ["name", "value"] },
      '{"name": "test"}',
    ),
  );
});

Deno.test("outputFormatMatches - markers", () => {
  assert(
    outputFormatMatches(
      { kind: "markers", start: "```", end: "```" },
      "```code here```",
    ),
  );
  assert(
    !outputFormatMatches(
      { kind: "markers", start: "```", end: "```" },
      "no markers",
    ),
  );
});

Deno.test("outputFormatDescription", () => {
  assertEquals(outputFormatDescription({ kind: "json" }), "valid JSON");
  assertEquals(
    outputFormatDescription({ kind: "exact", value: "test" }),
    "exact: 'test'",
  );
});

// ---------------------------------------------------------------------------
// Confidence extraction tests
// ---------------------------------------------------------------------------

Deno.test("extractResponseConfidence - baseline", () => {
  const conf = extractResponseConfidence("normal response", makeMetadata(50));
  assert(conf > 0.7 && conf < 0.95);
});

Deno.test("extractResponseConfidence - hedging lowers confidence", () => {
  const conf = extractResponseConfidence(
    "I think maybe this could be right",
    makeMetadata(50),
  );
  const baseline = extractResponseConfidence("this is right", makeMetadata(50));
  assert(conf < baseline);
});

Deno.test("extractResponseConfidence - self-correction lowers confidence", () => {
  const conf = extractResponseConfidence(
    "wait, actually, I was wrong",
    makeMetadata(50),
  );
  assert(conf < 0.7);
});

Deno.test("extractResponseConfidence - stop finish reason boosts", () => {
  const metadata: ResponseMetadata = {
    ...makeMetadata(50),
    finishReason: "stop",
  };
  const conf = extractResponseConfidence("good answer", metadata);
  const baseline = extractResponseConfidence("good answer", makeMetadata(50));
  assert(conf > baseline);
});

// ---------------------------------------------------------------------------
// Subtask helper tests
// ---------------------------------------------------------------------------

Deno.test("createAtomicSubtask", () => {
  const s = createAtomicSubtask("Calculate 2 + 2");
  assertEquals(s.description, "Calculate 2 + 2");
  assert(s.id.length > 0);
  assertEquals(s.dependsOn.length, 0);
  assertEquals(s.complexityEstimate, 0.5);
});

Deno.test("createSubtask", () => {
  const s = createSubtask("task_1", "Add numbers", { a: 1, b: 2 });
  assertEquals(s.id, "task_1");
  assertEquals(s.description, "Add numbers");
});

Deno.test("createSubtaskOutput", () => {
  const o = createSubtaskOutput("task_1", 42);
  assertEquals(o.subtaskId, "task_1");
  assertEquals(o.value, 42);
});

// ---------------------------------------------------------------------------
// Microagent config test
// ---------------------------------------------------------------------------

Deno.test("defaultMicroagentConfig", () => {
  const config = defaultMicroagentConfig();
  assertEquals(config.maxOutputTokens, 750);
  assertEquals(config.temperature, 0.1);
  assertEquals(config.timeoutMs, 30000);
});

// ---------------------------------------------------------------------------
// Decomposition tests
// ---------------------------------------------------------------------------

Deno.test("defaultDecomposeContext", () => {
  const ctx = defaultDecomposeContext("/home/user/project");
  assertEquals(ctx.workingDirectory, "/home/user/project");
  assertEquals(ctx.maxDepth, 10);
  assertEquals(ctx.currentDepth, 0);
});

Deno.test("childContext increments depth", () => {
  const parent = defaultDecomposeContext();
  const child = childContext(parent);
  assertEquals(child.currentDepth, 1);
  assertEquals(child.maxDepth, parent.maxDepth);
});

Deno.test("atomicDecomposition", () => {
  const subtask = createAtomicSubtask("Test");
  const result = atomicDecomposition(subtask);
  assert(result.isMinimal);
  assertEquals(result.subtasks.length, 1);
  assertEquals(result.compositionFunction.kind, "identity");
});

Deno.test("compositeDecomposition", () => {
  const s1 = createSubtask("a", "Task A");
  const s2 = createSubtask("b", "Task B");
  const result = compositeDecomposition([s1, s2], { kind: "sequence" });
  assert(!result.isMinimal);
  assertEquals(result.subtasks.length, 2);
});

Deno.test("validateDecomposition - valid", () => {
  const s1 = createSubtask("a", "A");
  const s2 = { ...createSubtask("b", "B"), dependsOn: ["a"] };
  const result = compositeDecomposition([s1, s2], { kind: "sequence" });
  validateDecomposition(result); // should not throw
});

Deno.test("validateDecomposition - invalid dependency", () => {
  const s1 = { ...createSubtask("a", "A"), dependsOn: ["nonexistent"] };
  const result = compositeDecomposition([s1], { kind: "sequence" });
  assertThrows(() => validateDecomposition(result), MdapError);
});

Deno.test("validateDecomposition - empty", () => {
  const result: DecompositionResult = {
    subtasks: [],
    compositionFunction: { kind: "identity" },
    isMinimal: true,
    totalComplexity: 0,
  };
  assertThrows(() => validateDecomposition(result), MdapError);
});

Deno.test("topologicalSort - linear chain", () => {
  const a = createSubtask("a", "A");
  const b = { ...createSubtask("b", "B"), dependsOn: ["a"] };
  const c = { ...createSubtask("c", "C"), dependsOn: ["b"] };
  const sorted = topologicalSort([a, b, c]);
  assertEquals(sorted[0].id, "a");
  assertEquals(sorted[1].id, "b");
  assertEquals(sorted[2].id, "c");
});

Deno.test("topologicalSort - parallel", () => {
  const a = createSubtask("a", "A");
  const b = createSubtask("b", "B");
  const c = { ...createSubtask("c", "C"), dependsOn: ["a", "b"] };
  const sorted = topologicalSort([a, b, c]);
  const cPos = sorted.findIndex((s) => s.id === "c");
  const aPos = sorted.findIndex((s) => s.id === "a");
  const bPos = sorted.findIndex((s) => s.id === "b");
  assert(aPos < cPos);
  assert(bPos < cPos);
});

Deno.test("topologicalSort - circular dependency", () => {
  const a = { ...createSubtask("a", "A"), dependsOn: ["c"] };
  const b = { ...createSubtask("b", "B"), dependsOn: ["a"] };
  const c = { ...createSubtask("c", "C"), dependsOn: ["b"] };
  assertThrows(() => topologicalSort([a, b, c]), MdapError);
});

// ---------------------------------------------------------------------------
// Composer tests
// ---------------------------------------------------------------------------

function makeOutput(id: string, value: unknown): SubtaskOutput {
  return { subtaskId: id, value };
}

Deno.test("Composer - identity", () => {
  const c = new Composer();
  const result = c.compose([makeOutput("a", "hello")], { kind: "identity" });
  assertEquals(result, "hello");
});

Deno.test("Composer - concatenate", () => {
  const c = new Composer();
  const result = c.compose(
    [makeOutput("a", "hello"), makeOutput("b", "world")],
    { kind: "concatenate" },
  );
  assertEquals(result, "hello\nworld");
});

Deno.test("Composer - sequence", () => {
  const c = new Composer();
  const result = c.compose(
    [makeOutput("a", 1), makeOutput("b", 2), makeOutput("c", 3)],
    { kind: "sequence" },
  );
  assertEquals(result, [1, 2, 3]);
});

Deno.test("Composer - object_merge", () => {
  const c = new Composer();
  const result = c.compose(
    [makeOutput("a", { x: 1 }), makeOutput("b", { y: 2 })],
    { kind: "object_merge" },
  );
  assertEquals(result, { x: 1, y: 2 });
});

Deno.test("Composer - last_only", () => {
  const c = new Composer();
  const result = c.compose(
    [makeOutput("a", 1), makeOutput("b", 2), makeOutput("c", 3)],
    { kind: "last_only" },
  );
  assertEquals(result, 3);
});

Deno.test("Composer - reduce sum", () => {
  const c = new Composer();
  const result = c.compose(
    [makeOutput("a", 10), makeOutput("b", 20), makeOutput("c", 30)],
    { kind: "reduce", operation: "sum" },
  );
  assertEquals(result, 60);
});

Deno.test("Composer - reduce product", () => {
  const c = new Composer();
  const result = c.compose(
    [makeOutput("a", 2), makeOutput("b", 3), makeOutput("c", 4)],
    { kind: "reduce", operation: "product" },
  );
  assertEquals(result, 24);
});

Deno.test("Composer - reduce max", () => {
  const c = new Composer();
  const result = c.compose(
    [makeOutput("a", 10), makeOutput("b", 50), makeOutput("c", 30)],
    { kind: "reduce", operation: "max" },
  );
  assertEquals(result, 50);
});

Deno.test("Composer - reduce and", () => {
  const c = new Composer();
  assertEquals(
    c.compose(
      [makeOutput("a", true), makeOutput("b", true)],
      { kind: "reduce", operation: "and" },
    ),
    true,
  );
  assertEquals(
    c.compose(
      [makeOutput("a", true), makeOutput("b", false)],
      { kind: "reduce", operation: "and" },
    ),
    false,
  );
});

Deno.test("Composer - reduce or", () => {
  const c = new Composer();
  assertEquals(
    c.compose(
      [makeOutput("a", false), makeOutput("b", false)],
      { kind: "reduce", operation: "or" },
    ),
    false,
  );
  assertEquals(
    c.compose(
      [makeOutput("a", false), makeOutput("b", true)],
      { kind: "reduce", operation: "or" },
    ),
    true,
  );
});

Deno.test("Composer - empty results throws", () => {
  const c = new Composer();
  assertThrows(
    () => c.compose([], { kind: "identity" }),
    MdapError,
  );
});

Deno.test("Composer - custom composition", () => {
  const c = new Composer();
  const result = c.compose(
    [makeOutput("a", 1), makeOutput("b", 2)],
    { kind: "custom", description: "test composition" },
  ) as { composition: string; results: unknown[] };
  assertEquals(result.composition, "test composition");
  assertEquals(result.results, [1, 2]);
});

// ---------------------------------------------------------------------------
// Scaling law tests
// ---------------------------------------------------------------------------

Deno.test("calculateKMin - basic", () => {
  const k = calculateKMin(100, 0.99, 0.95);
  assert(k >= 1);
  assert(k <= 10);
});

Deno.test("calculateKMin - low p needs higher k", () => {
  const k = calculateKMin(10, 0.6, 0.95);
  assert(k > 1);
});

Deno.test("calculateKMin - p <= 0.5 returns MAX_SAFE_INTEGER", () => {
  assertEquals(calculateKMin(10, 0.5, 0.95), Number.MAX_SAFE_INTEGER);
  assertEquals(calculateKMin(10, 0.4, 0.95), Number.MAX_SAFE_INTEGER);
});

Deno.test("calculatePFull - high p and k gives high success", () => {
  const pFull = calculatePFull(10, 0.99, 5);
  assert(pFull > 0.99);
});

Deno.test("calculatePFull - p <= 0.5 returns 0", () => {
  assertEquals(calculatePFull(10, 0.5, 5), 0);
});

Deno.test("calculateExpectedVotes - basic", () => {
  const votes = calculateExpectedVotes(0.99, 3);
  assert(Math.abs(votes - 3.06) < 0.1);
});

Deno.test("calculateExpectedVotes - p=0.5 returns Infinity", () => {
  assertEquals(calculateExpectedVotes(0.5, 3), Infinity);
});

Deno.test("estimateMdap - valid inputs", () => {
  const est = estimateMdap(100, 0.99, 0.95, 0.001, 0.95);
  assert(est.successProbability > 0.9);
  assert(est.recommendedK >= 1);
  assert(est.expectedCostUsd > 0);
  assert(est.expectedApiCalls > 0);
});

Deno.test("estimateMdap - invalid p throws", () => {
  assertThrows(() => estimateMdap(100, 0.4, 0.95, 0.001, 0.95), MdapError);
});

Deno.test("estimateMdap - invalid steps throws", () => {
  assertThrows(() => estimateMdap(0, 0.99, 0.95, 0.001, 0.95), MdapError);
});

Deno.test("estimateMdap - high step count", () => {
  const est = estimateMdap(1_000_000, 0.99, 0.95, 0.0001, 0.95);
  assert(est.successProbability > 0.9);
});

Deno.test("estimatePerStepSuccess", () => {
  const p = estimatePerStepSuccess(100, 80, 10);
  assert(Math.abs(p - 0.889) < 0.01);
  assertEquals(estimatePerStepSuccess(100, 0, 100), 0.5);
});

Deno.test("estimateValidResponseRate", () => {
  assertEquals(estimateValidResponseRate(100, 10), 0.9);
  assertEquals(estimateValidResponseRate(0, 0), 0.95);
});

Deno.test("calculateExpectedCost", () => {
  const cost = calculateExpectedCost(100, 3, 0.95, 0.99, 0.001);
  assert(cost > 0);
  const costHighK = calculateExpectedCost(100, 10, 0.95, 0.99, 0.001);
  assert(costHighK > cost);
});

Deno.test("suggestKForBudget", () => {
  const k = suggestKForBudget(100, 0.99, 0.95, 0.001, 1.0);
  assert(k >= 1);
  const kSmall = suggestKForBudget(100, 0.99, 0.95, 0.001, 0.1);
  assert(kSmall <= k);
});

Deno.test("MODEL_COSTS and estimateCallCost", () => {
  const sonnet = MODEL_COSTS.claudeSonnet;
  const cost = estimateCallCost(sonnet, 1000, 500);
  // 1000 input = $0.003, 500 output = $0.0075
  assert(Math.abs(cost - 0.0105) < 0.001);
});

// ---------------------------------------------------------------------------
// Metrics tests
// ---------------------------------------------------------------------------

Deno.test("MdapMetrics - creation", () => {
  const m = new MdapMetrics("test_exec_001");
  assertEquals(m.executionId, "test_exec_001");
  assert(m.startTime != null);
});

Deno.test("MdapMetrics - recordSubtask", () => {
  const m = new MdapMetrics("test");
  m.totalSteps = 2;
  m.recordSubtask({
    subtaskId: "1",
    description: "Test subtask 1",
    samplesNeeded: 5,
    redFlagsHit: 1,
    redFlagReasons: ["ResponseTooLong"],
    finalConfidence: 0.9,
    executionTimeMs: 500,
    winnerVotes: 3,
    totalVotes: 4,
    succeeded: true,
    inputTokens: 100,
    outputTokens: 50,
    complexityEstimate: 0.5,
  });
  m.recordSubtask({
    subtaskId: "2",
    description: "Test subtask 2",
    samplesNeeded: 5,
    redFlagsHit: 1,
    redFlagReasons: ["ResponseTooLong"],
    finalConfidence: 0.9,
    executionTimeMs: 500,
    winnerVotes: 3,
    totalVotes: 4,
    succeeded: true,
    inputTokens: 100,
    outputTokens: 50,
    complexityEstimate: 0.5,
  });
  assertEquals(m.completedSteps, 2);
  assertEquals(m.totalSamples, 10);
  assertEquals(m.redFlaggedSamples, 2);
});

Deno.test("MdapMetrics - finalize", () => {
  const m = new MdapMetrics("test");
  m.totalSteps = 2;
  m.recordSubtask({
    subtaskId: "1",
    description: "Test",
    samplesNeeded: 5,
    redFlagsHit: 0,
    redFlagReasons: [],
    finalConfidence: 0.9,
    executionTimeMs: 500,
    winnerVotes: 3,
    totalVotes: 4,
    succeeded: true,
    inputTokens: 100,
    outputTokens: 50,
    complexityEstimate: 0.5,
  });
  m.finalize(true);
  assert(m.endTime != null);
  assert(m.finalSuccess);
  assert(m.averageVotesPerStep > 0);
});

Deno.test("MdapMetrics - summary", () => {
  const m = new MdapMetrics("test");
  m.totalSteps = 1;
  m.recordSubtask({
    subtaskId: "1",
    description: "Test",
    samplesNeeded: 5,
    redFlagsHit: 0,
    redFlagReasons: [],
    finalConfidence: 0.9,
    executionTimeMs: 500,
    winnerVotes: 3,
    totalVotes: 4,
    succeeded: true,
    inputTokens: 100,
    outputTokens: 50,
    complexityEstimate: 0.5,
  });
  m.finalize(true);
  const summary = m.summary();
  assert(summary.includes("Steps:"));
  assert(summary.includes("Samples:"));
  assert(summary.includes("Cost:"));
});

Deno.test("MdapMetrics - redFlagAnalysis", () => {
  const m = new MdapMetrics("test");
  m.recordSubtask({
    subtaskId: "1",
    description: "Test",
    samplesNeeded: 5,
    redFlagsHit: 2,
    redFlagReasons: ["ResponseTooLong", "ResponseTooLong"],
    finalConfidence: 0.9,
    executionTimeMs: 500,
    winnerVotes: 3,
    totalVotes: 3,
    succeeded: true,
    inputTokens: 100,
    outputTokens: 50,
    complexityEstimate: 0.5,
  });
  m.redFlaggedSamples = 5;
  const analysis = m.redFlagAnalysis();
  assert(analysis.includes("ResponseTooLong"));
});

// ---------------------------------------------------------------------------
// Tool intent tests
// ---------------------------------------------------------------------------

Deno.test("toolCategoryContains", () => {
  assert(toolCategoryContains("file_read", "read_file"));
  assert(toolCategoryContains("file_write", "write_file"));
  assert(toolCategoryContains("search", "grep"));
  assert(toolCategoryContains("mcp", "mcp__rullama-rag__query"));
  assert(!toolCategoryContains("file_read", "bash"));
  assert(toolCategoryContains({ custom: "my_" }, "my_tool"));
});

Deno.test("readOnlyCategories and sideEffectCategories", () => {
  const ro = readOnlyCategories();
  assert(ro.has("file_read"));
  assert(ro.has("search"));
  assert(!ro.has("file_write"));

  const se = sideEffectCategories();
  assert(se.has("file_write"));
  assert(se.has("bash"));
  assert(!se.has("file_read"));
});

Deno.test("toolSchemaToPrompt", () => {
  const schema = {
    name: "read_file",
    description: "Read a file",
    parameters: { path: "The file path" },
    required: ["path"],
  };
  const prompt = toolSchemaToPrompt(schema);
  assert(prompt.includes("read_file"));
  assert(prompt.includes("(required)"));
});

Deno.test("parseToolIntent - no intent", () => {
  const result = parseToolIntent("task-1", "Just a regular response.");
  assertEquals(result.kind, "no_intent");
});

Deno.test("parseToolIntent - with JSON block", () => {
  const response = `I need to read a file.

\`\`\`json
{
  "tool_name": "read_file",
  "arguments": {"path": "/test.txt"},
  "rationale": "Check contents"
}
\`\`\`
`;
  const result = parseToolIntent("task-1", response);
  assertEquals(result.kind, "with_intent");
  if (result.kind === "with_intent") {
    assertEquals(result.toolIntent.toolName, "read_file");
  }
});

// ---------------------------------------------------------------------------
// Borda count tests
// ---------------------------------------------------------------------------

Deno.test("bordaCountWinner - basic", () => {
  const winner = bordaCountWinner([
    { key: "a", value: "A", confidence: 0.9 },
    { key: "b", value: "B", confidence: 0.6 },
    { key: "a", value: "A", confidence: 0.8 },
  ]);
  assert(winner != null);
  assertEquals(winner!.key, "a");
  assert(winner!.score > 1.5);
});

Deno.test("bordaCountWinner - empty", () => {
  assertEquals(bordaCountWinner([]), null);
});

// ---------------------------------------------------------------------------
// FirstToAheadByKVoter tests
// ---------------------------------------------------------------------------

Deno.test("FirstToAheadByKVoter - unanimous voting", async () => {
  const voter = FirstToAheadByKVoter.create(3, 50);
  const validator = new AcceptAllValidator();

  let count = 0;
  const result = await voter.vote(
    // deno-lint-ignore require-await
    async () => {
      count++;
      return makeResponse(`answer_a_${count}`, "answer_a");
    },
    validator,
    () => "answer_a",
  );

  assertEquals(result.winnerVotes, 3);
  assertEquals(result.confidence, 1.0);
  assertEquals(result.redFlaggedCount, 0);
});

Deno.test("FirstToAheadByKVoter - red-flagging", async () => {
  const voter = FirstToAheadByKVoter.create(3, 50);
  const validator = {
    validate(response: string, _metadata: ResponseMetadata): RedFlagResult {
      if (response.includes("bad")) {
        return {
          valid: false,
          reason: { kind: "confused_reasoning" as const, pattern: "bad" },
          severity: 0.8,
        };
      }
      return { valid: true };
    },
  };

  let count = 0;
  const result = await voter.vote(
    // deno-lint-ignore require-await
    async () => {
      count++;
      const raw = count % 3 === 0 ? "bad response" : "good";
      return makeResponse("answer", raw);
    },
    validator,
    () => "answer",
  );

  assert(result.redFlaggedCount > 0);
  assert(result.totalSamples > result.totalVotes);
});

Deno.test("FirstToAheadByKVoter - max samples exceeded", async () => {
  const voter = FirstToAheadByKVoter.withEarlyStopping(
    10,
    5,
    disabledEarlyStopping(),
  );
  const validator = new AcceptAllValidator();

  let count = 0;
  try {
    await voter.vote(
      // deno-lint-ignore require-await
      async () => {
        count++;
        return makeResponse(`unique_${count}`, "response");
      },
      validator,
      (v) => v,
    );
    assert(false, "Should have thrown");
  } catch (e) {
    assert(e instanceof MdapError);
    assertEquals(e.kind.type, "voting");
    if (e.kind.type === "voting") {
      assertEquals(e.kind.details.kind, "max_samples_exceeded");
    }
  }
});

Deno.test("FirstToAheadByKVoter - all red-flagged", async () => {
  const voter = FirstToAheadByKVoter.create(3, 20);
  const validator = {
    validate(_response: string, _metadata: ResponseMetadata): RedFlagResult {
      return {
        valid: false,
        reason: { kind: "empty_response" as const },
        severity: 1.0,
      };
    },
  };

  try {
    await voter.vote(
      // deno-lint-ignore require-await
      async () => makeResponse("value", "response"),
      validator,
      (v) => v,
    );
    assert(false, "Should have thrown");
  } catch (e) {
    assert(e instanceof MdapError);
    assertEquals(e.kind.type, "voting");
    if (e.kind.type === "voting") {
      assertEquals(e.kind.details.kind, "all_samples_red_flagged");
    }
  }
});

Deno.test("VoterBuilder - basic", () => {
  const voter = new VoterBuilder().k(5).maxSamples(100).batchSize(2).build();
  assertEquals(voter.k, 5);
  assertEquals(voter.maxSamples, 100);
});

Deno.test("VoterBuilder - confidence weighted", () => {
  const voter = new VoterBuilder().confidenceWeighted(true).build();
  assertEquals(voter.votingMethod, "confidence_weighted");
});

Deno.test("FirstToAheadByKVoter.create - k < 1 throws", () => {
  assertThrows(() => FirstToAheadByKVoter.create(0, 50));
});
