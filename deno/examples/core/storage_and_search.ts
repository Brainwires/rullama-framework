// Example: Storage, search, and output parsing
// Shows how to use output parsers, plans, error handling, and content source trust.
// Run: deno run deno/examples/core/storage_and_search.ts

import {
  canOverride,
  type ContentSource,
  extractJson,
  FrameworkError,
  JsonListParser,
  JsonOutputParser,
  PlanBudget,
  PlanMetadata,
  type PlanStep,
  RegexOutputParser,
  requiresSanitization,
  SerializablePlan,
} from "@rullama/core";

async function main() {
  console.log("=== Storage, Search, and Output Parsing ===");

  // 1. Output parsers for structured LLM responses
  console.log("\n=== JSON Output Parser ===");

  interface SentimentResult {
    sentiment: string;
    score: number;
    explanation: string;
  }

  const jsonParser = new JsonOutputParser<SentimentResult>();
  console.log(`  Instructions: ${jsonParser.formatInstructions()}`);

  // Parse JSON from LLM text (handles code fences and surrounding prose)
  const llmResponse1 = `Here is my analysis:
\`\`\`json
{"sentiment": "positive", "score": 0.92, "explanation": "The text expresses enthusiasm and optimism."}
\`\`\``;

  try {
    const result = jsonParser.parse(llmResponse1);
    console.log(`  Sentiment: ${result.sentiment}`);
    console.log(`  Score: ${result.score}`);
    console.log(`  Explanation: ${result.explanation}`);
  } catch (e) {
    console.log(`  Parse error: ${(e as Error).message}`);
  }

  // Parse JSON embedded in prose
  const llmResponse2 =
    'After analysis, the result is {"sentiment": "neutral", "score": 0.5, "explanation": "Mixed signals."}. Hope that helps!';
  try {
    const result = jsonParser.parse(llmResponse2);
    console.log(
      `  Embedded JSON: sentiment=${result.sentiment}, score=${result.score}`,
    );
  } catch (e) {
    console.log(`  Parse error: ${(e as Error).message}`);
  }

  // 2. JSON list parser
  console.log("\n=== JSON List Parser ===");

  interface TodoItem {
    task: string;
    priority: string;
  }

  const listParser = new JsonListParser<TodoItem>();
  console.log(`  Instructions: ${listParser.formatInstructions()}`);

  const llmList =
    '[{"task": "Fix auth bug", "priority": "high"}, {"task": "Update docs", "priority": "low"}]';
  try {
    const items = listParser.parse(llmList);
    for (const item of items) {
      console.log(`  - [${item.priority}] ${item.task}`);
    }
  } catch (e) {
    console.log(`  Parse error: ${(e as Error).message}`);
  }

  // 3. Regex output parser
  console.log("\n=== Regex Output Parser ===");

  const regexParser = new RegexOutputParser(
    /ANSWER:\s*(?<answer>.+?)\s*CONFIDENCE:\s*(?<confidence>\d+)%/,
  );
  console.log(`  Instructions: ${regexParser.formatInstructions()}`);

  try {
    const result = regexParser.parse(
      "ANSWER: TypeScript is great CONFIDENCE: 95%",
    );
    console.log(`  Answer: ${result.answer}`);
    console.log(`  Confidence: ${result.confidence}%`);
  } catch (e) {
    console.log(`  Parse error: ${(e as Error).message}`);
  }

  // 4. JSON extraction utility
  console.log("\n=== JSON Extraction ===");
  const testCases = [
    '{"key": "value"}',
    'Some text before {"key": "value"} and after',
    '```json\n{"key": "value"}\n```',
    "No JSON here at all",
  ];

  for (const text of testCases) {
    const extracted = extractJson(text);
    const preview = text.slice(0, 40) + (text.length > 40 ? "..." : "");
    console.log(`  "${preview}" => ${extracted ?? "(none)"}`);
  }

  // 5. Execution plans with budgets
  console.log("\n=== Execution Plans ===");

  const steps: PlanStep[] = [
    {
      step_number: 1,
      description: "Read existing auth module",
      tool_hint: "read_file",
      estimated_tokens: 500,
    },
    {
      step_number: 2,
      description: "Analyze security vulnerabilities",
      estimated_tokens: 2000,
    },
    {
      step_number: 3,
      description: "Generate fix suggestions",
      tool_hint: "write_file",
      estimated_tokens: 1500,
    },
    {
      step_number: 4,
      description: "Write unit tests",
      tool_hint: "write_file",
      estimated_tokens: 1000,
    },
  ];

  const plan = new SerializablePlan(
    "Fix authentication security issues",
    steps,
  );
  console.log(`  Plan ID: ${plan.plan_id}`);
  console.log(`  Steps: ${plan.stepCount()}`);
  console.log(`  Estimated tokens: ${plan.totalEstimatedTokens()}`);

  // Check plan against a budget
  const budget = new PlanBudget()
    .withMaxSteps(10)
    .withMaxTokens(10000)
    .withMaxCostUsd(0.05);

  const budgetCheck = budget.check(plan);
  console.log(`  Budget check: ${budgetCheck ?? "OK (within budget)"}`);

  // A tighter budget that fails
  const tightBudget = new PlanBudget().withMaxSteps(2);
  const tightCheck = tightBudget.check(plan);
  console.log(`  Tight budget: ${tightCheck ?? "OK"}`);

  // Parse a plan from LLM text
  const llmPlanText = `Here is my execution plan:
  {
    "steps": [
      {"description": "Read the config file", "tool": "read_file", "estimated_tokens": 300},
      {"description": "Update the settings", "tool": "write_file", "estimated_tokens": 500}
    ]
  }`;
  const parsedPlan = SerializablePlan.parseFromText(
    "Update config settings",
    llmPlanText,
  );
  if (parsedPlan) {
    console.log(
      `  Parsed plan: ${parsedPlan.stepCount()} steps, ${parsedPlan.totalEstimatedTokens()} tokens`,
    );
  }

  // 6. Plan metadata with branching
  console.log("\n=== Plan Metadata ===");

  const planMeta = new PlanMetadata(
    "conv-001",
    "Refactor auth module",
    "Step 1: analyze...\nStep 2: implement...",
  );
  planMeta.withModel("claude-sonnet").withIterations(5);
  planMeta.setStatus("active");

  console.log(`  Plan: ${planMeta.title}`);
  console.log(`  Status: ${planMeta.status}, Model: ${planMeta.model_id}`);
  console.log(`  Is root: ${planMeta.isRoot()}`);

  // Create a branch (sub-plan)
  const branch = planMeta.createBranch(
    "auth-refactor-v2",
    "Alternative approach using OAuth",
    "Step 1: setup OAuth...",
  );
  planMeta.addChild(branch.plan_id);

  console.log(`  Branch: ${branch.branch_name}, depth: ${branch.depth}`);
  console.log(`  Parent has children: ${planMeta.hasChildren()}`);

  // Export as markdown
  const markdown = planMeta.toMarkdown();
  console.log(`  Markdown export: ${markdown.split("\n").length} lines`);

  // 7. Content source trust levels
  console.log("\n=== Content Source Trust ===");

  const sources: ContentSource[] = [
    "system_prompt",
    "user_input",
    "agent_reasoning",
    "external_content",
  ];

  for (const source of sources) {
    console.log(
      `  ${source}: needs_sanitization=${requiresSanitization(source)}`,
    );
  }

  console.log("\n  Override matrix (can source override target?):");
  for (const source of sources) {
    const canOverrideList = sources.filter((target) =>
      canOverride(source, target)
    );
    if (canOverrideList.length > 0) {
      console.log(
        `    ${source} can override: [${canOverrideList.join(", ")}]`,
      );
    } else {
      console.log(`    ${source} can override: (none)`);
    }
  }

  // 8. Framework error handling
  console.log("\n=== Error Handling ===");

  const errors = [
    new FrameworkError({ type: "config", message: "Missing API key" }),
    FrameworkError.providerAuth("openai", "Invalid API key"),
    FrameworkError.providerModel("anthropic", "claude-4", "Model not found"),
    FrameworkError.embeddingDimension(384, 768),
    FrameworkError.storageSchema("messages", "Missing 'content' column"),
    new FrameworkError({
      type: "tool_execution",
      message: "Timeout after 30s",
    }),
  ];

  for (const err of errors) {
    console.log(`  [${err.kind.type}] ${err.message}`);
  }

  // Catch and handle by kind
  try {
    throw FrameworkError.providerAuth("demo", "expired token");
  } catch (e: unknown) {
    if (
      e instanceof FrameworkError &&
      (e as FrameworkError).kind.type === "provider_auth"
    ) {
      console.log(
        `\n  Caught provider auth error for: ${
          (e as FrameworkError).kind.provider
        }`,
      );
    }
  }

  console.log(
    "\nDone! Use output parsers for structured LLM extraction and plans for execution budgeting.",
  );
}

await main();
