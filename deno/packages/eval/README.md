# @rullama/eval

Evaluation harness for LLM agents: an N-trial Monte Carlo suite runner with
Wilson-score 95% confidence intervals, `EvaluationCase` interface and built-in
cases, a tool-call sequence recorder with diffing, adversarial test templates
(prompt injection, ambiguity, missing context, budget exhaustion), long-horizon
stability cases, regression gating for CI, fault analysis and ranking metrics
(NDCG@K, MRR, precision@K). No `@rullama/*` dependencies.

Extracted from the old `@rullama/agents` package in v0.11.0 to mirror Rust's
`rullama-eval` crate.

## Install

```sh
deno add jsr:@rullama/eval
```

## Quick Example

```ts
import {
  AlwaysPassCase,
  type EvaluationCase,
  EvaluationSuite,
  failingCases,
  overallSuccessRate,
  StochasticCase,
  trialFailure,
  trialSuccess,
} from "@rullama/eval";

// Your code under test -- an agent call, a tool, a prompt...
const myAgent = (input: string) => Promise.resolve(`hello, ${input}`);

// A case wraps whatever you want to measure; run(trial) returns a TrialResult.
class GreetsUser implements EvaluationCase {
  name() {
    return "greets_user";
  }
  category() {
    return "smoke";
  }
  async run(trial: number) {
    const start = performance.now();
    const output = await myAgent(`trial ${trial}`);
    const ms = performance.now() - start;
    return output.includes("hello")
      ? trialSuccess(trial, ms)
      : trialFailure(trial, ms, "no greeting");
  }
}

const suite = new EvaluationSuite(20); // 20 trials per case
const result = await suite.runSuite([
  new GreetsUser(),
  new AlwaysPassCase("baseline"),
  new StochasticCase("flaky", 0.7),
]);

console.log(`overall: ${(overallSuccessRate(result) * 100).toFixed(1)}%`);
console.log("below 90%:", failingCases(result, 0.9));
for (const [name, stats] of Object.entries(result.stats)) {
  console.log(name, stats.success_rate, stats.confidence_interval);
}
```

## What else is here

- `ToolSequenceRecorder` + `computeSequenceDiff` -- record and diff the tool
  calls an agent makes.
- `standardAdversarialSuite()` / `promptInjectionCase` / … -- payload templates
  you wire to your own `EvaluationCase`.
- `longHorizonStabilitySuite()` -- loop-detection and goal-preservation
  simulations, no provider needed.
- `RegressionSuite` + `isCiPassing` -- compare a `SuiteResult` against stored
  per-category baselines.
- `analyzeSuiteForFaults` -- turn failures into prioritized `FaultReport`s.
- `ndcgAtK`, `mrr`, `precisionAtK` -- pure ranking metrics.
- `loadFixturesFromDir` / `FixtureRunner` -- JSON fixture-driven cases.
