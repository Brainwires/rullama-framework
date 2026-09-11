# @rullama/mdap

Multi-Dimensional Adaptive Planning — the MAKER voting framework for reliable
multi-step agent execution: k-out-of-n consensus voting, red-flag validation of
microagent outputs, recursive task decomposition and composition, and
cost/probability scaling laws.

Equivalent to Rust's `rullama-mdap` crate. Pure TypeScript, no I/O of its own —
you supply the sampler that calls a model.

## Install

```sh
deno add jsr:@rullama/mdap
```

## Quick Example

Vote on the answer to one microagent step. The voter keeps sampling in batches
until one candidate is `k` votes ahead of every other (or an early-stop rule
fires), discarding any sample the red-flag validator rejects.

```ts
import {
  extractResponseConfidence,
  FirstToAheadByKVoter,
  type SampledResponse,
  StandardRedFlagValidator,
} from "@rullama/mdap";

// Replace with a real model call; must be stateless so it can be re-sampled.
async function askModel(prompt: string): Promise<string> {
  await Promise.resolve();
  return `${prompt.length % 2 === 0 ? "42" : "41"}`;
}

const sampler = async (): Promise<SampledResponse<string>> => {
  const started = Date.now();
  const text = await askModel("What is 6 * 7? Answer with the number only.");
  const metadata = {
    tokenCount: text.length,
    responseTimeMs: Date.now() - started,
    formatValid: /^\d+$/.test(text),
    finishReason: "stop",
  };
  return {
    value: text.trim(),
    rawResponse: text,
    metadata,
    confidence: extractResponseConfidence(text, metadata),
  };
};

const voter = FirstToAheadByKVoter.create(3, 20); // k = 3, at most 20 samples
const validator = StandardRedFlagValidator.withFormat({
  kind: "pattern",
  regex: "^\\d+$",
});

const result = await voter.vote(sampler, validator, (v) => v);
console.log(result.winner, result.winnerVotes, "/", result.totalVotes);
```

Before running, estimate what the run will cost and how likely it is to succeed
end-to-end (MAKER paper equations 13–18):

```ts
import { estimateCallCost, estimateMdap, MODEL_COSTS } from "@rullama/mdap";

const perSample = estimateCallCost(MODEL_COSTS.claudeHaiku, 400, 50);
const estimate = estimateMdap(
  /* numSteps */ 12,
  /* perStepSuccessRate */ 0.9,
  /* validResponseRate */ 0.95,
  perSample,
  /* targetSuccessRate */ 0.99,
);
console.log(estimate.recommendedK, estimate.expectedCostUsd);
```

## Components

| Export                                                                                                                              | Description                                                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `FirstToAheadByKVoter`, `VoterBuilder`                                                                                              | Algorithm 2 voting; `"confidence_weighted"` and `"borda_count"` methods, RASC early stopping, loss-of-hope |
| `StandardRedFlagValidator`, `AcceptAllValidator`, `strictRedFlagConfig`, `relaxedRedFlagConfig`                                     | Reject over-long, truncated, mis-formatted, self-correcting or mostly-blank samples                        |
| `outputFormatMatches`, `outputFormatDescription`                                                                                    | `OutputFormat` matching (exact, regex, JSON, JSON-with-fields, markers, one-of)                            |
| `extractResponseConfidence`                                                                                                         | CISC-style confidence heuristic from finish reason, length, hedging and assertion patterns                 |
| `createSubtask`, `createAtomicSubtask`, `atomicDecomposition`, `compositeDecomposition`, `validateDecomposition`, `topologicalSort` | Build, validate and order subtask graphs                                                                   |
| `Composer`                                                                                                                          | Combine subtask outputs: identity, concatenate, sequence, object merge, last-only, reduce, custom handlers |
| `calculateKMin`, `calculatePFull`, `calculateExpectedVotes`, `estimateMdap`, `suggestKForBudget`, `MODEL_COSTS`                     | Scaling laws and cost estimation                                                                           |
| `MdapMetrics`                                                                                                                       | Per-subtask and per-round execution metrics with `summary()` / `redFlagAnalysis()`                         |
| `parseToolIntent`, `toolSchemaToPrompt`, `toolCategoryContains`, `readOnlyCategories`, `sideEffectCategories`                       | Structured tool-call intents for stateless microagents                                                     |
| `defaultEarlyStopping`, `aggressiveEarlyStopping`, `conservativeEarlyStopping`, `disabledEarlyStopping`                             | `EarlyStoppingConfig` presets                                                                              |
| `defaultMicroagentConfig`, `MicroagentProvider`                                                                                     | Microagent execution config and the provider interface a model adapter implements                          |
| `MdapError`                                                                                                                         | Structured error with `isRetryable()`, `isRedFlag()`, `isToolError()`, `isUserError()`                     |

`MdapResult<T>` is a deprecated alias of `T` kept for the Rust port; it is
removed in 0.13.
