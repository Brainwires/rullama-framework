# MDAP — Multi-Dimensional Adaptive Planning

> This module was previously the standalone `brainwires-mdap` crate. It now lives in `brainwires-agent` behind the `mdap` feature flag.

MAKER voting framework — microagents, decomposition, red flags, and scaling for the Brainwires Agent Framework.

## Paper

This crate is a Rust implementation of **MAKER** (Multi-Agent K-consensus Error correction) as described in:

> **Solving a Million-Step LLM Task with Zero Errors**
> Elliot Meyerson, Giuseppe Paolo, Roberto Dailey, Hormoz Shahrzad, Olivier Francon, Conor F. Hayes, Xin Qiu, Babak Hodjat, Risto Miikkulainen
> arXiv:2511.09030, November 2025
> https://arxiv.org/abs/2511.09030

The paper introduces **Massively Decomposed Agentic Processes (MDAPs)** — a scaling approach that decomposes tasks into minimal subtasks handled by focused microagents, with multi-agent voting for error correction at every step. MAKER achieves zero-error execution on million-step tasks through extreme decomposition, first-to-ahead-by-k voting, and red-flag output validation.

This implementation also integrates techniques from three supplementary papers:

- **RASC** ([arXiv:2408.17017](https://arxiv.org/abs/2408.17017)) — early stopping with variance tracking and loss-of-hope detection
- **CISC** ([arXiv:2502.06233v1](https://arxiv.org/abs/2502.06233v1)) — confidence-weighted voting with dynamic confidence extraction
- **Ranked Voting** ([arXiv:2505.10772](https://arxiv.org/abs/2505.10772)) — Borda count ranking as an alternative consensus method

## Overview

This module provides a complete implementation of the MAKER framework organized around five core components that map directly to the paper's algorithms and equations:

- **Voting** (Algorithm 2) — first-to-ahead-by-k consensus with three voting methods, early stopping, and confidence weighting
- **Microagents** (MAD) — focused single-step agents (m=1) that execute one subtask with minimal context
- **Red Flags** (Algorithm 3) — output validation that catches self-correction, confused reasoning, truncation, and format violations
- **Decomposition** (Algorithm 4) — binary recursive task decomposition with AI-driven splitting and dependency resolution
- **Scaling Laws** (Equations 13–19) — cost and probability estimation for choosing optimal k given a budget or reliability target

**Design principles:**

- **Paper-faithful** — algorithms, equations, and heuristics follow the MAKER paper directly
- **Composable** — each component is independent; use voting without decomposition, red flags without microagents, etc.
- **Provider-agnostic** — generic over `MicroagentProvider` trait; works with any LLM backend
- **Intent-based tool use** — microagents express tool *intent* (deterministic) rather than executing tools (non-deterministic), preserving voting correctness
- **Full observability** — per-subtask metrics, voting round tracking, red-flag breakdowns, and cost analysis

```text
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    brainwires-agent::mdap                            │
  │                                                                      │
  │  Task ──► Decomposition (Alg.4) ──► Subtask DAG                      │
  │                                         │                            │
  │               ┌─────────────────────────┘                            │
  │               ▼                                                      │
  │  ┌─── Per Subtask ──────────────────────────────────────────────┐    │
  │  │                                                              │    │
  │  │  Microagent ──► Sample k responses ──► Red Flags (Alg.3)     │    │
  │  │  (m=1 steps)        │                      │                 │    │
  │  │                     ▼                      ▼                 │    │
  │  │              Valid responses ──► Voting (Alg.2)              │    │
  │  │                                      │                       │    │
  │  │              ┌───────────────────────┘                       │    │
  │  │              ▼                                               │    │
  │  │  Winner + VoteResult + SubtaskMetric                         │    │
  │  └──────────────────────────────────────────────────────────────┘    │
  │               │                                                      │
  │               ▼                                                      │
  │  Composer ──► Final result        Scaling (Eq.13-19) ──► Estimates   │
  │                                   Metrics ──► Cost/performance data  │
  └──────────────────────────────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires-mdap = "0.11"
```

Run a simple voting consensus:

```rust
use brainwires_mdap::{
    FirstToAheadByKVoter, SampledResponse, ResponseMetadata,
    StandardRedFlagValidator, RedFlagValidator,
};

// Create a voter with k=3, max 20 samples
let voter = FirstToAheadByKVoter::new(3, 20);

// Red-flag validator
let validator = StandardRedFlagValidator::strict();

// Vote with a sampler function that queries an LLM
let result = voter.vote(
    || async {
        let response = call_my_llm("What is 2+2?").await?;
        Ok(SampledResponse::new(
            response.text.clone(),
            ResponseMetadata {
                token_count: response.tokens,
                response_time_ms: response.latency,
                format_valid: true,
                finish_reason: Some("stop".into()),
                model: Some("claude-sonnet".into()),
            },
            response.text,
        ))
    },
    &validator,
).await?;

println!("Winner: {} (votes: {}/{})", result.winner, result.winner_votes, result.total_votes);
```

Estimate cost before running:

```rust
use brainwires_mdap::{estimate_mdap, ModelCosts};

let estimate = estimate_mdap(
    10,                           // num_steps (subtasks)
    0.85,                         // per-step success probability
    0.99,                         // target overall success rate
    &ModelCosts::claude_sonnet(), // model pricing
    500,                          // avg input tokens per call
    200,                          // avg output tokens per call
)?;

println!("Recommended k={}, cost=${:.4}, P(success)={:.4}",
    estimate.recommended_k,
    estimate.expected_cost_usd,
    estimate.success_probability,
);
```

## Architecture

### Voting (Algorithm 2)

The core consensus mechanism. Multiple independent LLM samples are collected and the first answer to lead by k votes wins.

#### `FirstToAheadByKVoter`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `k` | `usize` | — | Votes-ahead margin required to declare a winner |
| `max_samples` | `usize` | — | Maximum samples before giving up |
| `parallel_limit` | `usize` | `k` | Max concurrent samples |
| `batch_size` | `usize` | `k` | Samples per batch |
| `early_stopping` | `EarlyStoppingConfig` | disabled | RASC-style early stopping |
| `voting_method` | `VotingMethod` | `FirstToAheadByK` | Consensus algorithm |
| `use_confidence_weights` | `bool` | `false` | CISC confidence weighting |

**Constructors:**

| Method | Description |
|--------|-------------|
| `new(k, max_samples)` | Standard first-to-ahead-by-k voting |
| `with_early_stopping(k, max_samples, config)` | With RASC early stopping |
| `with_confidence_weighting(k, max_samples)` | CISC confidence-weighted voting |
| `with_borda_count(k, max_samples)` | Ranked voting via Borda count |

**Methods:**

| Method | Description |
|--------|-------------|
| `vote(sampler, validator)` | Execute voting — samples via `sampler`, validates via `validator`, returns `VoteResult` |
| `vote_simple(sampler)` | Simplified voting without red-flag validation |

**`VoterBuilder`** — fluent builder: `VoterBuilder::new().k(3).max_samples(20).voting_method(BordaCount).build()`

#### `VotingMethod`

| Variant | Paper | Description |
|---------|-------|-------------|
| `FirstToAheadByK` | MAKER Alg. 2 | First answer to lead by k votes wins (default) |
| `BordaCount` | [arXiv:2505.10772](https://arxiv.org/abs/2505.10772) | Ranked voting based on confidence scores |
| `ConfidenceWeighted` | [arXiv:2502.06233v1](https://arxiv.org/abs/2502.06233v1) | Votes weighted by response confidence |

#### `EarlyStoppingConfig`

RASC-style early stopping ([arXiv:2408.17017](https://arxiv.org/abs/2408.17017)) to reduce unnecessary samples when consensus is already clear.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_confidence` | `f64` | — | Confidence threshold to stop early |
| `min_votes` | `usize` | — | Minimum votes before early stopping is eligible |
| `enabled` | `bool` | `true` | Master toggle |
| `max_variance_threshold` | `f64` | `0.1` | Maximum vote distribution variance to trigger stop |
| `loss_of_hope_enabled` | `bool` | `true` | Stop if no candidate can possibly win |
| `min_weighted_confidence` | `f64` | `0.0` | Minimum weighted confidence for CISC |

**Presets:** `aggressive()` (stop fast), `conservative()` (higher confidence required), `disabled()`

#### `VoteResult<T>`

| Field | Type | Description |
|-------|------|-------------|
| `winner` | `T` | The winning answer |
| `winner_votes` | `usize` | Votes for the winner |
| `total_votes` | `usize` | Total valid votes cast |
| `total_samples` | `usize` | Total samples including red-flagged |
| `red_flagged_count` | `usize` | Samples that failed red-flag validation |
| `vote_distribution` | `HashMap<String, usize>` | Vote counts per unique answer |
| `confidence` | `f64` | Voting confidence score |
| `red_flag_reasons` | `Vec<String>` | Reasons for red-flagged responses |
| `early_stopped` | `bool` | Whether early stopping triggered |
| `weighted_confidence` | `Option<f64>` | CISC weighted confidence |
| `voting_method` | `VotingMethod` | Which method was used |

#### `SampledResponse<T>`

| Field | Type | Description |
|-------|------|-------------|
| `value` | `T` | The parsed/hashed response value |
| `metadata` | `ResponseMetadata` | Token count, timing, format validity, finish reason, model |
| `raw_response` | `String` | Original LLM response text |
| `confidence` | `f64` | Response confidence (for CISC weighting) |

### Microagents (MAD)

Maximal Agentic Decomposition — each microagent executes exactly one subtask (m=1 step) with a focused system prompt that discourages hedging, self-correction, and unnecessary explanation.

#### `Subtask`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `String` | Unique subtask identifier |
| `description` | `String` | What to do |
| `input_state` | `Value` | Input data from prior subtasks |
| `expected_output_format` | `Option<OutputFormat>` | Expected output format for red-flag validation |
| `depends_on` | `Vec<String>` | Subtask IDs that must complete first |
| `complexity_estimate` | `f32` | Estimated difficulty (0.0–1.0) |
| `instructions` | `Option<String>` | Additional instructions |

**Constructors:** `atomic(id, description)`, `new(id, description, input_state)`

#### `MicroagentConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_output_tokens` | `usize` | `750` | Token limit (per paper recommendation) |
| `temperature` | `f32` | `0.1` | Low temperature for consistency |
| `system_prompt_template` | `Option<String>` | Paper default | Custom system prompt |
| `red_flag_config` | `RedFlagConfig` | strict | Red-flag validation settings |
| `timeout_ms` | `Option<u64>` | `None` | Execution timeout |

#### `MicroagentProvider` trait

```rust
#[async_trait]
pub trait MicroagentProvider: Send + Sync {
    async fn chat(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> MdapResult<MicroagentResponse>;

    fn available_tools(&self) -> Vec<ToolSchema> { vec![] }
    fn has_tools(&self) -> bool { false }
}
```

#### `Microagent<P>`

| Method | Description |
|--------|-------------|
| `new(subtask, provider, config)` | Create a microagent for a specific subtask |
| `with_defaults(subtask, provider)` | Create with default config |
| `execute_once()` | Single LLM call — returns `SampledResponse` for voting |
| `execute_with_voting(voter, validator)` | Execute with voting consensus — returns `VoteResult` |

#### Confidence Extraction

`extract_response_confidence(text, metadata)` computes a 0.1–0.99 confidence score by analyzing:

- Finish reason (`stop` vs `length`)
- Response length relative to token limit
- Hedging language ("maybe", "perhaps", "I think")
- Self-correction patterns ("Wait,", "Actually,", "Let me reconsider")
- Confident assertions ("definitely", "clearly", "the answer is")
- Format validity

### Red Flags (Algorithm 3)

Output validation that catches unreliable LLM responses before they enter voting.

#### `RedFlagConfig`

| Field | Type | Default (strict) | Description |
|-------|------|-------------------|-------------|
| `max_response_tokens` | `usize` | `750` | Maximum response length |
| `require_exact_format` | `bool` | `true` | Enforce expected output format |
| `flag_self_correction` | `bool` | `true` | Flag "Wait,", "Actually,", etc. |
| `confusion_patterns` | `Vec<String>` | 10 patterns | Regex patterns indicating confused reasoning |
| `min_response_length` | `usize` | `1` | Minimum response length |
| `max_empty_line_ratio` | `f64` | `0.5` | Maximum empty line ratio |

**Presets:** `strict()` (paper-recommended), `relaxed()` (fewer false positives)

**Strict confusion patterns:** `"Wait,"`, `"Actually,"`, `"Let me reconsider"`, `"I made a mistake"`, `"On second thought"`, `"Hmm,"`, `"I think I"`, `"Let me correct"`, `"Sorry, I meant"`, `"That's not right"`

#### `StandardRedFlagValidator`

Implements Algorithm 3. Validation checks (in order):

1. **Length** — token count vs max, minimum length
2. **Self-correction** — confusion pattern regex matching
3. **Format** — expected output format matching
4. **Truncation** — finish reason analysis
5. **Empty lines** — empty line ratio

#### `RedFlagResult`

| Variant | Description |
|---------|-------------|
| `Valid` | Response passed all checks |
| `Flagged { reason, severity }` | Response failed validation with reason and 0.0–1.0 severity |

#### `OutputFormat`

| Variant | Description |
|---------|-------------|
| `Exact(String)` | Must match exactly |
| `Pattern(String)` | Must match regex |
| `Json` | Must be valid JSON |
| `JsonWithFields(Vec<String>)` | JSON with required fields |
| `Markers { start, end }` | Must contain start/end markers |
| `OneOf(Vec<String>)` | Must be one of the enumerated values |
| `Custom { description, validator_id }` | Custom validation logic |

#### Other Validators

- `AcceptAllValidator` — always returns `Valid` (useful for testing)
- `CompositeValidator` — chains multiple validators; first failure wins

### Decomposition (Algorithm 4)

Breaks complex tasks into a DAG of minimal subtasks.

#### `DecomposeContext`

| Field | Type | Description |
|-------|------|-------------|
| `working_directory` | `Option<String>` | Working directory for file operations |
| `available_tools` | `Vec<ToolSchema>` | Tools available to microagents |
| `max_depth` | `usize` | Maximum recursion depth |
| `current_depth` | `usize` | Current recursion depth |
| `additional_context` | `Option<String>` | Extra context for the decomposer |

#### `DecompositionStrategy`

| Variant | Description |
|---------|-------------|
| `BinaryRecursive { max_depth }` | Paper's approach — AI-driven binary splitting (Algorithm 4) |
| `Simple { max_depth }` | Text-based splitting (testing only) |
| `Sequential` | Linear step extraction |
| `CodeOperations` | Code-specific decomposition |
| `AIDriven { discriminator_k }` | AI splitting with discriminator voting |
| `None` | Atomic — no decomposition |

#### `DecompositionResult`

| Field | Type | Description |
|-------|------|-------------|
| `subtasks` | `Vec<Subtask>` | Ordered subtask list |
| `composition_function` | `CompositionFunction` | How to combine results |
| `is_minimal` | `bool` | Whether the task was already minimal |
| `total_complexity` | `f32` | Sum of subtask complexities |

#### `TaskDecomposer` trait

```rust
#[async_trait]
pub trait TaskDecomposer: Send + Sync {
    async fn decompose(&self, task: &str, context: &DecomposeContext)
        -> MdapResult<DecompositionResult>;
    fn is_minimal(&self, task: &str) -> bool;
    fn strategy(&self) -> DecompositionStrategy;
}
```

#### `BinaryRecursiveDecomposer<P>`

AI-driven implementation of Algorithm 4. Uses the LLM with voting (k consensus) to decide how to split each task, recursing until subtasks are minimal.

**Minimal task heuristics:** very short (< 50 chars), single-action verbs (`return`, `calculate`, `get`, `set`, `check`, etc.), no multi-step conjunctions.

#### `SequentialDecomposer`

Non-AI decomposer that extracts numbered steps or splits by sentences. Useful for pre-structured tasks.

#### Utilities

- `validate_decomposition(result)` — checks non-empty, valid dependencies, no circular references
- `topological_sort(subtasks)` — Kahn's algorithm for dependency ordering

### Composer

Combines subtask outputs into a final result.

#### `CompositionFunction`

| Variant | Description |
|---------|-------------|
| `Identity` | Return single result as-is |
| `Concatenate` | Join as strings |
| `Sequence` | Collect into JSON array |
| `ObjectMerge` | Merge into JSON object |
| `LastOnly` | Take the last result |
| `Custom(String)` | Custom handler by name |
| `Reduce { operation }` | Reduce: `sum`, `multiply`, `max`, `min`, `and`, `or`, `concat` |

#### `Composer`

| Method | Description |
|--------|-------------|
| `new()` | Create an empty composer |
| `register_handler(name, handler)` | Register a custom `CompositionHandler` |
| `compose(function, outputs)` | Compose subtask outputs using the given function |

#### `CompositionBuilder`

Fluent builder with input validation: `CompositionBuilder::new(function).add_result(output).compose()`

### Tool Intent

Microagents express tool *intent* without executing — this keeps voting deterministic since tool execution has side effects.

#### `ToolSchema`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `String` | Tool name |
| `description` | `String` | What the tool does |
| `parameters` | `HashMap<String, String>` | Parameter names and descriptions |
| `required` | `Vec<String>` | Required parameters |
| `category` | `Option<ToolCategory>` | Tool classification |

Converts from `brainwires_core::Tool` via `From` trait.

#### `ToolIntent`

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `String` | Which tool to call |
| `arguments` | `Value` | Tool arguments as JSON |
| `rationale` | `Option<String>` | Why this tool is needed |

#### `ToolCategory`

| Variant | Side Effects | Description |
|---------|-------------|-------------|
| `FileRead` | No | Read files |
| `FileWrite` | Yes | Write/edit files |
| `Search` | No | File/text search |
| `SemanticSearch` | No | Embedding-based search |
| `Bash` | Yes | Shell commands |
| `Git` | Yes | Git operations |
| `Web` | No | HTTP requests |
| `AgentPool` | Yes | Agent management |
| `TaskManager` | Yes | Task management |
| `Mcp` | Yes | MCP server tools |
| `Custom(String)` | — | Custom category |

`read_only_categories()` returns categories safe for microagents. `side_effect_categories()` returns categories that modify state.

#### Intent Parsing

`parse_tool_intent(response)` extracts `ToolIntent` from LLM responses containing `tool_intent` JSON blocks. Returns `IntentParseResult::NoIntent`, `WithIntent`, or `ParseError`.

### Scaling Laws (Equations 13–19)

Cost and probability estimation from the paper's mathematical framework.

#### `estimate_mdap()`

Main estimation function implementing Equations 13–19:

```rust
pub fn estimate_mdap(
    num_steps: usize,         // s: number of subtasks
    per_step_success: f64,    // p: per-step success probability (must be > 0.5)
    target_success: f64,      // target overall success rate (0, 1)
    model_costs: &ModelCosts, // pricing per 1K tokens
    avg_input_tokens: usize,  // average input tokens per call
    avg_output_tokens: usize, // average output tokens per call
) -> MdapResult<MdapEstimate>
```

#### Key Equations

| Function | Equation | Formula |
|----------|----------|---------|
| `calculate_p_full(p, k, s)` | Eq. 13 | `P_full = (1 + ((1-p)/p)^k)^(-s)` |
| `calculate_k_min(p, s, target)` | Eq. 14 | `k_min = ceil(ln(t^(-1/s) - 1) / ln((1-p)/p))` |
| `calculate_expected_votes(p, k)` | — | `E[votes] ≈ k / (2p - 1)` |
| `calculate_expected_cost(...)` | Eq. 19 | `E[cost] ≈ c·s·k / (v·(2p-1))` |

#### `ModelCosts`

| Preset | Input/1K | Output/1K |
|--------|----------|-----------|
| `claude_sonnet()` | $0.003 | $0.015 |
| `claude_haiku()` | $0.00025 | $0.00125 |
| `gpt4o()` | $0.0025 | $0.01 |
| `gpt4o_mini()` | $0.00015 | $0.0006 |

#### `MdapEstimate`

| Field | Type | Description |
|-------|------|-------------|
| `expected_cost_usd` | `f64` | Estimated total cost |
| `expected_api_calls` | `usize` | Estimated total API calls |
| `success_probability` | `f64` | Overall success probability |
| `recommended_k` | `usize` | Minimum k for target success rate |
| `estimated_time_seconds` | `f64` | Estimated wall-clock time |
| `per_step_success` | `f64` | Per-step success probability used |
| `num_steps` | `usize` | Number of subtasks |

#### `suggest_k_for_budget()`

Budget-constrained k selection — finds the largest k affordable within a dollar budget.

### Metrics

Full observability into MDAP execution.

#### `MdapMetrics`

Comprehensive metrics covering execution, steps, sampling, voting, cost, time, and success rate.

| Method | Description |
|--------|-------------|
| `new(execution_id)` | Create new metrics tracker |
| `with_config(config_summary)` | Attach configuration snapshot |
| `start()` | Record start time |
| `finalize(success)` | Record end time and success |
| `record_subtask(metric)` | Record per-subtask metrics |
| `record_voting_round(metric)` | Record per-round metrics |
| `add_sample_cost(input_tokens, output_tokens, cost)` | Accumulate cost |
| `summary()` | Human-readable summary string |
| `red_flag_analysis()` | Red-flag breakdown string |
| `to_json()` / `from_json()` | Serialization |

#### `SubtaskMetric`

| Field | Type | Description |
|-------|------|-------------|
| `subtask_id` | `String` | Which subtask |
| `description` | `String` | Subtask description |
| `samples_needed` | `usize` | Samples taken to reach consensus |
| `red_flags_hit` | `usize` | Red-flagged samples |
| `red_flag_reasons` | `Vec<String>` | Why samples were flagged |
| `final_confidence` | `f64` | Voting confidence |
| `execution_time_ms` | `u64` | Wall-clock time |
| `winner_votes` / `total_votes` | `usize` | Vote counts |
| `succeeded` | `bool` | Whether the subtask succeeded |
| `input_tokens` / `output_tokens` | `usize` | Token usage |
| `complexity_estimate` | `f32` | Subtask complexity |

### Error Handling

`MdapError` is a comprehensive error enum with sub-error types for each component:

| Variant | Sub-errors | Description |
|---------|-----------|-------------|
| `Voting(VotingError)` | MaxSamplesExceeded, AllSamplesRedFlagged, InvalidK, etc. | Voting failures |
| `RedFlag(RedFlagError)` | ResponseTooLong, SelfCorrectionDetected, InvalidJson, etc. | Validation failures |
| `Decomposition(DecompositionError)` | MaxDepthExceeded, CircularDependency, etc. | Decomposition failures |
| `Microagent(MicroagentError)` | ExecutionFailed, Timeout, ContextTooLarge, etc. | Execution failures |
| `Composition(CompositionError)` | MissingResult, IncompatibleTypes, etc. | Composition failures |
| `Scaling(ScalingError)` | InvalidSuccessProbability, VotingCannotConverge, etc. | Estimation failures |
| `Config(MdapConfigError)` | InvalidK, InvalidTargetSuccessRate, etc. | Configuration errors |
| `ToolRecursionLimit` | — | Tool intent recursion exceeded |
| `ToolExecutionFailed` | — | Tool execution failure |
| `ToolNotAllowed` | — | Tool not permitted for microagent |

**Helper methods:** `is_retryable()`, `is_user_error()`, `is_tool_error()`, `is_red_flag()`, `should_restart_voting()`

## Usage Examples

### Voting with red-flag validation

```rust
use brainwires_mdap::prelude::*;

let voter = FirstToAheadByKVoter::new(3, 20);
let validator = StandardRedFlagValidator::strict();

let result = voter.vote(
    || async { sample_llm_response().await },
    &validator,
).await?;

println!("Winner: {}", result.winner);
println!("Confidence: {:.2}", result.confidence);
println!("Red-flagged: {}", result.red_flagged_count);
```

### Voting with early stopping (RASC)

```rust
use brainwires_mdap::{FirstToAheadByKVoter, EarlyStoppingConfig};

let voter = FirstToAheadByKVoter::with_early_stopping(
    3,
    20,
    EarlyStoppingConfig::aggressive(),
);

let result = voter.vote(sampler, &validator).await?;
if result.early_stopped {
    println!("Stopped early at {} samples", result.total_samples);
}
```

### Confidence-weighted voting (CISC)

```rust
use brainwires_mdap::FirstToAheadByKVoter;

let voter = FirstToAheadByKVoter::with_confidence_weighting(3, 20);

let result = voter.vote(sampler, &validator).await?;
println!("Weighted confidence: {:?}", result.weighted_confidence);
```

### Builder pattern for voter configuration

```rust
use brainwires_mdap::{VoterBuilder, VotingMethod, EarlyStoppingConfig};

let voter = VoterBuilder::new()
    .k(5)
    .max_samples(30)
    .voting_method(VotingMethod::BordaCount)
    .early_stopping(EarlyStoppingConfig::conservative())
    .parallel_limit(4)
    .build()?;
```

### Cost estimation before execution

```rust
use brainwires_mdap::{estimate_mdap, ModelCosts, suggest_k_for_budget};

// What k do I need for 99% success on 10 steps?
let estimate = estimate_mdap(10, 0.85, 0.99, &ModelCosts::claude_sonnet(), 500, 200)?;
println!("Need k={}, cost=${:.4}", estimate.recommended_k, estimate.expected_cost_usd);

// What k can I afford with $0.50?
let k = suggest_k_for_budget(10, 0.85, &ModelCosts::claude_haiku(), 500, 200, 0.50)?;
println!("Budget allows k={}", k);
```

### Task decomposition

```rust
use brainwires_mdap::{
    decomposition::{BinaryRecursiveDecomposer, DecomposeContext, validate_decomposition, topological_sort},
};

let decomposer = BinaryRecursiveDecomposer::new(provider, 4, 3, 15);
let context = DecomposeContext::new().with_max_depth(4);

let result = decomposer.decompose("Implement an LRU cache with get/put", &context).await?;
validate_decomposition(&result)?;

let ordered = topological_sort(&result.subtasks)?;
for subtask in &ordered {
    println!("{}: {}", subtask.id, subtask.description);
}
```

### Composing subtask results

```rust
use brainwires_mdap::{Composer, CompositionFunction, SubtaskOutput};

let composer = Composer::new();

let outputs = vec![
    SubtaskOutput::new("step-1", serde_json::json!("struct definition")),
    SubtaskOutput::new("step-2", serde_json::json!("impl block")),
    SubtaskOutput::new("step-3", serde_json::json!("tests")),
];

let final_result = composer.compose(&CompositionFunction::Concatenate, &outputs)?;
```

### Tracking metrics

```rust
use brainwires_mdap::{MdapMetrics, SubtaskMetric, ConfigSummary};

let mut metrics = MdapMetrics::new("exec-001".into());
metrics.start();

metrics.record_subtask(SubtaskMetric {
    subtask_id: "step-1".into(),
    description: "Parse input".into(),
    samples_needed: 5,
    red_flags_hit: 1,
    final_confidence: 0.95,
    succeeded: true,
    // ... other fields
    ..Default::default()
});

metrics.finalize(true);
println!("{}", metrics.summary());
println!("{}", metrics.red_flag_analysis());
```

## Integration

Use via the `brainwires` facade crate with the `mdap` feature, or depend on `brainwires-agent` directly:

```toml
# Via facade
[dependencies]
brainwires = { version = "0.11", features = ["mdap"] }

# Direct
[dependencies]
brainwires-mdap = "0.11"
```

The `prelude` module re-exports the most commonly used types:

```rust
use brainwires_mdap::prelude::*;
```

## License

Licensed under the MIT License. See [LICENSE](../../LICENSE) for details.
