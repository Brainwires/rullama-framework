# brainwires-eval

[![Crates.io](https://img.shields.io/crates/v/brainwires-eval.svg)](https://crates.io/crates/brainwires-eval)
[![Documentation](https://docs.rs/brainwires-eval/badge.svg)](https://docs.rs/brainwires-eval)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

Evaluation harness for the Brainwires Agent Framework.

## Overview

A self-contained framework for writing and running evaluation cases
against agents (or anything else). Cases are deterministic where
possible; when they're not, ranking metrics measure quality.

Originally an internal-only `brainwires-eval` module, then folded
into `brainwires-agent`, re-extracted in 0.11 (Phase 11e) so the
framework's evaluation surface is its own dependency-free crate.
Zero `brainwires-*` deps internally.

## Modules

- `case` — `EvaluationCase` trait
- `trial` — `TrialResult` + `EvaluationStats`
- `suite` — `EvaluationSuite` + `SuiteResult` (Monte Carlo runner
  with Wilson confidence intervals)
- `fixtures` — YAML-based fixture cases (`tests/fixtures/*.yaml`)
- `regression` — `RegressionSuite` for change-detection runs
- `stability_tests` — flakiness detection across repeated runs
- `adversarial` — adversarial case generation
- `recorder` — recording trial results to disk
- `fault_report` — structured fault reports
- `ranking_metrics` — `ndcg_at_k`, `mrr`, `precision_at_k` (with
  graded relevance support)

## Migration from `brainwires-agent::eval`

```toml
# Before
brainwires-agent = { features = ["eval"] }

# After
brainwires-eval = "0.11"
```

```rust
// Before
use brainwires_agent::eval::{EvaluationCase, TrialResult, ndcg_at_k};

// After
use brainwires_eval::{EvaluationCase, TrialResult, ndcg_at_k};
```

## License

MIT OR Apache-2.0
