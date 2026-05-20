//! # brainwires-eval
//!
//! Evaluation framework for Brainwires agents.
//!
//! ## What's included
//!
//! | Module | Key type | Purpose |
//! |--------|----------|---------|
//! | [`trial`] | [`TrialResult`], [`EvaluationStats`] | Per-trial results + Wilson-score 95 % CI |
//! | [`case`] | [`EvaluationCase`] | Trait for a single evaluatable scenario |
//! | [`suite`] | [`EvaluationSuite`], [`SuiteResult`] | N-trial Monte Carlo runner |
//! | [`recorder`] | [`ToolSequenceRecorder`], [`SequenceDiff`] | Record + diff tool call sequences |
//! | [`adversarial`] | [`AdversarialTestCase`] | Prompt injection, ambiguity, budget stress |

pub mod adversarial;
pub mod case;
pub mod fault_report;
pub mod fixtures;
pub mod ranking_metrics;
pub mod recorder;
pub mod regression;
pub mod stability_tests;
pub mod suite;
pub mod trial;

// ── Top-level re-exports ──────────────────────────────────────────────────────

// Trial types
pub use trial::{ConfidenceInterval95, EvaluationStats, TrialResult};

// Case trait + built-in helpers
pub use case::{AlwaysFailCase, AlwaysPassCase, EvaluationCase, StochasticCase};

// Suite types
pub use suite::{EvaluationSuite, SuiteConfig, SuiteResult};

// Recorder
pub use recorder::{SequenceDiff, ToolCallRecord, ToolSequenceRecorder};

// Adversarial
pub use adversarial::{AdversarialTestCase, AdversarialTestType};

// Regression suite
pub use regression::{
    CategoryBaseline, CategoryRegressionResult, RegressionConfig, RegressionResult, RegressionSuite,
};

// Stability tests
pub use stability_tests::{
    GoalPreservationCase, LoopDetectionSimCase, long_horizon_stability_suite,
};

// Fault report
pub use fault_report::{FaultKind, FaultReport, analyze_suite_for_faults};

// Fixtures
pub use fixtures::{
    Assertion, ExpectedBehavior, Fixture, FixtureCase, FixtureMessage, FixtureRunner, RunOutcome,
    load_fixture_file, load_fixtures_from_dir,
};

// Ranking metrics
pub use ranking_metrics::{mrr, ndcg_at_k, precision_at_k};
