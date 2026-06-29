//! Empirical scoring evaluations for cognition, storage, agent allocation, and reasoning.
//!
//! These eval cases validate that scoring heuristics produce correct relative
//! orderings — not just structural correctness (weights sum to 1.0) but actual
//! ranking quality measured via NDCG, which feeds into the [`AutonomousFeedbackLoop`].
//!
//! ## Usage
//!
//! ```rust,ignore
//! use brainwires_autonomy::eval::{
//!     entity_importance_suite, multi_factor_suite,
//!     agent_scoring_suite, reasoning_eval_suite,
//! };
//!
//! let cases = [
//!     entity_importance_suite(),
//!     multi_factor_suite(),
//!     agent_scoring_suite(),
//!     reasoning_eval_suite(),
//! ].concat();
//! let loop_ = AutonomousFeedbackLoop::new(config, cases, provider);
//! ```
//!
//! The same case list can be passed to `EvalStrategy` inside
//! `SelfImprovementController`.
//!
//! [`AutonomousFeedbackLoop`]: crate::self_improve::AutonomousFeedbackLoop

pub mod agent_eval;
pub mod entity_eval;
pub mod memory_eval;
pub mod reasoning_eval;

pub use agent_eval::{ResourceBidScoringCase, TaskBidScoringCase, agent_scoring_suite};
pub use entity_eval::{
    EntityImportanceRankingCase, EntitySingleMentionCase, EntityTypeBonusCase,
    entity_importance_suite,
};
pub use memory_eval::{MultiFactorRankingCase, TierDemotionCase, multi_factor_suite};
pub use reasoning_eval::{ComplexityHeuristicCase, reasoning_eval_suite};
