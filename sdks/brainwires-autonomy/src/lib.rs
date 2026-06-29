#![deny(missing_docs)]
//! # brainwires-autonomy
//!
//! Autonomous agent operations — self-improvement, Git workflows, and
//! human-out-of-loop execution for the Brainwires Agent Framework.
//!
//! ## Feature flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `self-improve` | Self-improvement controller and strategies |
//! | `eval-driven` | Eval-driven feedback loop (requires `brainwires-eval`) |
//! | `supervisor` | Agent supervisor with health monitoring |
//! | `attention` | Attention mechanism with RAG integration |
//! | `parallel` | Parallel coordinator with optional MDAP |
//! | `training` | Autonomous training loop |
//! | `git-workflow` | Automated Git workflow pipeline |
//! | `webhook` | Webhook server for Git forge events |
//! | `full` | All features enabled |

pub mod config;
pub mod error;
pub mod metrics;
pub mod safety;

#[cfg(feature = "self-improve")]
pub mod self_improve;

pub mod agent_ops;

#[cfg(feature = "eval-driven")]
pub mod eval;

#[cfg(feature = "git-workflow")]
pub mod git_workflow;

/// GPIO hardware control — re-exported from `brainwires-hardware`.
#[cfg(feature = "gpio")]
pub use brainwires_hardware::gpio;

pub use config::AutonomyConfig;
pub use error::AutonomyError;
pub use metrics::{SessionMetrics, SessionReport};
pub use safety::{ApprovalPolicy, AutonomousOperation, SafetyGuard};
