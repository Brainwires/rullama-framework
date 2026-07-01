//! Automated Git workflow pipeline — issue to PR to merge.
//!
//! Provides the full autonomous pipeline: trigger → investigate → branch → fix → PR → merge.
//! Supports GitHub via the [`GitForge`] trait, webhook-driven triggers, configurable
//! merge policies, and CI/CD orchestration.

pub mod branch_manager;
pub mod change_maker;
pub mod forge;
pub mod investigator;
pub mod merge_policy;
pub mod pipeline;
pub mod pr_manager;
pub mod trigger;

#[cfg(feature = "webhook")]
pub mod cicd_orchestrator;
#[cfg(feature = "webhook")]
pub mod webhook;
#[cfg(feature = "webhook")]
pub mod webhook_config;
#[cfg(feature = "webhook")]
pub mod webhook_log;

pub use branch_manager::BranchManager;
pub use change_maker::ChangeMaker;
pub use forge::{CheckStatus, CreatePrParams, GitForge, Issue, MergeMethod, PullRequest};
pub use investigator::{InvestigationResult, IssueInvestigator};
pub use merge_policy::{MergeDecision, MergePolicy};
pub use pipeline::GitWorkflowPipeline;
pub use pr_manager::PullRequestManager;
pub use trigger::{ProgrammaticTrigger, WorkflowEvent, WorkflowTrigger};

#[cfg(feature = "webhook")]
pub use cicd_orchestrator::CiCdOrchestrator;
#[cfg(feature = "webhook")]
pub use webhook::WebhookServer;
#[cfg(feature = "webhook")]
pub use webhook_config::InterpolationContext;
#[cfg(feature = "webhook")]
pub use webhook_log::{WebhookAction, WebhookLogger};
