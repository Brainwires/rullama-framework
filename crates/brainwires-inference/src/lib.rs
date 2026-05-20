//! # brainwires-inference
//!
//! LLM-driven workhorses for the Brainwires Agent Framework.
//!
//! This crate is the home for everything in the framework that drives an
//! LLM call (chat / completion) or constructs prompts for one. It depends
//! on `brainwires-agent` for coordination primitives (locks, message
//! bus, agent lifecycle, runtime context) — coordination is what holds
//! agents together; inference is what they do.
//!
//! Extracted from `brainwires-agent` in Phase 11f to separate the
//! "what holds agents together" (`brainwires-agent`, the coordination
//! crate) from "what makes them think" (this crate).
//!
//! ## Modules
//!
//! - [`chat_agent`] — streaming chat completion loop with per-user session management
//! - [`task_agent`] — autonomous task execution loop
//! - [`planner_agent`] — LLM-powered dynamic task planning
//! - [`judge_agent`] — LLM-powered cycle evaluation
//! - [`validator_agent`] / [`validation_agent`] — LLM-driven validation
//! - [`validation_loop`] — quality-check loop wrapping validation agents
//! - [`cycle_orchestrator`] — Plan → Work → Judge cycle driver
//! - [`plan_executor`] — execution of LLM-generated plans
//! - [`summarization`] — history compaction via LLM
//! - [`system_prompts`] — registry of agent prompt templates

pub mod agent_hooks;
pub mod chat_agent;
pub mod context;
pub mod cycle_orchestrator;
pub mod judge_agent;
pub mod plan_executor;
pub mod planner_agent;
pub mod pool;
pub mod runtime;
pub mod summarization;
pub mod system_prompts;
pub mod task_agent;
pub mod task_orchestrator;
pub mod validation_agent;
pub mod validation_loop;
pub mod validator_agent;

pub use agent_hooks::*;
pub use chat_agent::*;
pub use context::*;
pub use cycle_orchestrator::*;
pub use judge_agent::*;
pub use plan_executor::*;
pub use planner_agent::*;
pub use pool::*;
pub use runtime::*;
pub use summarization::*;
pub use system_prompts::*;
pub use task_agent::*;
pub use task_orchestrator::*;
pub use validation_agent::*;
pub use validation_loop::*;
pub use validator_agent::*;
