#![deny(missing_docs)]
//! Brainwires Agents - Agent orchestration, coordination, and lifecycle management
//!
//! This crate provides the multi-agent infrastructure for autonomous task execution:
//!
//! ## Core Components
//! - **CommunicationHub** - Inter-agent messaging bus with 50+ message types
//! - **FileLockManager** - File access coordination with deadlock detection
//! - **ResourceLockManager** - Scoped resource locking with heartbeat-based liveness
//! - **OperationTracker** - Operation tracking with heartbeat-based liveness checking
//! - **ValidationLoop** - Quality checks before agent completion (Bug #5 prevention)
//! - **TaskManager** - Hierarchical task decomposition and dependency tracking
//! - **TaskQueue** - Priority-based task scheduling with dependency awareness
//!
//! ## Coordination Patterns
//! - **ContractNet** - Bidding protocol for agent negotiation
//! - **Saga** - Compensating transactions for distributed operations
//! - **OptimisticConcurrency** - Optimistic locking with version-based conflict detection
//! - **WaitQueue** - Queue-based coordination primitives
//! - **MarketAllocation** - Market-based task allocation
//! - **ThreeStateModel** - State snapshots for rollback support
//!
//! ## Analysis & Validation
//! - **ResourceChecker** - Conflict detection and resolution
//! - **ValidationAgent** - Rule-based validation
//! - **Confidence** - Response confidence scoring
//! - **WorktreeManager** - Git worktree management for agent isolation
//!
//! ## Feature Flags
//! - `tools` - Enable validation tool integration (check_duplicates, verify_build, check_syntax)

// Re-export core types
pub use brainwires_core;

// Re-export the tool runtime for ToolExecutor / ToolRegistry trait surface.
pub use brainwires_tool_runtime;

// ── LLM-driven workhorses moved to brainwires-inference in Phase 11f ─────────
// chat_agent, summarization, planner_agent, judge_agent, validator_agent,
// validation_agent, validation_loop, cycle_orchestrator, plan_executor,
// system_prompts, task_agent — see crates/brainwires-inference/

// ── Personas (pluggable system-prompt assembly) ──────────────────────────────

pub mod personas;

// agent_hooks moved to brainwires-inference in Phase 11f (TaskAgent-coupled).

// runtime + context moved to brainwires-inference in Phase 11f (the
// AgentRuntime drives the inference workhorses; AgentContext owns the
// AgentLifecycleHooks trait object).

// ── Schema + lifecycle ───────────────────────────────────────────────────────

pub mod execution_graph;
// pool moved to brainwires-inference in Phase 11f (TaskAgent pool, not generic).
pub mod roles;

// ── Core components ──────────────────────────────────────────────────────────

pub mod communication;
// `confidence` moved to `brainwires-core` in Phase 11a; the agent-side
// shim was dropped in Phase 11g. Use `brainwires_core::confidence::*`
// directly.
pub mod file_locks;
pub mod operation_tracker;
pub mod resource_locks;
pub mod task_manager;
pub mod task_queue;

// ── Coordination patterns ────────────────────────────────────────────────────

pub mod contract_net;
pub mod market_allocation;
pub mod optimistic;
pub mod saga;
pub mod state_model;
pub mod wait_queue;

// ── Access control ─────────────────────────────────────────────────────────

pub mod access_control;

// ── Agent management (lifecycle trait + MCP tool registry) ─────────────────
//
// Moved out of brainwires-network in Phase 2 — both modules import only
// brainwires_core/serde/anyhow/async_trait, so they belong here with the
// rest of the agent-runtime surface.

/// Agent lifecycle management — `AgentManager` trait + `SpawnConfig`.
pub mod agent_manager;
/// Pre-built MCP tools for agent operations — `AgentToolRegistry`.
pub mod agent_tools;

pub use agent_manager::{AgentInfo, AgentManager, AgentResult, SpawnConfig};
pub use agent_tools::AgentToolRegistry;

// ── Git coordination ───────────────────────────────────────────────────────

pub mod git_coordination;

// plan_executor moved to brainwires-inference in Phase 11f.

// task_orchestrator moved to brainwires-inference in Phase 11f
// (TaskAgent-coupled, not a generic orchestrator).

// ── Workflow graph builder ───────────────────────────────────────────────────

pub mod workflow;

// ── OpenTelemetry export ─────────────────────────────────────────────────────
#[cfg(feature = "otel")]
pub mod otel;

// Eval — extracted to its own brainwires-eval crate in Phase 11e.

// MDAP — extracted to its own brainwires-mdap crate in Phase 11b.

// SEAL — extracted to its own brainwires-seal crate in Phase 11d.

// Skills — extracted to its own brainwires-skills crate in Phase 11c.

// ── Analysis ────────────────────────────────────────────────────────────────

pub mod resource_checker;
// validation_agent + validation_loop moved to brainwires-inference in Phase 11f.
#[cfg(feature = "native")]
pub mod worktree;

// ── Re-exports ───────────────────────────────────────────────────────────────

// agent_hooks re-exports moved to brainwires-inference in Phase 11f
// (the trait references TaskAgentResult, which is TaskAgent-specific).

// AgentRuntime + run_agent_loop re-exports moved to brainwires-inference in
// Phase 11f.

// Core components
pub use communication::{
    AgentMessage, CommunicationHub, ConflictInfo, ConflictType, GitOperationType,
};
// confidence types moved to brainwires-core in Phase 11a; re-export shim
// dropped in Phase 11g. Import from `brainwires_core::confidence::*` directly.
pub use file_locks::{FileLockManager, LockType};
pub use operation_tracker::OperationTracker;
pub use resource_checker::{ConflictCheck, ResourceChecker};
pub use resource_locks::{
    ResourceLockGuard, ResourceLockManager, ResourceScope, ResourceType as ResourceLockType,
};
pub use task_manager::{TaskManager, format_duration_secs};
pub use task_queue::TaskQueue;
#[cfg(feature = "native")]
pub use worktree::WorktreeManager;

// Access control
pub use access_control::{AccessControlManager, ContentionStrategy, LockBundle, LockPersistence};

// Git coordination
pub use git_coordination::{
    GitCoordinator, GitLockRequirements, GitOperationLocks, GitOperationRunner,
    get_lock_requirements, git_tools,
};

// Task orchestration re-exports moved to brainwires-inference in Phase 11f
// (TaskAgent-coupled).

// Workflow graph builder
pub use workflow::{WorkflowBuilder, WorkflowContext, WorkflowResult};

// Coordination patterns
pub use contract_net::ContractNetManager;
pub use market_allocation::MarketAllocator;
pub use optimistic::OptimisticController;
pub use saga::SagaExecutor;
pub use state_model::{StateModelProposedOperation, StateSnapshot, ThreeStateModel};
pub use wait_queue::WaitQueue;

// Schema + lifecycle (AgentContext re-export moved to brainwires-inference in 11f)
pub use brainwires_tool_runtime::{PreHookDecision, ToolPreHook};
pub use execution_graph::{ExecutionGraph, RunTelemetry, StepNode, ToolCallRecord};
// pool re-exports moved to brainwires-inference in Phase 11f.

// SEAL re-exports — extracted to brainwires-seal in Phase 11d.
// LLM-driven workhorses (chat_agent, planner_agent, judge_agent, validator_agent,
// validation_agent, validation_loop, cycle_orchestrator, plan_executor,
// summarization, system_prompts, task_agent) — extracted to brainwires-inference
// in Phase 11f. Import from `brainwires_inference::*` directly or via
// `brainwires::inference::*` (facade).

/// Prelude module for convenient imports — coordination + patterns + schema.
///
/// LLM-driven workhorses (`ChatAgent`, `TaskAgent`, planner/judge/validator,
/// validation_loop, cycle_orchestrator, plan_executor, system_prompts,
/// summarization) live in `brainwires-inference` since Phase 11f. Import
/// from there directly or use the umbrella `brainwires::inference` module.
pub mod prelude {
    // Schema + lifecycle (AgentContext lives in brainwires-inference now)
    pub use super::execution_graph::{ExecutionGraph, RunTelemetry, StepNode, ToolCallRecord};
    pub use brainwires_tool_runtime::{PreHookDecision, ToolPreHook};

    // Core components
    pub use super::communication::{AgentMessage, CommunicationHub, ConflictInfo, ConflictType};
    pub use super::file_locks::{FileLockManager, LockType};
    pub use super::operation_tracker::OperationTracker;
    pub use super::resource_checker::{ConflictCheck, ResourceChecker};
    pub use super::resource_locks::{ResourceLockManager, ResourceScope};
    pub use super::state_model::{StateSnapshot, ThreeStateModel};
    pub use super::task_manager::{TaskManager, format_duration_secs};
    pub use super::task_queue::TaskQueue;
    #[cfg(feature = "native")]
    pub use super::worktree::WorktreeManager;
    pub use brainwires_core::confidence::{ConfidenceFactors, ResponseConfidence};

    // Access control
    pub use super::access_control::{AccessControlManager, ContentionStrategy, LockPersistence};

    // Git coordination
    pub use super::git_coordination::{GitCoordinator, git_tools};

    // Workflow graph builder
    pub use super::workflow::{WorkflowBuilder, WorkflowContext, WorkflowResult};

    // Coordination patterns
    pub use super::contract_net::ContractNetManager;
    pub use super::market_allocation::MarketAllocator;
    pub use super::optimistic::OptimisticController;
    pub use super::saga::SagaExecutor;
    pub use super::wait_queue::WaitQueue;
}
