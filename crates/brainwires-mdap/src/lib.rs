//! MDAP - MAKER voting framework (merged from `brainwires-mdap`)
//!
//! Multi-Dimensional Adaptive Planning system implementing the MAKER paper's
//! approach to reliable agent execution through:
//!
//! - **Voting**: First-to-ahead-by-k consensus algorithm for error correction
//! - **Microagents**: Minimal context single-step agents (m=1 decomposition)
//! - **Decomposition**: Task decomposition strategies (binary recursive, sequential)
//! - **Red Flags**: Output validation and format checking
//! - **Scaling**: Cost/probability estimation and optimization
//! - **Metrics**: Execution metrics collection and reporting
//! - **Composer**: Result composition from subtask outputs
//! - **Tool Intent**: Structured tool calling intent for stateless execution

pub mod composer;
pub mod decomposition;
pub mod error;
pub mod metrics;
pub mod microagent;
pub mod red_flags;
pub mod scaling;
pub mod tool_intent;
pub mod voting;

// Re-exports
pub use composer::{Composer, CompositionBuilder, StandardComposer};
pub use decomposition::{
    AtomicDecomposer, BinaryRecursiveDecomposer, CompositionFunction, DecomposeContext,
    DecompositionResult, DecompositionStrategy, SequentialDecomposer, SimpleRecursiveDecomposer,
    TaskDecomposer,
};
pub use error::{MdapError, MdapResult};
pub use metrics::MdapMetrics;
pub use microagent::{
    Microagent, MicroagentConfig, MicroagentConfigBuilder, MicroagentProvider, MicroagentResponse,
    Subtask, SubtaskOutput,
};
pub use red_flags::{OutputFormat, RedFlagConfig, StandardRedFlagValidator};
pub use scaling::{MdapEstimate, ModelCosts, estimate_mdap};
pub use tool_intent::{SubtaskOutputWithIntent, ToolCategory, ToolIntent, ToolSchema};
pub use voting::{FirstToAheadByKVoter, ResponseMetadata, SampledResponse, VoteResult};

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::decomposition::{DecomposeContext, DecompositionResult, TaskDecomposer};
    pub use super::error::{MdapError, MdapResult};
    pub use super::microagent::{
        Microagent, MicroagentProvider, MicroagentResponse, Subtask, SubtaskOutput,
    };
    pub use super::red_flags::{OutputFormat, RedFlagConfig};
    pub use super::tool_intent::{ToolCategory, ToolIntent, ToolSchema};
    pub use super::voting::{FirstToAheadByKVoter, SampledResponse, VoteResult};
}
