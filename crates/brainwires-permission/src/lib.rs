#![deny(missing_docs)]
//! Permission system for agent capability management
//!
//! This crate provides a comprehensive capability-based permission system for
//! brainwires agents, including:
//!
//! - **Capabilities**: Granular control over filesystem, tools, network, git, and spawning
//! - **Profiles**: Pre-defined capability sets (read_only, standard_dev, full_access)
//! - **Configuration**: TOML-based configuration via ~/.brainwires/permissions.toml
//! - **Policies**: Rule-based enforcement with conditions and actions
//! - **Audit**: Event logging with querying and statistics
//! - **Trust**: Trust levels, violation tracking, and trust factor management

pub mod approval;
pub mod audit;
pub mod config;
pub mod policy;
pub mod profiles;
pub mod trust;
pub mod types;

// Re-export main types for convenience
pub use config::PermissionsConfig;
#[cfg(feature = "native")]
pub use config::{default_permissions_path, ensure_permissions_dir};
pub use profiles::CapabilityProfile;
pub use types::{
    AgentCapabilities, FilesystemCapabilities, GitCapabilities, GitOperation, NetworkCapabilities,
    PathPattern, ResourceQuotas, SpawningCapabilities, ToolCapabilities, ToolCategory,
};

// Re-export policy types
pub use policy::{
    EnforcementMode, Policy, PolicyAction, PolicyCondition, PolicyDecision, PolicyEngine,
    PolicyRequest,
};

// Re-export audit types
pub use audit::{
    ActionOutcome, AuditEvent, AuditEventType, AuditLogger, AuditQuery, AuditStatistics,
    FeedbackPolarity, FeedbackSignal,
};

// Anomaly detection lives in `brainwires-telemetry::anomaly`. Depend on
// brainwires-telemetry directly:
//   use brainwires_telemetry::anomaly::{AnomalyConfig, AnomalyDetector, ...};

// Re-export trust types
pub use trust::{TrustFactor, TrustLevel, TrustManager, TrustStatistics, ViolationSeverity};

// Re-export approval types
pub use approval::{
    ApprovalAction, ApprovalDetails, ApprovalRequest, ApprovalResponse, ApprovalSeverity,
};
