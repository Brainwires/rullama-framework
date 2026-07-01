/**
 * Permission system for agent capability management.
 *
 * This module provides a comprehensive capability-based permission system for
 * rullama agents, including:
 *
 * - **Capabilities**: Granular control over filesystem, tools, network, git, and spawning
 * - **Profiles**: Pre-defined capability sets (read_only, standard_dev, full_access)
 * - **Configuration**: JSON-based configuration
 * - **Policies**: Rule-based enforcement with conditions and actions
 * - **Audit**: Event logging with querying and statistics
 * - **Trust**: Trust levels, violation tracking, and trust factor management
 * - **Approval**: Interactive approval workflows for sensitive operations
 *
 * Anomaly detection moved to `@rullama/telemetry/anomaly` in v0.11.0
 * to match the Rust restructure.
 *
 * Rust equivalent: `rullama-permissions` crate
 * @module
 */

// ── Types ───────────────────────────────────────────────────────────
export {
  AgentCapabilities,
  ALL_TOOL_CATEGORIES,
  conservativeResourceQuotas,
  defaultFilesystemCapabilities,
  defaultGitCapabilities,
  defaultNetworkCapabilities,
  defaultResourceQuotas,
  defaultSpawningCapabilities,
  defaultToolCapabilities,
  disabledNetworkCapabilities,
  disabledSpawningCapabilities,
  fullFilesystemCapabilities,
  fullGitCapabilities,
  fullNetworkCapabilities,
  fullSpawningCapabilities,
  fullToolCapabilities,
  generousResourceQuotas,
  isDestructiveGitOp,
  parseCapabilityProfile,
  PathPattern,
  readOnlyGitCapabilities,
  standardGitCapabilities,
  standardResourceQuotas,
} from "./types.ts";

export type {
  CapabilityProfile,
  FilesystemCapabilities,
  GitCapabilities,
  GitOperation,
  NetworkCapabilities,
  ResourceQuotas,
  SpawningCapabilities,
  ToolCapabilities,
  ToolCategory,
} from "./types.ts";

// ── Policy ──────────────────────────────────────────────────────────
export {
  conditionMatches,
  createPolicy,
  createPolicyRequest,
  isDecisionAllowed,
  isDecisionRequiresApproval,
  PolicyActions,
  PolicyEngine,
  policyMatches,
  policyRequestForFile,
  policyRequestForGit,
  policyRequestForNetwork,
  policyRequestForTool,
} from "./policy.ts";

export type {
  EnforcementMode,
  Policy,
  PolicyAction,
  PolicyCondition,
  PolicyDecision,
  PolicyRequest,
} from "./policy.ts";

// ── Audit ───────────────────────────────────────────────────────────
export {
  AuditLogger,
  createAuditEvent,
  createAuditQuery,
  queryMatches,
  withAction,
  withAgent,
  withDuration,
  withError,
  withMetadata,
  withOutcome,
  withPolicyDecision,
  withTarget,
  withTrustLevel,
} from "./audit.ts";

export type {
  ActionOutcome,
  AuditEvent,
  AuditEventType,
  AuditQuery,
  AuditStatistics,
  FeedbackPolarity,
  FeedbackSignal,
} from "./audit.ts";

// ── Trust ───────────────────────────────────────────────────────────
export {
  compareTrustLevels,
  createSystemTrustFactor,
  createTrustFactor,
  decayRecentViolations,
  defaultViolationCounts,
  recordViolation,
  trustFactorRecordFailure,
  trustFactorRecordSuccess,
  trustFactorRecordViolation,
  trustFactorReset,
  trustFactorSetLevel,
  trustLevelFromScore,
  trustLevelFromU8,
  trustLevelToU8,
  TrustManager,
  violationPenalty,
  violationRecentPenalty,
  violationsTotalPenalty,
} from "./trust.ts";

export type {
  TrustFactor,
  TrustLevel,
  TrustStatistics,
  ViolationCounts,
  ViolationSeverity,
} from "./trust.ts";

// ── Config ──────────────────────────────────────────────────────────
export {
  configToCapabilities,
  loadPermissionsConfig,
  parseDuration,
  parseGitOperation,
  parseSize,
  parseToolCategory,
} from "./config.ts";

export type {
  DefaultConfig,
  FilesystemConfig,
  GitConfig,
  NetworkConfig,
  PermissionsConfig,
  PoliciesConfig,
  PolicyConditionConfig,
  PolicyRuleConfig,
  QuotasConfig,
  SpawningConfig,
  ToolsConfig,
} from "./config.ts";

// ── Profiles ────────────────────────────────────────────────────────
// Re-exported from types (CapabilityProfile and parseCapabilityProfile are already above)

// ── Approval ────────────────────────────────────────────────────────
export {
  approvalActionCategory,
  approvalActionDescription,
  approvalActionSeverity,
  isApprovalResponseApproved,
  isApprovalResponseSessionPersistent,
} from "./approval.ts";

export type {
  ApprovalAction,
  ApprovalDetails,
  ApprovalRequest,
  ApprovalResponse,
  ApprovalSeverity,
} from "./approval.ts";
