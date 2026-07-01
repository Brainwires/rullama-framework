/**
 * Policy Engine — Declarative rule-based access control.
 *
 * Provides a flexible policy system for fine-grained permission control beyond
 * static capabilities. Policies can match on tool names, categories, file paths,
 * domains, git operations, and trust levels.
 *
 * Rust equivalent: `rullama-permissions/src/policy.rs`
 * @module
 */

import {
  AgentCapabilities,
  type GitOperation,
  PathPattern,
  type ToolCategory,
} from "./types.ts";

// ── Enforcement Mode ────────────────────────────────────────────────

/**
 * Policy enforcement mode.
 *
 * Rust equivalent: `EnforcementMode` enum (serde `rename_all = "lowercase"`)
 */
export type EnforcementMode = "coercive" | "normative" | "adaptive";

// ── Policy Action ───────────────────────────────────────────────────

/**
 * Action to take when a policy matches.
 *
 * Rust equivalent: `PolicyAction` enum (serde `rename_all = "snake_case"`)
 */
export type PolicyAction =
  | { type: "allow" }
  | { type: "deny" }
  | { type: "require_approval" }
  | { type: "allow_with_audit" }
  | { type: "deny_with_message"; message: string }
  | { type: "escalate" };

/** Convenience constructors matching the Rust enum variants. */
export const PolicyActions = {
  Allow: { type: "allow" } as PolicyAction,
  Deny: { type: "deny" } as PolicyAction,
  RequireApproval: { type: "require_approval" } as PolicyAction,
  AllowWithAudit: { type: "allow_with_audit" } as PolicyAction,
  DenyWithMessage: (message: string): PolicyAction => ({
    type: "deny_with_message",
    message,
  }),
  Escalate: { type: "escalate" } as PolicyAction,
} as const;

// ── Policy Condition ────────────────────────────────────────────────

/**
 * Condition for policy matching.
 *
 * Recursive AND/OR/NOT composition is supported.
 *
 * Rust equivalent: `PolicyCondition` enum (serde `rename_all = "snake_case"`)
 */
export type PolicyCondition =
  | { type: "tool"; name: string }
  | { type: "tool_category"; category: ToolCategory }
  | { type: "file_path"; pattern: string }
  | { type: "min_trust_level"; level: number }
  | { type: "domain"; pattern: string }
  | { type: "git_op"; operation: GitOperation }
  | { type: "time_range"; start_hour: number; end_hour: number }
  | { type: "and"; conditions: PolicyCondition[] }
  | { type: "or"; conditions: PolicyCondition[] }
  | { type: "not"; condition: PolicyCondition }
  | { type: "always" };

/**
 * Check if a condition matches the given request.
 *
 * Rust equivalent: `PolicyCondition::matches()`
 */
export function conditionMatches(
  condition: PolicyCondition,
  request: PolicyRequest,
): boolean {
  switch (condition.type) {
    case "tool":
      return request.tool_name === condition.name;
    case "tool_category":
      return request.tool_category === condition.category;
    case "file_path": {
      if (!request.file_path) return false;
      const pat = new PathPattern(condition.pattern);
      return pat.matches(request.file_path);
    }
    case "min_trust_level":
      return request.trust_level >= condition.level;
    case "domain": {
      if (!request.domain) return false;
      const pat = condition.pattern;
      if (pat.startsWith("*.")) {
        const suffix = pat.slice(1);
        return request.domain.endsWith(suffix) ||
          request.domain === pat.slice(2);
      }
      return request.domain === pat;
    }
    case "git_op":
      return request.git_operation === condition.operation;
    case "time_range": {
      const hour = new Date().getHours();
      if (condition.start_hour <= condition.end_hour) {
        return hour >= condition.start_hour && hour < condition.end_hour;
      }
      return hour >= condition.start_hour || hour < condition.end_hour;
    }
    case "and":
      return condition.conditions.every((c) => conditionMatches(c, request));
    case "or":
      return condition.conditions.some((c) => conditionMatches(c, request));
    case "not":
      return !conditionMatches(condition.condition, request);
    case "always":
      return true;
  }
}

// ── Policy Request ──────────────────────────────────────────────────

/**
 * Request context for policy evaluation.
 *
 * Rust equivalent: `PolicyRequest` struct
 */
export interface PolicyRequest {
  /** Tool being invoked. */
  tool_name: string | undefined;
  /** Tool category. */
  tool_category: ToolCategory | undefined;
  /** File path being accessed. */
  file_path: string | undefined;
  /** Network domain being accessed. */
  domain: string | undefined;
  /** Git operation being performed. */
  git_operation: GitOperation | undefined;
  /** Current trust level (0-4). */
  trust_level: number;
  /** Agent ID making the request. */
  agent_id: string | undefined;
  /** Additional metadata. */
  metadata: Record<string, string>;
}

/**
 * Create a new empty PolicyRequest.
 *
 * Rust equivalent: `PolicyRequest::new()`
 */
export function createPolicyRequest(
  overrides?: Partial<PolicyRequest>,
): PolicyRequest {
  return {
    tool_name: undefined,
    tool_category: undefined,
    file_path: undefined,
    domain: undefined,
    git_operation: undefined,
    trust_level: 0,
    agent_id: undefined,
    metadata: {},
    ...overrides,
  };
}

/**
 * Create a request for a tool invocation.
 *
 * Rust equivalent: `PolicyRequest::for_tool()`
 */
export function policyRequestForTool(toolName: string): PolicyRequest {
  return createPolicyRequest({
    tool_name: toolName,
    tool_category: AgentCapabilities.categorizeTool(toolName),
  });
}

/**
 * Create a request for file access.
 *
 * Rust equivalent: `PolicyRequest::for_file()`
 */
export function policyRequestForFile(
  path: string,
  toolName: string,
): PolicyRequest {
  return createPolicyRequest({
    tool_name: toolName,
    tool_category: AgentCapabilities.categorizeTool(toolName),
    file_path: path,
  });
}

/**
 * Create a request for network access.
 *
 * Rust equivalent: `PolicyRequest::for_network()`
 */
export function policyRequestForNetwork(domain: string): PolicyRequest {
  return createPolicyRequest({
    domain,
    tool_category: "Web",
  });
}

/**
 * Create a request for git operation.
 *
 * Rust equivalent: `PolicyRequest::for_git()`
 */
export function policyRequestForGit(operation: GitOperation): PolicyRequest {
  return createPolicyRequest({
    git_operation: operation,
    tool_category: "Git",
  });
}

// ── Policy Decision ─────────────────────────────────────────────────

/**
 * Policy decision result.
 *
 * Rust equivalent: `PolicyDecision` struct
 */
export interface PolicyDecision {
  /** The action to take. */
  action: PolicyAction;
  /** Policy that made the decision (if any). */
  matched_policy: string | undefined;
  /** Reason for the decision. */
  reason: string | undefined;
  /** Whether this decision should be audited. */
  audit: boolean;
}

/**
 * Check if a decision allows the action.
 *
 * Rust equivalent: `PolicyDecision::is_allowed()`
 */
export function isDecisionAllowed(decision: PolicyDecision): boolean {
  return decision.action.type === "allow" ||
    decision.action.type === "allow_with_audit";
}

/**
 * Check if a decision requires approval.
 *
 * Rust equivalent: `PolicyDecision::requires_approval()`
 */
export function isDecisionRequiresApproval(decision: PolicyDecision): boolean {
  return decision.action.type === "require_approval";
}

// ── Policy ──────────────────────────────────────────────────────────

/**
 * A single policy rule.
 *
 * Rust equivalent: `Policy` struct
 */
export interface Policy {
  /** Unique identifier. */
  id: string;
  /** Human-readable name. */
  name: string;
  /** Description of what this policy does. */
  description: string;
  /** Priority (higher = evaluated first). */
  priority: number;
  /** Conditions that must match. */
  conditions: PolicyCondition[];
  /** Action to take when matched. */
  action: PolicyAction;
  /** Enforcement mode. */
  enforcement: EnforcementMode;
  /** Whether policy is enabled. */
  enabled: boolean;
}

/**
 * Create a new policy with the given ID.
 *
 * Rust equivalent: `Policy::new()`
 */
export function createPolicy(id: string, overrides?: Partial<Policy>): Policy {
  return {
    id,
    name: id,
    description: "",
    priority: 50,
    conditions: [],
    action: PolicyActions.Allow,
    enforcement: "coercive",
    enabled: true,
    ...overrides,
  };
}

/**
 * Check if all conditions of a policy match.
 *
 * Rust equivalent: `Policy::matches()`
 */
export function policyMatches(policy: Policy, request: PolicyRequest): boolean {
  if (!policy.enabled) return false;
  if (policy.conditions.length === 0) return false;
  return policy.conditions.every((c) => conditionMatches(c, request));
}

// ── Policy Engine ───────────────────────────────────────────────────

/**
 * Policy engine for evaluating requests against rules.
 *
 * Rust equivalent: `PolicyEngine` struct
 */
export class PolicyEngine {
  #policies: Policy[] = [];
  #defaultAction: PolicyAction = PolicyActions.Allow;

  /** Create a new empty policy engine. Rust equivalent: `PolicyEngine::new()` */
  constructor() {}

  /**
   * Create a policy engine with default security policies.
   *
   * Rust equivalent: `PolicyEngine::with_defaults()`
   */
  static withDefaults(): PolicyEngine {
    const engine = new PolicyEngine();

    engine.addPolicy(createPolicy("protect_env_files", {
      name: "Protect Environment Files",
      description: "Deny access to .env files which may contain secrets",
      conditions: [{ type: "file_path", pattern: "**/.env*" }],
      action: PolicyActions.Deny,
      priority: 100,
    }));

    engine.addPolicy(createPolicy("protect_secrets", {
      name: "Protect Secret Files",
      description: "Deny access to files containing 'secret' in the path",
      conditions: [{ type: "file_path", pattern: "**/*secret*" }],
      action: PolicyActions.DenyWithMessage(
        "Access to secret files is not permitted",
      ),
      priority: 100,
    }));

    engine.addPolicy(createPolicy("protect_credentials", {
      name: "Protect Credential Files",
      description: "Deny access to credential files",
      conditions: [{ type: "file_path", pattern: "**/credentials*" }],
      action: PolicyActions.Deny,
      priority: 100,
    }));

    engine.addPolicy(createPolicy("approve_git_reset", {
      name: "Approve Git Reset",
      description: "Require approval for git reset operations",
      conditions: [{ type: "git_op", operation: "Reset" }],
      action: PolicyActions.RequireApproval,
      priority: 90,
    }));

    engine.addPolicy(createPolicy("approve_git_rebase", {
      name: "Approve Git Rebase",
      description: "Require approval for git rebase operations",
      conditions: [{ type: "git_op", operation: "Rebase" }],
      action: PolicyActions.RequireApproval,
      priority: 90,
    }));

    engine.addPolicy(createPolicy("audit_bash", {
      name: "Audit Bash Commands",
      description: "Log all bash command executions",
      conditions: [{ type: "tool_category", category: "Bash" }],
      action: PolicyActions.AllowWithAudit,
      priority: 10,
    }));

    return engine;
  }

  /** Set the default action for when no policy matches. Rust equivalent: `PolicyEngine::set_default_action()` */
  setDefaultAction(action: PolicyAction): void {
    this.#defaultAction = action;
  }

  /** Add a policy to the engine. Rust equivalent: `PolicyEngine::add_policy()` */
  addPolicy(policy: Policy): void {
    this.#policies.push(policy);
    this.#policies.sort((a, b) => b.priority - a.priority);
  }

  /** Remove a policy by ID. Rust equivalent: `PolicyEngine::remove_policy()` */
  removePolicy(id: string): Policy | undefined {
    const idx = this.#policies.findIndex((p) => p.id === id);
    if (idx === -1) return undefined;
    return this.#policies.splice(idx, 1)[0];
  }

  /** Get a policy by ID. Rust equivalent: `PolicyEngine::get_policy()` */
  getPolicy(id: string): Policy | undefined {
    return this.#policies.find((p) => p.id === id);
  }

  /** Get all policies. Rust equivalent: `PolicyEngine::policies()` */
  policies(): readonly Policy[] {
    return this.#policies;
  }

  /**
   * Evaluate a request against all policies.
   *
   * Rust equivalent: `PolicyEngine::evaluate()`
   */
  evaluate(request: PolicyRequest): PolicyDecision {
    for (const policy of this.#policies) {
      if (policyMatches(policy, request)) {
        const shouldAudit = policy.action.type === "allow_with_audit" ||
          policy.action.type === "deny" ||
          policy.action.type === "deny_with_message" ||
          policy.action.type === "require_approval";

        return {
          action: policy.action,
          matched_policy: policy.id,
          reason: policy.description || undefined,
          audit: shouldAudit,
        };
      }
    }

    return {
      action: this.#defaultAction,
      matched_policy: undefined,
      reason: undefined,
      audit: false,
    };
  }
}
