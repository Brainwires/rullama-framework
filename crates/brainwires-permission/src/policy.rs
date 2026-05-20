//! Policy Engine - Declarative rule-based access control
//!
//! Provides a flexible policy system for fine-grained permission control beyond
//! static capabilities. Policies can match on tool names, categories, file paths,
//! domains, git operations, and trust levels.
//!
//! # Example
//!
//! ```rust,ignore
//! use brainwires::permissions::policy::{Policy, PolicyEngine, PolicyCondition, PolicyAction};
//!
//! let mut engine = PolicyEngine::new();
//!
//! // Deny access to secret files
//! engine.add_policy(Policy::new("protect_secrets")
//!     .with_condition(PolicyCondition::FilePath("**/.env*".into()))
//!     .with_action(PolicyAction::Deny)
//!     .with_priority(100));
//!
//! // Require approval for destructive git operations
//! engine.add_policy(Policy::new("approve_git_reset")
//!     .with_condition(PolicyCondition::GitOp(GitOperation::Reset))
//!     .with_action(PolicyAction::RequireApproval)
//!     .with_priority(90));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{GitOperation, PathPattern, ToolCategory};

/// Policy enforcement mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EnforcementMode {
    /// Hard enforcement - policy decisions cannot be overridden
    #[default]
    Coercive,
    /// Soft enforcement - decisions can be overridden with justification
    Normative,
    /// Adaptive enforcement - learns from overrides to adjust policy
    Adaptive,
}

/// Action to take when a policy matches
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum PolicyAction {
    /// Allow the action
    #[default]
    Allow,
    /// Deny the action
    Deny,
    /// Require explicit user approval
    RequireApproval,
    /// Allow but log for audit
    AllowWithAudit,
    /// Deny with a custom message
    DenyWithMessage(String),
    /// Escalate to a higher authority (e.g., orchestrator)
    Escalate,
}

/// Condition for policy matching
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyCondition {
    /// Match specific tool by name
    Tool(String),
    /// Match tools by category
    ToolCategory(ToolCategory),
    /// Match file path pattern
    FilePath(String),
    /// Match trust level (minimum required)
    MinTrustLevel(u8),
    /// Match network domain
    Domain(String),
    /// Match git operation
    GitOp(GitOperation),
    /// Match time range (hour of day, 0-23)
    TimeRange {
        /// Start hour (0-23).
        start_hour: u8,
        /// End hour (0-23).
        end_hour: u8,
    },
    /// Logical AND of conditions
    And(Vec<PolicyCondition>),
    /// Logical OR of conditions
    Or(Vec<PolicyCondition>),
    /// Logical NOT of condition
    Not(Box<PolicyCondition>),
    /// Always matches
    Always,
}

impl PolicyCondition {
    /// Check if this condition matches the given request
    pub fn matches(&self, request: &PolicyRequest) -> bool {
        match self {
            PolicyCondition::Tool(name) => request.tool_name.as_ref() == Some(name),
            PolicyCondition::ToolCategory(cat) => request.tool_category.as_ref() == Some(cat),
            PolicyCondition::FilePath(pattern) => {
                if let Some(path) = &request.file_path {
                    let pat = PathPattern::new(pattern);
                    pat.matches(path)
                } else {
                    false
                }
            }
            PolicyCondition::MinTrustLevel(level) => request.trust_level >= *level,
            PolicyCondition::Domain(pattern) => {
                if let Some(domain) = &request.domain {
                    if pattern.starts_with("*.") {
                        let suffix = &pattern[1..]; // ".example.com"
                        domain.ends_with(suffix) || domain == &pattern[2..]
                    } else {
                        domain == pattern
                    }
                } else {
                    false
                }
            }
            PolicyCondition::GitOp(op) => request.git_operation.as_ref() == Some(op),
            PolicyCondition::TimeRange {
                start_hour,
                end_hour,
            } => {
                let hour = chrono::Local::now().hour() as u8;
                if start_hour <= end_hour {
                    hour >= *start_hour && hour < *end_hour
                } else {
                    // Wraps around midnight
                    hour >= *start_hour || hour < *end_hour
                }
            }
            PolicyCondition::And(conditions) => conditions.iter().all(|c| c.matches(request)),
            PolicyCondition::Or(conditions) => conditions.iter().any(|c| c.matches(request)),
            PolicyCondition::Not(condition) => !condition.matches(request),
            PolicyCondition::Always => true,
        }
    }
}

use chrono::Timelike;

/// Request context for policy evaluation
#[derive(Debug, Clone, Default)]
pub struct PolicyRequest {
    /// Tool being invoked
    pub tool_name: Option<String>,
    /// Tool category
    pub tool_category: Option<ToolCategory>,
    /// File path being accessed
    pub file_path: Option<String>,
    /// Network domain being accessed
    pub domain: Option<String>,
    /// Git operation being performed
    pub git_operation: Option<GitOperation>,
    /// Current trust level (0-4)
    pub trust_level: u8,
    /// Agent ID making the request
    pub agent_id: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl PolicyRequest {
    /// Create a new empty request
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a request for a tool invocation
    pub fn for_tool(tool_name: &str) -> Self {
        let category = super::AgentCapabilities::categorize_tool(tool_name);
        Self {
            tool_name: Some(tool_name.to_string()),
            tool_category: Some(category),
            ..Default::default()
        }
    }

    /// Create a request for file access
    pub fn for_file(path: &str, tool_name: &str) -> Self {
        let category = super::AgentCapabilities::categorize_tool(tool_name);
        Self {
            tool_name: Some(tool_name.to_string()),
            tool_category: Some(category),
            file_path: Some(path.to_string()),
            ..Default::default()
        }
    }

    /// Create a request for network access
    pub fn for_network(domain: &str) -> Self {
        Self {
            domain: Some(domain.to_string()),
            tool_category: Some(ToolCategory::Web),
            ..Default::default()
        }
    }

    /// Create a request for git operation
    pub fn for_git(operation: GitOperation) -> Self {
        Self {
            git_operation: Some(operation),
            tool_category: Some(ToolCategory::Git),
            ..Default::default()
        }
    }

    /// Set trust level
    pub fn with_trust_level(mut self, level: u8) -> Self {
        self.trust_level = level;
        self
    }

    /// Set agent ID
    pub fn with_agent_id(mut self, id: &str) -> Self {
        self.agent_id = Some(id.to_string());
        self
    }
}

/// Policy decision result
#[derive(Debug, Clone)]
pub struct PolicyDecision {
    /// The action to take
    pub action: PolicyAction,
    /// Policy that made the decision (if any)
    pub matched_policy: Option<String>,
    /// Reason for the decision
    pub reason: Option<String>,
    /// Whether this decision should be audited
    pub audit: bool,
}

impl PolicyDecision {
    /// Create an allow decision
    pub fn allow() -> Self {
        Self {
            action: PolicyAction::Allow,
            matched_policy: None,
            reason: None,
            audit: false,
        }
    }

    /// Create a deny decision
    pub fn deny(reason: &str) -> Self {
        Self {
            action: PolicyAction::Deny,
            matched_policy: None,
            reason: Some(reason.to_string()),
            audit: true,
        }
    }

    /// Check if the decision allows the action
    pub fn is_allowed(&self) -> bool {
        matches!(
            self.action,
            PolicyAction::Allow | PolicyAction::AllowWithAudit
        )
    }

    /// Check if the decision requires approval
    pub fn requires_approval(&self) -> bool {
        matches!(self.action, PolicyAction::RequireApproval)
    }
}

/// A single policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what this policy does
    #[serde(default)]
    pub description: String,
    /// Priority (higher = evaluated first)
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Conditions that must match
    pub conditions: Vec<PolicyCondition>,
    /// Action to take when matched
    pub action: PolicyAction,
    /// Enforcement mode
    #[serde(default)]
    pub enforcement: EnforcementMode,
    /// Whether policy is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_priority() -> i32 {
    50
}

fn default_true() -> bool {
    true
}

impl Policy {
    /// Create a new policy with the given ID
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            name: id.to_string(),
            description: String::new(),
            priority: 50,
            conditions: Vec::new(),
            action: PolicyAction::Allow,
            enforcement: EnforcementMode::Coercive,
            enabled: true,
        }
    }

    /// Set the policy name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set the policy description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a condition
    pub fn with_condition(mut self, condition: PolicyCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Set the action
    pub fn with_action(mut self, action: PolicyAction) -> Self {
        self.action = action;
        self
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the enforcement mode
    pub fn with_enforcement(mut self, mode: EnforcementMode) -> Self {
        self.enforcement = mode;
        self
    }

    /// Check if all conditions match
    pub fn matches(&self, request: &PolicyRequest) -> bool {
        if !self.enabled {
            return false;
        }
        if self.conditions.is_empty() {
            return false; // No conditions = no match (safety default)
        }
        self.conditions.iter().all(|c| c.matches(request))
    }
}

/// Policy engine for evaluating requests against rules
#[derive(Debug, Clone, Default)]
pub struct PolicyEngine {
    /// Registered policies, sorted by priority
    policies: Vec<Policy>,
    /// Default action when no policy matches
    default_action: PolicyAction,
}

impl PolicyEngine {
    /// Create a new empty policy engine
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
            default_action: PolicyAction::Allow,
        }
    }

    /// Create a policy engine with default security policies
    pub fn with_defaults() -> Self {
        let mut engine = Self::new();

        // Protect sensitive files
        engine.add_policy(
            Policy::new("protect_env_files")
                .with_name("Protect Environment Files")
                .with_description("Deny access to .env files which may contain secrets")
                .with_condition(PolicyCondition::FilePath("**/.env*".into()))
                .with_action(PolicyAction::Deny)
                .with_priority(100),
        );

        engine.add_policy(
            Policy::new("protect_secrets")
                .with_name("Protect Secret Files")
                .with_description("Deny access to files containing 'secret' in the path")
                .with_condition(PolicyCondition::FilePath("**/*secret*".into()))
                .with_action(PolicyAction::DenyWithMessage(
                    "Access to secret files is not permitted".into(),
                ))
                .with_priority(100),
        );

        engine.add_policy(
            Policy::new("protect_credentials")
                .with_name("Protect Credential Files")
                .with_description("Deny access to credential files")
                .with_condition(PolicyCondition::FilePath("**/credentials*".into()))
                .with_action(PolicyAction::Deny)
                .with_priority(100),
        );

        // Require approval for destructive git operations
        engine.add_policy(
            Policy::new("approve_git_reset")
                .with_name("Approve Git Reset")
                .with_description("Require approval for git reset operations")
                .with_condition(PolicyCondition::GitOp(GitOperation::Reset))
                .with_action(PolicyAction::RequireApproval)
                .with_priority(90),
        );

        engine.add_policy(
            Policy::new("approve_git_rebase")
                .with_name("Approve Git Rebase")
                .with_description("Require approval for git rebase operations")
                .with_condition(PolicyCondition::GitOp(GitOperation::Rebase))
                .with_action(PolicyAction::RequireApproval)
                .with_priority(90),
        );

        // Audit bash commands
        engine.add_policy(
            Policy::new("audit_bash")
                .with_name("Audit Bash Commands")
                .with_description("Log all bash command executions")
                .with_condition(PolicyCondition::ToolCategory(ToolCategory::Bash))
                .with_action(PolicyAction::AllowWithAudit)
                .with_priority(10),
        );

        engine
    }

    /// Set the default action for when no policy matches
    pub fn set_default_action(&mut self, action: PolicyAction) {
        self.default_action = action;
    }

    /// Add a policy to the engine
    pub fn add_policy(&mut self, policy: Policy) {
        self.policies.push(policy);
        // Keep sorted by priority (highest first)
        self.policies.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Remove a policy by ID
    pub fn remove_policy(&mut self, id: &str) -> Option<Policy> {
        if let Some(pos) = self.policies.iter().position(|p| p.id == id) {
            Some(self.policies.remove(pos))
        } else {
            None
        }
    }

    /// Get a policy by ID
    pub fn get_policy(&self, id: &str) -> Option<&Policy> {
        self.policies.iter().find(|p| p.id == id)
    }

    /// Get all policies
    pub fn policies(&self) -> &[Policy] {
        &self.policies
    }

    /// Evaluate a request against all policies
    pub fn evaluate(&self, request: &PolicyRequest) -> PolicyDecision {
        // Find first matching policy (already sorted by priority)
        for policy in &self.policies {
            if policy.matches(request) {
                return PolicyDecision {
                    action: policy.action.clone(),
                    matched_policy: Some(policy.id.clone()),
                    reason: if policy.description.is_empty() {
                        None
                    } else {
                        Some(policy.description.clone())
                    },
                    audit: matches!(
                        policy.action,
                        PolicyAction::AllowWithAudit
                            | PolicyAction::Deny
                            | PolicyAction::DenyWithMessage(_)
                            | PolicyAction::RequireApproval
                    ),
                };
            }
        }

        // No policy matched, use default
        PolicyDecision {
            action: self.default_action.clone(),
            matched_policy: None,
            reason: None,
            audit: false,
        }
    }

    /// Load policies from TOML config
    pub fn load_from_config(config: &super::config::PermissionsConfig) -> Self {
        let mut engine = Self::new();

        // Load policies from config
        for rule in &config.policies.rules {
            if let Some(policy) = Self::parse_policy_rule(rule) {
                engine.add_policy(policy);
            }
        }

        engine
    }

    /// Parse a policy rule from config
    fn parse_policy_rule(rule: &super::config::PolicyRuleConfig) -> Option<Policy> {
        let mut policy = Policy::new(&rule.name)
            .with_name(&rule.name)
            .with_priority(rule.priority as i32);

        // Parse conditions using the conversion
        for condition in rule.get_conditions() {
            if let Some(cond) = Self::parse_condition(&condition) {
                policy = policy.with_condition(cond);
            }
        }

        // Parse action
        let action = match rule.action.to_lowercase().as_str() {
            "allow" => PolicyAction::Allow,
            "deny" => PolicyAction::Deny,
            "requireapproval" | "require_approval" => PolicyAction::RequireApproval,
            "allowwithaudit" | "allow_with_audit" => PolicyAction::AllowWithAudit,
            "escalate" => PolicyAction::Escalate,
            _ => {
                if rule.action.starts_with("DenyWithMessage:") {
                    PolicyAction::DenyWithMessage(rule.action[16..].to_string())
                } else {
                    PolicyAction::Deny
                }
            }
        };
        policy = policy.with_action(action);

        // Parse enforcement
        let mode = match rule.enforcement.to_lowercase().as_str() {
            "coercive" => EnforcementMode::Coercive,
            "normative" => EnforcementMode::Normative,
            "adaptive" => EnforcementMode::Adaptive,
            _ => EnforcementMode::Coercive,
        };
        policy = policy.with_enforcement(mode);

        Some(policy)
    }

    /// Parse a condition from config
    fn parse_condition(condition: &super::config::PolicyCondition) -> Option<PolicyCondition> {
        if let Some(tool) = &condition.tool {
            return Some(PolicyCondition::Tool(tool.clone()));
        }
        if let Some(category) = &condition.tool_category
            && let Some(cat) = super::config::parse_tool_category(category)
        {
            return Some(PolicyCondition::ToolCategory(cat));
        }
        if let Some(path) = &condition.file_path {
            return Some(PolicyCondition::FilePath(path.clone()));
        }
        if let Some(domain) = &condition.domain {
            return Some(PolicyCondition::Domain(domain.clone()));
        }
        if let Some(git_op) = &condition.git_op
            && let Some(op) = super::config::parse_git_operation(git_op)
        {
            return Some(PolicyCondition::GitOp(op));
        }
        if let Some(trust) = condition.min_trust_level {
            return Some(PolicyCondition::MinTrustLevel(trust));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_matching() {
        let policy = Policy::new("test")
            .with_condition(PolicyCondition::Tool("write_file".into()))
            .with_action(PolicyAction::Deny);

        let request = PolicyRequest::for_tool("write_file");
        assert!(policy.matches(&request));

        let request2 = PolicyRequest::for_tool("read_file");
        assert!(!policy.matches(&request2));
    }

    #[test]
    fn test_file_path_condition() {
        let policy = Policy::new("test")
            .with_condition(PolicyCondition::FilePath("**/.env*".into()))
            .with_action(PolicyAction::Deny);

        let request = PolicyRequest::for_file(".env", "read_file");
        assert!(policy.matches(&request));

        let request2 = PolicyRequest::for_file(".env.local", "read_file");
        assert!(policy.matches(&request2));

        let request3 = PolicyRequest::for_file("src/main.rs", "read_file");
        assert!(!policy.matches(&request3));
    }

    #[test]
    fn test_domain_condition() {
        let cond = PolicyCondition::Domain("*.github.com".into());

        let request = PolicyRequest::for_network("api.github.com");
        assert!(cond.matches(&request));

        let request2 = PolicyRequest::for_network("github.com");
        assert!(cond.matches(&request2));

        let request3 = PolicyRequest::for_network("evil.com");
        assert!(!cond.matches(&request3));
    }

    #[test]
    fn test_compound_conditions() {
        let cond = PolicyCondition::And(vec![
            PolicyCondition::ToolCategory(ToolCategory::FileWrite),
            PolicyCondition::FilePath("**/test/**".into()),
        ]);

        let mut request = PolicyRequest::for_file("src/test/file.rs", "write_file");
        request.tool_category = Some(ToolCategory::FileWrite);
        assert!(cond.matches(&request));

        let mut request2 = PolicyRequest::for_file("src/main.rs", "write_file");
        request2.tool_category = Some(ToolCategory::FileWrite);
        assert!(!cond.matches(&request2));
    }

    #[test]
    fn test_policy_engine_evaluation() {
        let mut engine = PolicyEngine::new();

        engine.add_policy(
            Policy::new("deny_secrets")
                .with_condition(PolicyCondition::FilePath("**/.env*".into()))
                .with_action(PolicyAction::Deny)
                .with_priority(100),
        );

        engine.add_policy(
            Policy::new("allow_read")
                .with_condition(PolicyCondition::ToolCategory(ToolCategory::FileRead))
                .with_action(PolicyAction::Allow)
                .with_priority(10),
        );

        // Should be denied (higher priority)
        let request = PolicyRequest::for_file(".env", "read_file");
        let decision = engine.evaluate(&request);
        assert!(!decision.is_allowed());
        assert_eq!(decision.matched_policy, Some("deny_secrets".to_string()));

        // Should be allowed
        let request2 = PolicyRequest::for_file("src/main.rs", "read_file");
        let mut request2 = request2;
        request2.tool_category = Some(ToolCategory::FileRead);
        let decision2 = engine.evaluate(&request2);
        assert!(decision2.is_allowed());
    }

    #[test]
    fn test_trust_level_condition() {
        let policy = Policy::new("require_trust")
            .with_condition(PolicyCondition::MinTrustLevel(2))
            .with_action(PolicyAction::Allow);

        let low_trust = PolicyRequest::new().with_trust_level(1);
        assert!(!policy.matches(&low_trust));

        let high_trust = PolicyRequest::new().with_trust_level(3);
        assert!(policy.matches(&high_trust));
    }

    #[test]
    fn test_git_operation_condition() {
        let policy = Policy::new("approve_reset")
            .with_condition(PolicyCondition::GitOp(GitOperation::Reset))
            .with_action(PolicyAction::RequireApproval);

        let request = PolicyRequest::for_git(GitOperation::Reset);
        assert!(policy.matches(&request));

        let request2 = PolicyRequest::for_git(GitOperation::Commit);
        assert!(!policy.matches(&request2));
    }

    #[test]
    fn test_default_policies() {
        let engine = PolicyEngine::with_defaults();

        // Should deny .env files
        let request = PolicyRequest::for_file(".env", "read_file");
        let decision = engine.evaluate(&request);
        assert!(!decision.is_allowed());

        // Should require approval for git reset
        let request2 = PolicyRequest::for_git(GitOperation::Reset);
        let decision2 = engine.evaluate(&request2);
        assert!(decision2.requires_approval());
    }

    #[test]
    fn test_not_condition() {
        let cond = PolicyCondition::Not(Box::new(PolicyCondition::Tool("read_file".into())));

        let request = PolicyRequest::for_tool("write_file");
        assert!(cond.matches(&request));

        let request2 = PolicyRequest::for_tool("read_file");
        assert!(!cond.matches(&request2));
    }

    #[test]
    fn test_or_condition() {
        let cond = PolicyCondition::Or(vec![
            PolicyCondition::Tool("write_file".into()),
            PolicyCondition::Tool("delete_file".into()),
        ]);

        let request = PolicyRequest::for_tool("write_file");
        assert!(cond.matches(&request));

        let request2 = PolicyRequest::for_tool("delete_file");
        assert!(cond.matches(&request2));

        let request3 = PolicyRequest::for_tool("read_file");
        assert!(!cond.matches(&request3));
    }
}
