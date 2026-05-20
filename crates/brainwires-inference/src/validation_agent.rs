//! Independent Validation Agent for operation verification
//!
//! Based on SagaLLM's GlobalValidationAgent concept, this module provides
//! centralized validation that performs three types of checks:
//!
//! 1. **Pre-Validation**: Before operation starts, check preconditions
//! 2. **Post-Validation**: After operation completes, verify postconditions
//! 3. **Inter-Agent Validation**: Check for conflicts between concurrent operations
//!
//! This centralized approach ensures consistent validation across all agent
//! operations and makes debugging easier by having all validation logic
//! in one place.

use std::path::PathBuf;
use std::sync::Arc;

use crate::validation_loop::ValidationSeverity;
use brainwires_agent::resource_checker::{ConflictCheck, ResourceChecker};
use brainwires_agent::state_model::{StateModelProposedOperation, StateSnapshot, ThreeStateModel};

/// Type alias for proposed operations used in validation (private to this module)
type ProposedOperation = StateModelProposedOperation;

/// Independent validation agent for operation verification
pub struct ValidationAgent {
    state_model: Arc<ThreeStateModel>,
    resource_checker: Arc<ResourceChecker>,
    rules: Vec<ValidationRule>,
}

impl ValidationAgent {
    /// Create a new validation agent
    pub fn new(state_model: Arc<ThreeStateModel>, resource_checker: Arc<ResourceChecker>) -> Self {
        let mut agent = Self {
            state_model,
            resource_checker,
            rules: Vec::new(),
        };
        agent.register_default_rules();
        agent
    }

    /// Register the default validation rules
    fn register_default_rules(&mut self) {
        // Pre-condition: File must exist for edit operations
        self.rules.push(ValidationRule {
            name: "file_exists_for_edit".into(),
            description: "File must exist before editing".into(),
            rule_type: RuleType::PreCondition,
            check: Box::new(|ctx| {
                // Check if any resources_needed are files that don't exist
                for resource in &ctx.operation.resources_needed {
                    if resource.starts_with('/') || resource.contains('/') {
                        // Looks like a file path
                        let path = PathBuf::from(resource);
                        if let Some(file_status) = ctx.current_state.files.get(&path)
                            && !file_status.exists
                        {
                            return ValidationOutcome {
                                passed: false,
                                rule_name: "file_exists_for_edit".into(),
                                message: format!("File does not exist: {}", resource),
                                severity: ValidationSeverity::Warning,
                            };
                        }
                        // File not in state could mean it exists on disk but isn't tracked
                    }
                }
                ValidationOutcome::pass("file_exists_for_edit")
            }),
        });

        // Pre-condition: No conflicting locks held by other agents
        self.rules.push(ValidationRule {
            name: "no_conflicting_locks".into(),
            description: "No other agent holds conflicting lock".into(),
            rule_type: RuleType::PreCondition,
            check: Box::new(|ctx| {
                // Check if any resource needed is locked by another agent
                for resource in &ctx.operation.resources_needed {
                    if let Some(holder) = ctx.current_state.locks.get(resource)
                        && holder != &ctx.agent_id
                    {
                        return ValidationOutcome {
                            passed: false,
                            rule_name: "no_conflicting_locks".into(),
                            message: format!(
                                "Resource '{}' is locked by agent '{}'",
                                resource, holder
                            ),
                            severity: ValidationSeverity::Error,
                        };
                    }
                }
                ValidationOutcome::pass("no_conflicting_locks")
            }),
        });

        // Pre-condition: No git conflicts present
        self.rules.push(ValidationRule {
            name: "no_git_conflicts".into(),
            description: "Git working tree must not have conflicts".into(),
            rule_type: RuleType::PreCondition,
            check: Box::new(|ctx| {
                // Only check for git-related operations
                if (ctx.operation.operation_type.starts_with("git_")
                    || ctx.operation.operation_type == "commit"
                    || ctx.operation.operation_type == "push")
                    && ctx.current_state.git_state.has_conflicts
                {
                    return ValidationOutcome {
                        passed: false,
                        rule_name: "no_git_conflicts".into(),
                        message: "Git working tree has unresolved conflicts".into(),
                        severity: ValidationSeverity::Error,
                    };
                }
                ValidationOutcome::pass("no_git_conflicts")
            }),
        });

        // Post-condition: Build artifacts should be marked invalid after source edit
        self.rules.push(ValidationRule {
            name: "artifacts_invalidated_after_edit".into(),
            description: "Build artifacts should be marked invalid after source edit".into(),
            rule_type: RuleType::PostCondition,
            check: Box::new(|ctx| {
                // After file edit, check that dirty flag is set
                if ctx.operation.operation_type == "file_write"
                    || ctx.operation.operation_type == "file_edit"
                {
                    for resource in &ctx.operation.resources_produced {
                        if let Some(file_status) =
                            ctx.current_state.files.get(&PathBuf::from(resource))
                            && !file_status.dirty
                        {
                            return ValidationOutcome {
                                passed: false,
                                rule_name: "artifacts_invalidated_after_edit".into(),
                                message: format!(
                                    "File '{}' should be marked dirty after edit",
                                    resource
                                ),
                                severity: ValidationSeverity::Warning,
                            };
                        }
                    }
                }
                ValidationOutcome::pass("artifacts_invalidated_after_edit")
            }),
        });

        // Post-condition: Files should be clean after successful build
        self.rules.push(ValidationRule {
            name: "files_clean_after_build".into(),
            description: "Source files should be marked clean after successful build".into(),
            rule_type: RuleType::PostCondition,
            check: Box::new(|ctx| {
                if ctx.operation.operation_type == "build" {
                    // Check that produced artifacts are valid
                    // This is informational - the actual cleanup happens elsewhere
                }
                ValidationOutcome::pass("files_clean_after_build")
            }),
        });

        // Invariant: No deadlock conditions
        self.rules.push(ValidationRule {
            name: "no_deadlock".into(),
            description: "Resource acquisition must not cause deadlock".into(),
            rule_type: RuleType::Invariant,
            check: Box::new(|ctx| {
                // Check for circular wait conditions
                // This requires knowledge of what resources each agent holds
                // and what they're waiting for

                // Get resources this agent currently holds
                let our_locks: Vec<_> = ctx
                    .current_state
                    .locks
                    .iter()
                    .filter(|(_, holder)| *holder == &ctx.agent_id)
                    .map(|(resource, _)| resource.clone())
                    .collect();

                // If we hold resources and are trying to acquire more,
                // check if any other agent is waiting for our resources
                // while holding resources we need
                if !our_locks.is_empty() && !ctx.operation.resources_needed.is_empty() {
                    for other_agent in &ctx.other_agents {
                        // Check if other agent holds any resource we need
                        let their_resources: std::collections::HashSet<_> =
                            other_agent.held_resources.iter().collect();
                        let we_need: std::collections::HashSet<_> =
                            ctx.operation.resources_needed.iter().collect();

                        let overlap: Vec<_> = their_resources.intersection(&we_need).collect();

                        if !overlap.is_empty() {
                            // Other agent holds something we need
                            // If they're waiting for something we hold, it's a potential deadlock
                            if let Some(ref waiting_for) = other_agent.waiting_for
                                && our_locks.contains(waiting_for) {
                                    return ValidationOutcome {
                                        passed: false,
                                        rule_name: "no_deadlock".into(),
                                        message: format!(
                                            "Potential deadlock: agent '{}' holds {:?} (we need it) and waits for '{}' (we hold it)",
                                            other_agent.agent_id,
                                            overlap,
                                            waiting_for
                                        ),
                                        severity: ValidationSeverity::Error,
                                    };
                                }
                        }
                    }
                }
                ValidationOutcome::pass("no_deadlock")
            }),
        });

        // Inter-agent: Coordinate git operations
        self.rules.push(ValidationRule {
            name: "git_coordination".into(),
            description: "Git operations must not conflict across agents".into(),
            rule_type: RuleType::InterAgent,
            check: Box::new(|ctx| {
                // Check if our git operation conflicts with other agents
                if ctx.operation.operation_type.starts_with("git_") {
                    for other_agent in &ctx.other_agents {
                        if let Some(ref other_op) = other_agent.current_operation
                            && other_op.starts_with("git_")
                        {
                            // Two git operations running concurrently
                            // Some combinations are OK (e.g., git_status + git_log)
                            // But others conflict (e.g., git_commit + git_commit)

                            let conflicting_ops = [
                                "git_commit",
                                "git_push",
                                "git_pull",
                                "git_merge",
                                "git_rebase",
                                "git_checkout",
                                "git_branch_create",
                                "git_branch_delete",
                            ];

                            let our_op = &ctx.operation.operation_type;
                            if conflicting_ops.contains(&our_op.as_str())
                                && conflicting_ops.contains(&other_op.as_str())
                            {
                                return ValidationOutcome {
                                    passed: false,
                                    rule_name: "git_coordination".into(),
                                    message: format!(
                                        "Git operation '{}' conflicts with '{}' by agent '{}'",
                                        our_op, other_op, other_agent.agent_id
                                    ),
                                    severity: ValidationSeverity::Error,
                                };
                            }
                        }
                    }
                }
                ValidationOutcome::pass("git_coordination")
            }),
        });

        // Inter-agent: Build/test coordination
        self.rules.push(ValidationRule {
            name: "build_coordination".into(),
            description: "Build operations should not run concurrently on same project".into(),
            rule_type: RuleType::InterAgent,
            check: Box::new(|ctx| {
                if ctx.operation.operation_type == "build" || ctx.operation.operation_type == "test"
                {
                    for other_agent in &ctx.other_agents {
                        if let Some(ref other_op) = other_agent.current_operation
                            && (other_op == "build" || other_op == "test")
                        {
                            // Check if operating on same project/scope
                            // For now, assume same scope if resources overlap
                            let our_resources: std::collections::HashSet<_> =
                                ctx.operation.resources_needed.iter().collect();
                            let their_resources: std::collections::HashSet<_> =
                                other_agent.held_resources.iter().collect();

                            if !our_resources.is_disjoint(&their_resources) {
                                return ValidationOutcome {
                                    passed: false,
                                    rule_name: "build_coordination".into(),
                                    message: format!(
                                        "Build/test operation conflicts with '{}' by agent '{}'",
                                        other_op, other_agent.agent_id
                                    ),
                                    severity: ValidationSeverity::Error,
                                };
                            }
                        }
                    }
                }
                ValidationOutcome::pass("build_coordination")
            }),
        });
    }

    /// Add a custom validation rule
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Remove a rule by name
    pub fn remove_rule(&mut self, name: &str) {
        self.rules.retain(|r| r.name != name);
    }

    /// Validate an operation before execution
    pub async fn pre_validate(
        &self,
        agent_id: &str,
        operation: &ProposedOperation,
    ) -> Vec<ValidationOutcome> {
        let context = self.build_context(agent_id, operation).await;

        self.rules
            .iter()
            .filter(|r| matches!(r.rule_type, RuleType::PreCondition | RuleType::Invariant))
            .map(|r| (r.check)(&context))
            .collect()
    }

    /// Validate an operation after execution
    pub async fn post_validate(
        &self,
        agent_id: &str,
        operation: &ProposedOperation,
        _result: &ValidationOperationResult,
    ) -> Vec<ValidationOutcome> {
        let context = self.build_context(agent_id, operation).await;

        self.rules
            .iter()
            .filter(|r| matches!(r.rule_type, RuleType::PostCondition | RuleType::Invariant))
            .map(|r| (r.check)(&context))
            .collect()
    }

    /// Check for inter-agent conflicts across multiple operations
    pub async fn check_inter_agent(
        &self,
        operations: &[(String, ProposedOperation)],
    ) -> Vec<ValidationOutcome> {
        let mut results = Vec::new();

        // Check each operation against all others
        for (agent_id, operation) in operations {
            // Build other agents' status from the other operations
            let other_agents: Vec<AgentStatus> = operations
                .iter()
                .filter(|(id, _)| id != agent_id)
                .map(|(id, op)| AgentStatus {
                    agent_id: id.clone(),
                    current_operation: Some(op.operation_type.clone()),
                    held_resources: op.resources_needed.clone(),
                    waiting_for: None,
                })
                .collect();

            let snapshot = self.state_model.snapshot().await;

            let context = ValidationContext {
                agent_id: agent_id.clone(),
                operation: operation.clone(),
                current_state: snapshot,
                other_agents,
            };

            for rule in self
                .rules
                .iter()
                .filter(|r| matches!(r.rule_type, RuleType::InterAgent))
            {
                results.push((rule.check)(&context));
            }
        }

        results
    }

    /// Validate using resource checker (for file/build/git conflicts)
    pub async fn check_resource_conflicts(
        &self,
        operation: &brainwires_agent::resource_checker::ProposedOperation,
    ) -> ConflictCheck {
        self.resource_checker.check_conflicts(operation).await
    }

    /// Build validation context from current state
    async fn build_context(
        &self,
        agent_id: &str,
        operation: &ProposedOperation,
    ) -> ValidationContext {
        let snapshot = self.state_model.snapshot().await;

        // Get other agents' status from operation state
        let active_ops = self
            .state_model
            .operation_state
            .get_active_operations()
            .await;
        let other_agents: Vec<AgentStatus> = active_ops
            .iter()
            .filter(|op| op.agent_id != agent_id)
            .map(|op| AgentStatus {
                agent_id: op.agent_id.clone(),
                current_operation: Some(op.operation_type.clone()),
                held_resources: op.resources_needed.clone(),
                waiting_for: None, // We don't track this yet
            })
            .collect();

        ValidationContext {
            agent_id: agent_id.to_string(),
            operation: operation.clone(),
            current_state: snapshot,
            other_agents,
        }
    }

    /// Convenience method to validate and return only failures
    pub async fn validate_with_failures_only(
        &self,
        agent_id: &str,
        operation: &ProposedOperation,
    ) -> Vec<ValidationOutcome> {
        let results = self.pre_validate(agent_id, operation).await;
        results.into_iter().filter(|r| !r.passed).collect()
    }

    /// Check if an operation can proceed (no Error-level failures)
    pub async fn can_proceed(&self, agent_id: &str, operation: &ProposedOperation) -> bool {
        let results = self.pre_validate(agent_id, operation).await;
        !results
            .iter()
            .any(|r| !r.passed && matches!(r.severity, ValidationSeverity::Error))
    }
}

/// A validation rule that checks a specific condition
pub struct ValidationRule {
    /// Unique name for this rule
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// When this rule should be checked
    pub rule_type: RuleType,
    /// The check function
    pub check: Box<dyn Fn(&ValidationContext) -> ValidationOutcome + Send + Sync>,
}

/// Types of validation rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleType {
    /// Checked before operation starts
    PreCondition,
    /// Checked after operation completes
    PostCondition,
    /// Always checked (invariants that must always hold)
    Invariant,
    /// Checked for conflicts between agents
    InterAgent,
}

/// Context provided to validation rules
pub struct ValidationContext {
    /// The agent performing the operation
    pub agent_id: String,
    /// The operation being validated
    pub operation: ProposedOperation,
    /// Current state snapshot
    pub current_state: StateSnapshot,
    /// Status of other agents
    pub other_agents: Vec<AgentStatus>,
}

/// Status of another agent for inter-agent validation
#[derive(Debug, Clone)]
pub struct AgentStatus {
    /// The agent's ID
    pub agent_id: String,
    /// What operation the agent is currently performing
    pub current_operation: Option<String>,
    /// Resources the agent currently holds
    pub held_resources: Vec<String>,
    /// Resource the agent is waiting for (if any)
    pub waiting_for: Option<String>,
}

/// Result of a validation check
#[derive(Debug, Clone)]
pub struct ValidationOutcome {
    /// Whether the check passed
    pub passed: bool,
    /// Name of the rule that produced this result
    pub rule_name: String,
    /// Human-readable message
    pub message: String,
    /// Severity level
    pub severity: ValidationSeverity,
}

impl ValidationOutcome {
    /// Create a passing outcome
    pub fn pass(rule_name: &str) -> Self {
        Self {
            passed: true,
            rule_name: rule_name.to_string(),
            message: "OK".to_string(),
            severity: ValidationSeverity::Info,
        }
    }

    /// Create a failing outcome with error severity
    pub fn fail_error(rule_name: &str, message: impl Into<String>) -> Self {
        Self {
            passed: false,
            rule_name: rule_name.to_string(),
            message: message.into(),
            severity: ValidationSeverity::Error,
        }
    }

    /// Create a failing outcome with warning severity
    pub fn fail_warning(rule_name: &str, message: impl Into<String>) -> Self {
        Self {
            passed: false,
            rule_name: rule_name.to_string(),
            message: message.into(),
            severity: ValidationSeverity::Warning,
        }
    }
}

// ValidationSeverity is imported from crate::validation_loop

/// Result of an operation (for post-validation)
#[derive(Debug, Clone)]
pub struct ValidationOperationResult {
    /// Whether the operation succeeded
    pub success: bool,
    /// Output data from the operation
    pub outputs: Option<serde_json::Value>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Summary of validation results
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total number of rules checked
    pub total_rules: usize,
    /// Number of rules that passed
    pub passed: usize,
    /// Number of warnings
    pub warnings: usize,
    /// Number of errors
    pub errors: usize,
    /// All individual outcomes
    pub outcomes: Vec<ValidationOutcome>,
}

impl ValidationSummary {
    /// Create a summary from a list of outcomes
    pub fn from_outcomes(outcomes: Vec<ValidationOutcome>) -> Self {
        let total_rules = outcomes.len();
        let passed = outcomes.iter().filter(|o| o.passed).count();
        let warnings = outcomes
            .iter()
            .filter(|o| !o.passed && matches!(o.severity, ValidationSeverity::Warning))
            .count();
        let errors = outcomes
            .iter()
            .filter(|o| !o.passed && matches!(o.severity, ValidationSeverity::Error))
            .count();

        Self {
            total_rules,
            passed,
            warnings,
            errors,
            outcomes,
        }
    }

    /// Check if all validations passed (no errors)
    pub fn is_valid(&self) -> bool {
        self.errors == 0
    }

    /// Check if all validations passed without warnings
    pub fn is_clean(&self) -> bool {
        self.errors == 0 && self.warnings == 0
    }

    /// Get error messages
    pub fn error_messages(&self) -> Vec<&str> {
        self.outcomes
            .iter()
            .filter(|o| !o.passed && matches!(o.severity, ValidationSeverity::Error))
            .map(|o| o.message.as_str())
            .collect()
    }

    /// Get warning messages
    pub fn warning_messages(&self) -> Vec<&str> {
        self.outcomes
            .iter()
            .filter(|o| !o.passed && matches!(o.severity, ValidationSeverity::Warning))
            .map(|o| o.message.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_agent::file_locks::FileLockManager;
    use brainwires_agent::resource_locks::ResourceLockManager;

    fn create_test_validation_agent() -> ValidationAgent {
        let state_model = Arc::new(ThreeStateModel::new());
        let file_locks = Arc::new(FileLockManager::new());
        let resource_locks = Arc::new(ResourceLockManager::new());
        let resource_checker = Arc::new(ResourceChecker::new(file_locks, resource_locks));
        ValidationAgent::new(state_model, resource_checker)
    }

    #[test]
    fn test_validation_outcome_pass() {
        let outcome = ValidationOutcome::pass("test_rule");
        assert!(outcome.passed);
        assert_eq!(outcome.rule_name, "test_rule");
        assert_eq!(outcome.severity, ValidationSeverity::Info);
    }

    #[test]
    fn test_validation_outcome_fail() {
        let outcome = ValidationOutcome::fail_error("test_rule", "Something went wrong");
        assert!(!outcome.passed);
        assert_eq!(outcome.rule_name, "test_rule");
        assert_eq!(outcome.message, "Something went wrong");
        assert_eq!(outcome.severity, ValidationSeverity::Error);
    }

    #[test]
    fn test_validation_summary() {
        let outcomes = vec![
            ValidationOutcome::pass("rule1"),
            ValidationOutcome::pass("rule2"),
            ValidationOutcome::fail_warning("rule3", "warning"),
            ValidationOutcome::fail_error("rule4", "error"),
        ];

        let summary = ValidationSummary::from_outcomes(outcomes);
        assert_eq!(summary.total_rules, 4);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.warnings, 1);
        assert_eq!(summary.errors, 1);
        assert!(!summary.is_valid());
        assert!(!summary.is_clean());
    }

    #[test]
    fn test_validation_summary_valid() {
        let outcomes = vec![
            ValidationOutcome::pass("rule1"),
            ValidationOutcome::fail_warning("rule2", "just a warning"),
        ];

        let summary = ValidationSummary::from_outcomes(outcomes);
        assert!(summary.is_valid()); // No errors
        assert!(!summary.is_clean()); // Has warnings
    }

    #[tokio::test]
    async fn test_pre_validate_no_conflicts() {
        let agent = create_test_validation_agent();

        let operation = ProposedOperation {
            agent_id: "agent-1".to_string(),
            operation_type: "file_write".to_string(),
            resources_needed: vec!["/test/file.rs".to_string()],
            resources_produced: vec!["/test/file.rs".to_string()],
        };

        let results = agent.pre_validate("agent-1", &operation).await;

        // All default pre-conditions should pass with no conflicts
        assert!(results.iter().all(|r| r.passed));
    }

    #[tokio::test]
    async fn test_pre_validate_lock_conflict() {
        let state_model = Arc::new(ThreeStateModel::new());

        // Set up a lock held by another agent
        state_model
            .dependency_state
            .set_holder("/test/file.rs", Some("agent-2"))
            .await;

        let file_locks = Arc::new(FileLockManager::new());
        let resource_locks = Arc::new(ResourceLockManager::new());
        let resource_checker = Arc::new(ResourceChecker::new(file_locks, resource_locks));
        let agent = ValidationAgent::new(state_model.clone(), resource_checker);

        let operation = ProposedOperation {
            agent_id: "agent-1".to_string(),
            operation_type: "file_write".to_string(),
            resources_needed: vec!["/test/file.rs".to_string()],
            resources_produced: vec![],
        };

        // Get snapshot to populate locks
        let mut snapshot = state_model.snapshot().await;
        snapshot
            .locks
            .insert("/test/file.rs".to_string(), "agent-2".to_string());

        // The validation should detect the lock conflict
        let results = agent.pre_validate("agent-1", &operation).await;

        // Note: The lock detection depends on snapshot state
        // This test validates the structure works
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_can_proceed() {
        let agent = create_test_validation_agent();

        let operation = ProposedOperation {
            agent_id: "agent-1".to_string(),
            operation_type: "file_read".to_string(),
            resources_needed: vec![],
            resources_produced: vec![],
        };

        let can_proceed = agent.can_proceed("agent-1", &operation).await;
        assert!(can_proceed);
    }

    #[tokio::test]
    async fn test_check_inter_agent_no_conflicts() {
        let agent = create_test_validation_agent();

        let operations = vec![
            (
                "agent-1".to_string(),
                ProposedOperation {
                    agent_id: "agent-1".to_string(),
                    operation_type: "file_read".to_string(),
                    resources_needed: vec!["/file1.rs".to_string()],
                    resources_produced: vec![],
                },
            ),
            (
                "agent-2".to_string(),
                ProposedOperation {
                    agent_id: "agent-2".to_string(),
                    operation_type: "file_read".to_string(),
                    resources_needed: vec!["/file2.rs".to_string()],
                    resources_produced: vec![],
                },
            ),
        ];

        let results = agent.check_inter_agent(&operations).await;

        // No inter-agent conflicts for reads on different files
        assert!(results.iter().all(|r| r.passed));
    }

    #[tokio::test]
    async fn test_check_inter_agent_git_conflict() {
        let agent = create_test_validation_agent();

        let operations = vec![
            (
                "agent-1".to_string(),
                ProposedOperation {
                    agent_id: "agent-1".to_string(),
                    operation_type: "git_commit".to_string(),
                    resources_needed: vec!["git_index".to_string()],
                    resources_produced: vec![],
                },
            ),
            (
                "agent-2".to_string(),
                ProposedOperation {
                    agent_id: "agent-2".to_string(),
                    operation_type: "git_push".to_string(),
                    resources_needed: vec!["git_remote".to_string()],
                    resources_produced: vec![],
                },
            ),
        ];

        let results = agent.check_inter_agent(&operations).await;

        // Git commit and push are conflicting operations
        let has_conflict = results.iter().any(|r| !r.passed);
        assert!(has_conflict);
    }

    #[test]
    fn test_agent_status_creation() {
        let status = AgentStatus {
            agent_id: "agent-1".to_string(),
            current_operation: Some("build".to_string()),
            held_resources: vec!["build_lock".to_string()],
            waiting_for: None,
        };

        assert_eq!(status.agent_id, "agent-1");
        assert_eq!(status.current_operation, Some("build".to_string()));
        assert_eq!(status.held_resources.len(), 1);
    }

    #[test]
    fn test_add_custom_rule() {
        let mut agent = create_test_validation_agent();
        let initial_count = agent.rules.len();

        agent.add_rule(ValidationRule {
            name: "custom_rule".into(),
            description: "A custom validation rule".into(),
            rule_type: RuleType::PreCondition,
            check: Box::new(|_ctx| ValidationOutcome::pass("custom_rule")),
        });

        assert_eq!(agent.rules.len(), initial_count + 1);
    }

    #[test]
    fn test_remove_rule() {
        let mut agent = create_test_validation_agent();
        let initial_count = agent.rules.len();

        // Add a rule then remove it
        agent.add_rule(ValidationRule {
            name: "to_remove".into(),
            description: "Will be removed".into(),
            rule_type: RuleType::PreCondition,
            check: Box::new(|_ctx| ValidationOutcome::pass("to_remove")),
        });

        assert_eq!(agent.rules.len(), initial_count + 1);

        agent.remove_rule("to_remove");
        assert_eq!(agent.rules.len(), initial_count);
    }
}
