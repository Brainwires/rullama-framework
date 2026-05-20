//! Policy Engine Example
//!
//! Demonstrates declarative policy rules with `PolicyEngine` — deny, allow-with-audit,
//! and require-approval actions evaluated against various request types.
//!
//! Run: cargo run -p brainwires-permission --features native --example policy_engine

use anyhow::Result;
use brainwires_permission::{
    GitOperation, Policy, PolicyAction, PolicyCondition, PolicyEngine, PolicyRequest,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Policy Engine Example ===\n");

    // 1. Create a fresh policy engine with custom rules
    let mut engine = PolicyEngine::new();

    println!("--- 1. Registering policies ---\n");

    engine.add_policy(
        Policy::new("deny_env_files")
            .with_name("Deny .env Files")
            .with_description("Block all access to .env files which may contain secrets")
            .with_condition(PolicyCondition::FilePath("**/.env*".into()))
            .with_action(PolicyAction::Deny)
            .with_priority(100),
    );
    println!("  Added: deny_env_files  (priority 100, Deny)");

    engine.add_policy(
        Policy::new("approve_git_reset")
            .with_name("Approve Git Reset")
            .with_description("Require human approval before performing git reset")
            .with_condition(PolicyCondition::GitOp(GitOperation::Reset))
            .with_action(PolicyAction::RequireApproval)
            .with_priority(90),
    );
    println!("  Added: approve_git_reset  (priority 90, RequireApproval)");

    engine.add_policy(
        Policy::new("audit_bash")
            .with_name("Audit Bash Tool")
            .with_description("Allow bash tool usage but log every invocation for audit")
            .with_condition(PolicyCondition::Tool("bash".into()))
            .with_action(PolicyAction::AllowWithAudit)
            .with_priority(50),
    );
    println!("  Added: audit_bash  (priority 50, AllowWithAudit)");

    // 2. Evaluate requests against the policy engine
    println!("\n--- 2. Evaluating requests ---\n");

    let file_request = PolicyRequest::for_file(".env.local", "read_file");
    let decision = engine.evaluate(&file_request);
    println!(
        "  Request: read .env.local\n    Decision: {:?}\n    Matched policy: {:?}\n    Allowed: {}\n",
        decision.action,
        decision.matched_policy,
        decision.is_allowed(),
    );

    let git_request = PolicyRequest::for_git(GitOperation::Reset);
    let decision = engine.evaluate(&git_request);
    println!(
        "  Request: git reset\n    Decision: {:?}\n    Matched policy: {:?}\n    Requires approval: {}\n",
        decision.action,
        decision.matched_policy,
        decision.requires_approval(),
    );

    let tool_request = PolicyRequest::for_tool("bash");
    let decision = engine.evaluate(&tool_request);
    println!(
        "  Request: bash tool\n    Decision: {:?}\n    Matched policy: {:?}\n    Allowed: {}\n    Audit: {}\n",
        decision.action,
        decision.matched_policy,
        decision.is_allowed(),
        decision.audit,
    );

    let safe_request = PolicyRequest::for_file("src/main.rs", "read_file");
    let decision = engine.evaluate(&safe_request);
    println!(
        "  Request: read src/main.rs\n    Decision: {:?}\n    Matched policy: {:?}\n    Allowed: {}\n",
        decision.action,
        decision.matched_policy,
        decision.is_allowed(),
    );

    // 3. List all registered policies
    println!("--- 3. Registered policies ---\n");

    for policy in engine.policies() {
        println!(
            "  [{}] {} (priority {}) -> {:?}",
            policy.id, policy.name, policy.priority, policy.action,
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
