//! Example: Git Workflow Pipeline — merge policies, events, and branch naming.
//!
//! ```bash
//! cargo run -p brainwires-autonomy --example git_workflow_pipeline --features git-workflow
//! ```

use brainwires_autonomy::config::GitWorkflowConfig;
use brainwires_autonomy::git_workflow::{
    WorkflowEvent,
    forge::{Comment, CommitRef, Issue, MergeMethod, PrState, PullRequest, RepoRef},
    merge_policy::{
        ConfidenceBasedPolicy, MergeContext, MergeDecision, MergePolicy, RequireApprovalPolicy,
    },
    trigger::ProgrammaticTrigger,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Git Workflow Pipeline Example ===\n");

    // 1. Configuration
    let config = GitWorkflowConfig::default();
    println!("--- GitWorkflowConfig ---");
    println!("  branch_prefix  = \"{}\"", config.branch_prefix);
    println!("  auto_merge     = {}", config.auto_merge);
    println!("  merge_method   = {}", config.merge_method);
    println!("  min_confidence = {:.0}%", config.min_confidence * 100.0);
    println!();

    // 2. Forge types
    println!("--- Forge Types ---");
    let repo = RepoRef {
        owner: "Brainwires".to_string(),
        name: "brainwires-cli".to_string(),
    };
    println!("  Repo: {}", repo.full_name());

    let issue = Issue {
        id: "123456".to_string(),
        number: 42,
        title: "Fix: authentication token expiry not handled".to_string(),
        body: "When the auth token expires during a long session, the CLI crashes.".to_string(),
        labels: vec!["bug".to_string(), "auth".to_string()],
        author: "community-user".to_string(),
        url: "https://github.com/Brainwires/brainwires-cli/issues/42".to_string(),
    };
    println!(
        "  Issue #{}: {} (by {})",
        issue.number, issue.title, issue.author
    );
    println!("  Labels: {:?}", issue.labels);
    println!();

    // 3. Workflow events
    println!("--- Workflow Events ---");
    let events: Vec<WorkflowEvent> = vec![
        WorkflowEvent::IssueOpened {
            issue: issue.clone(),
            repo: repo.clone(),
        },
        WorkflowEvent::IssueCommented {
            issue: issue.clone(),
            comment: Comment {
                id: "c-1".to_string(),
                author: "maintainer".to_string(),
                body: "@brainwires fix this please".to_string(),
            },
            repo: repo.clone(),
        },
        WorkflowEvent::PushReceived {
            branch: "main".to_string(),
            commits: vec![CommitRef {
                sha: "abc123".to_string(),
                message: "fix: handle token expiry gracefully".to_string(),
            }],
            repo: repo.clone(),
        },
        WorkflowEvent::Manual {
            description: "Re-investigate issue #42".to_string(),
            repo: repo.clone(),
        },
    ];

    for event in &events {
        match event {
            WorkflowEvent::IssueOpened { issue, .. } => {
                println!("  IssueOpened: #{} - {}", issue.number, issue.title);
            }
            WorkflowEvent::IssueCommented { comment, .. } => {
                println!(
                    "  IssueCommented: {} says \"{}\"",
                    comment.author, comment.body
                );
            }
            WorkflowEvent::PushReceived {
                branch, commits, ..
            } => {
                println!("  PushReceived: {} ({} commits)", branch, commits.len());
            }
            WorkflowEvent::Manual { description, .. } => {
                println!("  Manual: {description}");
            }
            WorkflowEvent::PrReviewApproved { pr, .. } => {
                println!("  PrReviewApproved: #{}", pr.number);
            }
        }
    }
    println!();

    // 4. Merge policies
    println!("--- Merge Policies ---");

    let pr = PullRequest {
        id: "pr-100".to_string(),
        number: 100,
        title: "Fix token expiry handling".to_string(),
        body: "Fixes #42 — handles auth token expiry gracefully".to_string(),
        head_branch: "autonomy/42-fix-auth-token".to_string(),
        base_branch: "main".to_string(),
        url: "https://github.com/Brainwires/brainwires-cli/pull/100".to_string(),
        state: PrState::Open,
    };

    let ctx = MergeContext {
        confidence: 0.85,
        diff_lines: 42,
        files_modified: 3,
    };

    // RequireApprovalPolicy
    let approval_policy = RequireApprovalPolicy;
    let decision = approval_policy.evaluate(&pr, &ctx).await;
    println!("  RequireApprovalPolicy: {}", format_decision(&decision));

    // ConfidenceBasedPolicy at 80% threshold
    let confidence_policy = ConfidenceBasedPolicy::new(0.80, MergeMethod::Squash);
    let decision = confidence_policy.evaluate(&pr, &ctx).await;
    println!(
        "  ConfidenceBasedPolicy(80%): {}",
        format_decision(&decision)
    );

    // ConfidenceBasedPolicy at 90% threshold — should wait
    let strict_policy = ConfidenceBasedPolicy::new(0.90, MergeMethod::Squash);
    let decision = strict_policy.evaluate(&pr, &ctx).await;
    println!(
        "  ConfidenceBasedPolicy(90%): {}",
        format_decision(&decision)
    );

    println!();

    // 5. Programmatic trigger
    println!("--- Programmatic Trigger ---");
    let _trigger = ProgrammaticTrigger::new();
    println!("  Created ProgrammaticTrigger (for manual event dispatch)");

    println!("\nDone.");
    Ok(())
}

fn format_decision(decision: &MergeDecision) -> String {
    match decision {
        MergeDecision::Approve { method } => format!("Approve ({method:?})"),
        MergeDecision::Wait { reason } => format!("Wait: {reason}"),
        MergeDecision::Reject { reason } => format!("Reject: {reason}"),
    }
}
