//! Example: CI/CD Orchestrator — webhook logging, repo config, and variable interpolation.
//!
//! ```bash
//! cargo run -p brainwires-autonomy --example cicd_orchestrator --features webhook
//! ```

use brainwires_autonomy::config::{CommandConfig, WebhookConfig, WebhookRepoConfig};
use brainwires_autonomy::git_workflow::{
    InterpolationContext, WebhookAction,
    webhook_log::{WebhookEvent, WebhookLogger},
};

fn main() {
    println!("=== CI/CD Orchestrator Example ===\n");

    // 1. Webhook configuration
    println!("--- Webhook Configuration ---");
    let webhook_config = WebhookConfig::default();
    println!(
        "  listen_addr = {}:{}",
        webhook_config.listen_addr, webhook_config.port
    );
    println!("  log_dir     = {}", webhook_config.log_dir);
    println!("  keep_days   = {}", webhook_config.keep_days);
    println!();

    // 2. Per-repo configuration
    println!("--- Per-Repository Config ---");
    let repo_config = WebhookRepoConfig {
        events: vec!["issues".to_string(), "push".to_string()],
        auto_investigate: true,
        auto_fix: true,
        auto_merge: false,
        labels_filter: vec!["auto-fix".to_string(), "bug".to_string()],
        post_commands: vec![CommandConfig {
            cmd: "echo".to_string(),
            args: vec!["Processed ${REPO_NAME} issue #${ISSUE_NUMBER}".to_string()],
            working_dir: Some("/tmp/${REPO_NAME}".to_string()),
        }],
    };

    println!("  events         = {:?}", repo_config.events);
    println!("  auto_investigate = {}", repo_config.auto_investigate);
    println!("  auto_fix       = {}", repo_config.auto_fix);
    println!("  auto_merge     = {}", repo_config.auto_merge);
    println!("  labels_filter  = {:?}", repo_config.labels_filter);
    println!(
        "  post_commands  = {} command(s)",
        repo_config.post_commands.len()
    );
    println!();

    // 3. Variable interpolation
    println!("--- Variable Interpolation ---");

    let issue_ctx = InterpolationContext::for_issue("brainwires-cli", 42, "Fix auth bug");
    println!("  Issue context:");
    println!("    Template: \"Fixing ${{REPO_NAME}} issue #${{ISSUE_NUMBER}}\"");
    println!(
        "    Result:   \"{}\"",
        issue_ctx.interpolate("Fixing ${REPO_NAME} issue #${ISSUE_NUMBER}")
    );

    let push_ctx = InterpolationContext::for_push("brainwires-cli", "refs/tags/v1.2.3", "abc123");
    println!("  Push context (tag):");
    println!("    Template: \"Deploy ${{REPO_NAME}} version ${{VERSION}}\"");
    println!(
        "    Result:   \"{}\"",
        push_ctx.interpolate("Deploy ${REPO_NAME} version ${VERSION}")
    );

    let branch_ctx = InterpolationContext::for_push("brainwires-cli", "main", "def456");
    println!("  Push context (branch):");
    println!("    Template: \"Build ${{BRANCH_NAME}} @ ${{COMMIT_SHA}}\"");
    println!(
        "    Result:   \"{}\"",
        branch_ctx.interpolate("Build ${BRANCH_NAME} @ ${COMMIT_SHA}")
    );

    // Interpolate a command config
    let interpolated = issue_ctx.interpolate_command(&repo_config.post_commands[0]);
    println!("  Command interpolation:");
    println!("    cmd:  {}", interpolated.cmd);
    println!("    args: {:?}", interpolated.args);
    println!("    dir:  {:?}", interpolated.working_dir);
    println!();

    // 4. Webhook event logging
    println!("--- Webhook Event Logging ---");
    let tempdir = std::env::temp_dir().join("brainwires-example-webhook-logs");
    let mut logger = WebhookLogger::new(tempdir.clone(), 7);
    logger.init().unwrap();

    let events = vec![
        WebhookEvent {
            timestamp: chrono::Utc::now(),
            event_type: "issues".to_string(),
            repo: "Brainwires/brainwires-cli".to_string(),
            git_ref: String::new(),
            action: WebhookAction::InvestigationStarted,
            message: "Investigating issue #42: Fix auth bug".to_string(),
            client_ip: Some("203.0.113.50".to_string()),
        },
        WebhookEvent {
            timestamp: chrono::Utc::now(),
            event_type: "issues".to_string(),
            repo: "Brainwires/brainwires-cli".to_string(),
            git_ref: String::new(),
            action: WebhookAction::FixApplied,
            message: "Applied fix for issue #42".to_string(),
            client_ip: None,
        },
        WebhookEvent {
            timestamp: chrono::Utc::now(),
            event_type: "issues".to_string(),
            repo: "Brainwires/brainwires-cli".to_string(),
            git_ref: String::new(),
            action: WebhookAction::PrCreated,
            message: "Created PR #100 for issue #42".to_string(),
            client_ip: None,
        },
        WebhookEvent {
            timestamp: chrono::Utc::now(),
            event_type: "push".to_string(),
            repo: "Brainwires/brainwires-cli".to_string(),
            git_ref: "refs/heads/main".to_string(),
            action: WebhookAction::DeploymentStarted,
            message: "3 commits pushed to main".to_string(),
            client_ip: Some("203.0.113.50".to_string()),
        },
    ];

    for event in &events {
        logger.log_event(event).unwrap();
        println!(
            "  Logged: {} {} — {}",
            event.action, event.event_type, event.message
        );
    }

    println!("\n  Log files written to: {}", tempdir.display());

    // Cleanup
    let _ = std::fs::remove_dir_all(&tempdir);

    println!("\nDone.");
}
