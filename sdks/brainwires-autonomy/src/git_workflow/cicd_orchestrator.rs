//! Full CI/CD pipeline orchestrator for community-driven automation.
//!
//! Receives workflow events (from webhooks, cron, or programmatic triggers)
//! and orchestrates the full pipeline: investigate -> fix -> test -> PR -> merge.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{RwLock, mpsc, watch};

use super::forge::{Issue, RepoRef};
use super::pipeline::GitWorkflowPipeline;
use super::trigger::WorkflowEvent;
use super::webhook_config::{InterpolationContext, should_handle_event};
use super::webhook_log::{WebhookAction, WebhookEvent, WebhookLogger};
use crate::config::{GitWorkflowConfig, WebhookRepoConfig};
use crate::safety::SafetyGuard;

/// Tracks an active investigation with cancellation support.
struct ActiveInvestigation {
    /// Issue number being investigated.
    _issue_number: u64,
    /// Channel to signal cancellation.
    cancel_tx: watch::Sender<bool>,
}

/// CI/CD orchestrator that manages the full autonomous pipeline.
///
/// Receives workflow events, checks per-repository configuration and safety
/// limits, dispatches issue investigations, executes post-event commands,
/// and logs all activity via the webhook logger.
pub struct CiCdOrchestrator {
    #[allow(dead_code)]
    config: GitWorkflowConfig,
    repo_configs: HashMap<String, WebhookRepoConfig>,
    pipeline: Arc<GitWorkflowPipeline>,
    logger: Arc<RwLock<WebhookLogger>>,
    active: Arc<RwLock<HashMap<String, ActiveInvestigation>>>,
    safety: Arc<RwLock<SafetyGuard>>,
}

impl CiCdOrchestrator {
    /// Create a new CI/CD orchestrator.
    pub fn new(
        config: GitWorkflowConfig,
        pipeline: GitWorkflowPipeline,
        safety: SafetyGuard,
    ) -> Self {
        let log_dir = std::path::PathBuf::from(&config.webhook.log_dir);
        let mut logger = WebhookLogger::new(log_dir, config.webhook.keep_days);
        if let Err(e) = logger.init() {
            tracing::warn!("Failed to initialize webhook logger: {e}");
        }

        let repo_configs = config.webhook.repos.clone();

        Self {
            config,
            repo_configs,
            pipeline: Arc::new(pipeline),
            logger: Arc::new(RwLock::new(logger)),
            active: Arc::new(RwLock::new(HashMap::new())),
            safety: Arc::new(RwLock::new(safety)),
        }
    }

    /// Run the orchestrator, consuming events from the given channel.
    pub async fn run(&self, mut rx: mpsc::Receiver<WorkflowEvent>) {
        while let Some(event) = rx.recv().await {
            if let Err(e) = self.handle_event(event).await {
                tracing::error!("CI/CD orchestrator error: {e}");
            }
        }
    }

    /// Handle a single workflow event.
    pub async fn handle_event(&self, event: WorkflowEvent) -> anyhow::Result<()> {
        match event {
            WorkflowEvent::IssueOpened {
                ref issue,
                ref repo,
            } => {
                self.handle_issue_opened(issue, repo).await?;
            }
            WorkflowEvent::IssueCommented {
                ref issue,
                ref comment,
                ref repo,
            } => {
                // Re-investigate if the comment contains a trigger keyword
                let trigger_keywords = ["@brainwires fix", "@brainwires investigate", "/autofix"];
                let should_trigger = trigger_keywords
                    .iter()
                    .any(|kw| comment.body.to_lowercase().contains(&kw.to_lowercase()));

                if should_trigger {
                    self.handle_issue_opened(issue, repo).await?;
                } else {
                    self.log_event(
                        "issue_comment",
                        &repo.full_name(),
                        "",
                        WebhookAction::Ignored,
                        &format!(
                            "Comment on #{} doesn't contain trigger keyword",
                            issue.number
                        ),
                    )
                    .await;
                }
            }
            WorkflowEvent::PushReceived {
                ref branch,
                ref commits,
                ref repo,
            } => {
                let sha = commits.first().map(|c| c.sha.as_str()).unwrap_or("");
                self.log_event(
                    "push",
                    &repo.full_name(),
                    branch,
                    WebhookAction::DeploymentStarted,
                    &format!("{} commits pushed", commits.len()),
                )
                .await;

                // Execute post-push commands if configured
                if let Some(repo_config) = self.repo_configs.get(&repo.full_name()) {
                    let ctx = InterpolationContext::for_push(&repo.name, branch, sha);
                    for cmd_config in &repo_config.post_commands {
                        let interpolated = ctx.interpolate_command(cmd_config);
                        self.execute_command(&interpolated).await;
                    }
                }
            }
            WorkflowEvent::PrReviewApproved { ref pr, ref repo } => {
                self.log_event(
                    "pull_request_review",
                    &repo.full_name(),
                    "",
                    WebhookAction::Ignored,
                    &format!("PR #{} approved", pr.number),
                )
                .await;
            }
            WorkflowEvent::Manual {
                ref description,
                ref repo,
            } => {
                tracing::info!("Manual trigger for {}: {description}", repo.full_name());
            }
        }

        Ok(())
    }

    async fn handle_issue_opened(&self, issue: &Issue, repo: &RepoRef) -> anyhow::Result<()> {
        let repo_key = repo.full_name();

        // Check repo configuration
        let repo_config = self.repo_configs.get(&repo_key);
        if let Some(config) = repo_config {
            if !should_handle_event(config, "issues", &issue.labels) {
                self.log_event(
                    "issues",
                    &repo_key,
                    "",
                    WebhookAction::Ignored,
                    &format!("Issue #{} doesn't match filters", issue.number),
                )
                .await;
                return Ok(());
            }

            if !config.auto_investigate {
                self.log_event(
                    "issues",
                    &repo_key,
                    "",
                    WebhookAction::Ignored,
                    &format!("Auto-investigate disabled for {repo_key}"),
                )
                .await;
                return Ok(());
            }
        }

        // Check for duplicate / cancel existing investigation
        let investigation_key = format!("{repo_key}#{}", issue.number);
        {
            let mut active = self.active.write().await;
            if let Some(existing) = active.remove(&investigation_key) {
                let _ = existing.cancel_tx.send(true);
                tracing::info!("Cancelled existing investigation for {}", investigation_key);
            }
        }

        // Check safety
        {
            let mut safety = self.safety.write().await;
            if let Err(stop) = safety.check_can_continue() {
                tracing::warn!("Safety stop prevents investigation: {stop}");
                return Ok(());
            }
        }

        // Start investigation
        let (cancel_tx, _cancel_rx) = watch::channel(false);
        {
            let mut active = self.active.write().await;
            active.insert(
                investigation_key.clone(),
                ActiveInvestigation {
                    _issue_number: issue.number,
                    cancel_tx,
                },
            );
        }

        self.log_event(
            "issues",
            &repo_key,
            "",
            WebhookAction::InvestigationStarted,
            &format!("Investigating issue #{}: {}", issue.number, issue.title),
        )
        .await;

        // Run the pipeline
        let result = self.pipeline.handle_issue(issue, repo).await;

        // Clean up active investigation
        {
            let mut active = self.active.write().await;
            active.remove(&investigation_key);
        }

        match result {
            Ok(()) => {
                self.log_event(
                    "issues",
                    &repo_key,
                    "",
                    WebhookAction::PrCreated,
                    &format!("Successfully processed issue #{}", issue.number),
                )
                .await;
            }
            Err(e) => {
                tracing::error!("Pipeline failed for issue #{}: {e}", issue.number);
            }
        }

        Ok(())
    }

    async fn log_event(
        &self,
        event_type: &str,
        repo: &str,
        git_ref: &str,
        action: WebhookAction,
        message: &str,
    ) {
        let event = WebhookEvent {
            timestamp: chrono::Utc::now(),
            event_type: event_type.to_string(),
            repo: repo.to_string(),
            git_ref: git_ref.to_string(),
            action,
            message: message.to_string(),
            client_ip: None,
        };

        if let Err(e) = self.logger.write().await.log_event(&event) {
            tracing::warn!("Failed to log webhook event: {e}");
        }
    }

    async fn execute_command(&self, cmd: &crate::config::CommandConfig) {
        let working_dir = cmd.working_dir.as_deref().unwrap_or(".");
        let result = tokio::process::Command::new(&cmd.cmd)
            .args(&cmd.args)
            .current_dir(working_dir)
            .output()
            .await;

        match result {
            Ok(output) if output.status.success() => {
                tracing::info!("Post-event command succeeded: {} {:?}", cmd.cmd, cmd.args);
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                tracing::warn!("Post-event command failed: {stderr}");
            }
            Err(e) => {
                tracing::error!("Failed to execute post-event command: {e}");
            }
        }
    }

    /// Get the number of active investigations.
    pub async fn active_count(&self) -> usize {
        self.active.read().await.len()
    }

    /// Get active investigation keys.
    pub async fn active_keys(&self) -> Vec<String> {
        self.active.read().await.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cicd_orchestrator_module_compiles() {
        // Compilation test — actual integration tests require mock providers
    }
}
