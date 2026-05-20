//! Cron / scheduled task runner for BrainClaw.
//!
//! `CronJob` and `CronStore` live in `brainwires_gateway::cron` so the
//! gateway's admin API can manage jobs.  This module provides the
//! `CronRunner` background task that polls the store and dispatches synthetic
//! `ChannelMessage`s to the `AgentInboundHandler`.
//!
//! # Cron expression format
//!
//! Standard 5-field cron: `min hour day month weekday`
//! e.g. `0 9 * * *` = every day at 09:00 UTC.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use brainwires_gateway::AgentInboundHandler;
use brainwires_gateway::channel_registry::ChannelRegistry;
use brainwires_gateway::cron::CronStore;
use brainwires_network::channels::ConversationId;
use brainwires_network::channels::message::{ChannelMessage, MessageContent, MessageId};
use chrono::Utc;
use uuid::Uuid;

/// Background task that polls cron jobs and fires them when due.
pub struct CronRunner {
    store: Arc<CronStore>,
    handler: Arc<AgentInboundHandler>,
    channels: Arc<ChannelRegistry>,
}

impl CronRunner {
    pub fn new(
        store: Arc<CronStore>,
        handler: Arc<AgentInboundHandler>,
        channels: Arc<ChannelRegistry>,
    ) -> Self {
        Self {
            store,
            handler,
            channels,
        }
    }

    /// Start the cron runner as a background tokio task.
    ///
    /// Polls every 30 seconds, fires any jobs whose next scheduled time has
    /// passed since their `last_run`.
    pub fn spawn(self: Arc<Self>) {
        tokio::spawn(async move {
            tracing::info!("Cron runner started");
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
                if let Err(e) = self.tick().await {
                    tracing::error!(error = %e, "Cron tick error");
                }
            }
        });
    }

    async fn tick(&self) -> Result<()> {
        let now = Utc::now();
        let jobs = self.store.list().await;

        for job in jobs {
            if !job.enabled {
                continue;
            }

            let since = job
                .last_run
                .unwrap_or_else(|| now - chrono::Duration::days(1));
            let Some(next) = job.next_run_after(since) else {
                tracing::warn!(job = %job.name, "Could not compute next run; skipping");
                continue;
            };

            if next <= now {
                self.fire_job(&job, now).await;
            }
        }

        Ok(())
    }

    async fn fire_job(&self, job: &brainwires_gateway::cron::CronJob, now: chrono::DateTime<Utc>) {
        tracing::info!(job = %job.name, schedule = %job.schedule, "Firing cron job");

        // Find a connected channel adapter for the target platform.
        let channel_ids = self.channels.find_by_type(&job.target_platform);
        let channel_id = match channel_ids.first() {
            Some(id) => *id,
            None => {
                tracing::warn!(
                    job = %job.name,
                    platform = %job.target_platform,
                    "No connected channel adapter for cron job; using nil UUID (response will be dropped)"
                );
                Uuid::nil()
            }
        };

        let msg = ChannelMessage {
            id: MessageId::new(Uuid::new_v4().to_string()),
            conversation: ConversationId {
                platform: job.target_platform.clone(),
                channel_id: job.target_channel_id.clone(),
                server_id: None,
            },
            author: job.target_user_id.clone(),
            content: MessageContent::Text(job.prompt.clone()),
            thread_id: None,
            reply_to: None,
            timestamp: now,
            attachments: vec![],
            metadata: HashMap::new(),
        };

        if let Err(e) = self.handler.dispatch_message(channel_id, msg).await {
            tracing::error!(job = %job.name, error = %e, "Cron job dispatch failed");
        } else if let Err(e) = self.store.record_run(job.id, now).await {
            tracing::warn!(job = %job.name, error = %e, "Failed to record cron run");
        }
    }
}
