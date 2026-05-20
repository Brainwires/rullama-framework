//! Dream consolidator — the core 4-phase memory consolidation engine.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

use brainwires_core::{Message, Provider, Role};

use super::fact_extractor::FactExtractor;
use super::metrics::{DreamMetrics, DreamReport};
use super::policy::DemotionPolicy;
use super::summarizer::DreamSummarizer;

/// Re-export the session store trait from the gateway crate is not viable across
/// crate boundaries without a shared dependency, so we define a minimal
/// mirror trait here that the consolidator operates against.
#[async_trait::async_trait]
pub trait DreamSessionStore: Send + Sync {
    /// List all session keys.
    async fn list_sessions(&self) -> Result<Vec<String>>;
    /// Load messages for a session.
    async fn load(&self, session_key: &str) -> Result<Option<Vec<Message>>>;
    /// Save (replace) messages for a session.
    async fn save(&self, session_key: &str, messages: &[Message]) -> Result<()>;
}

/// The dream consolidator runs the 4-phase consolidation cycle:
///
/// 1. **Orient** — scan sessions, count messages per session
/// 2. **Gather** — identify messages exceeding age/count thresholds
/// 3. **Consolidate** — summarise old messages, extract facts
/// 4. **Prune** — remove summarised originals, keep summaries + facts
pub struct DreamConsolidator {
    provider: Arc<dyn Provider>,
    policy: DemotionPolicy,
    metrics: DreamMetrics,
    #[cfg(feature = "telemetry")]
    analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,
}

impl DreamConsolidator {
    /// Create a new consolidator.
    pub fn new(provider: Arc<dyn Provider>, policy: DemotionPolicy) -> Self {
        Self {
            provider,
            policy,
            metrics: DreamMetrics::default(),
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Attach an analytics collector to record DreamCycle events.
    #[cfg(feature = "telemetry")]
    pub fn with_analytics(
        mut self,
        collector: std::sync::Arc<brainwires_telemetry::AnalyticsCollector>,
    ) -> Self {
        self.analytics_collector = Some(collector);
        self
    }

    /// Run a full 4-phase consolidation cycle across all sessions in the store.
    pub async fn run_cycle(&mut self, sessions: &dyn DreamSessionStore) -> Result<DreamReport> {
        let start = Instant::now();
        let mut report = DreamReport::default();

        // Phase 1: Orient — enumerate sessions
        let session_keys = sessions.list_sessions().await?;
        report.metrics.sessions_processed = session_keys.len();

        for key in &session_keys {
            match self.process_session(sessions, key).await {
                Ok(session_report) => {
                    report.metrics.messages_summarized += session_report.messages_summarized;
                    report.metrics.tokens_before += session_report.tokens_before;
                    report.metrics.tokens_after += session_report.tokens_after;
                    report.summaries_created.extend(session_report.summaries);
                    report.facts.extend(session_report.facts);
                    report.metrics.facts_extracted += session_report.fact_count;
                }
                Err(e) => {
                    report.errors.push(format!("Session {key}: {e}"));
                }
            }
        }

        report.metrics.duration = start.elapsed();
        self.metrics = report.metrics.clone();

        #[cfg(feature = "telemetry")]
        if let Some(ref collector) = self.analytics_collector {
            use brainwires_telemetry::AnalyticsEvent;
            let m = &report.metrics;
            collector.record(AnalyticsEvent::DreamCycle {
                session_id: None,
                sessions_processed: m.sessions_processed,
                messages_summarized: m.messages_summarized,
                facts_extracted: m.facts_extracted,
                tokens_before: m.tokens_before,
                tokens_after: m.tokens_after,
                duration_ms: m.duration.as_millis() as u64,
                timestamp: chrono::Utc::now(),
            });
        }

        Ok(report)
    }

    /// Process a single session through phases 2–4.
    async fn process_session(
        &self,
        sessions: &dyn DreamSessionStore,
        session_key: &str,
    ) -> Result<SessionConsolidationResult> {
        let mut result = SessionConsolidationResult::default();

        let messages = match sessions.load(session_key).await? {
            Some(m) if !m.is_empty() => m,
            _ => return Ok(result),
        };

        // Estimate token count (rough: 4 chars ≈ 1 token)
        let tokens_before: usize = messages.iter().map(|m| m.text_or_summary().len() / 4).sum();
        result.tokens_before = tokens_before;

        // Phase 2: Gather — split into keep vs. consolidate
        let keep_count = self.policy.keep_recent.max(1);
        if messages.len() <= keep_count {
            result.tokens_after = tokens_before;
            return Ok(result);
        }

        // Identify the system prompt (if any) and the recent tail to keep
        let has_system = messages
            .first()
            .map(|m| m.role == Role::System)
            .unwrap_or(false);

        let (system_msg, conversation) = if has_system {
            (Some(messages[0].clone()), &messages[1..])
        } else {
            (None, messages.as_slice())
        };

        if conversation.len() <= keep_count {
            result.tokens_after = tokens_before;
            return Ok(result);
        }

        let split_idx = conversation.len() - keep_count;
        let to_consolidate = &conversation[..split_idx];
        let to_keep = &conversation[split_idx..];

        // Phase 3: Consolidate — summarise and extract facts
        let summary = DreamSummarizer::summarize_messages(to_consolidate, &*self.provider).await?;
        result.messages_summarized = to_consolidate.len();
        result.summaries.push(summary.clone());

        let facts = FactExtractor::extract_facts(&summary, &*self.provider).await?;
        result.fact_count = facts.len();
        result.facts = facts;

        // Phase 4: Prune — rebuild the session with system prompt + summary + recent
        let mut new_messages = Vec::new();
        if let Some(sys) = system_msg {
            new_messages.push(sys);
        }
        // Insert the summary as a system-level context message
        new_messages.push(Message::system(format!("[Consolidated memory] {summary}")));
        new_messages.extend_from_slice(to_keep);

        let tokens_after: usize = new_messages
            .iter()
            .map(|m| m.text_or_summary().len() / 4)
            .sum();
        result.tokens_after = tokens_after;

        sessions.save(session_key, &new_messages).await?;

        Ok(result)
    }

    /// Return the metrics from the most recent cycle.
    pub fn last_metrics(&self) -> &DreamMetrics {
        &self.metrics
    }
}

/// Internal result for a single session consolidation.
#[derive(Debug, Default)]
struct SessionConsolidationResult {
    messages_summarized: usize,
    tokens_before: usize,
    tokens_after: usize,
    summaries: Vec<String>,
    facts: Vec<super::fact_extractor::ExtractedFact>,
    fact_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy() -> DemotionPolicy {
        DemotionPolicy {
            keep_recent: 2,
            ..DemotionPolicy::default()
        }
    }

    #[test]
    fn test_consolidator_construction() {
        use brainwires_core::{ChatOptions, ChatResponse, StreamChunk, Tool, Usage};

        struct FakeProvider;

        #[async_trait::async_trait]
        impl Provider for FakeProvider {
            fn name(&self) -> &str {
                "fake"
            }
            async fn chat(
                &self,
                _messages: &[Message],
                _tools: Option<&[Tool]>,
                _options: &ChatOptions,
            ) -> Result<ChatResponse> {
                Ok(ChatResponse {
                    message: Message::assistant("summary"),
                    usage: Usage::new(0, 0),
                    finish_reason: None,
                })
            }
            fn stream_chat<'a>(
                &'a self,
                _messages: &'a [Message],
                _tools: Option<&'a [Tool]>,
                _options: &'a ChatOptions,
            ) -> futures::stream::BoxStream<'a, Result<StreamChunk>> {
                Box::pin(futures::stream::empty())
            }
        }

        let provider = Arc::new(FakeProvider);
        let policy = make_policy();
        let consolidator = DreamConsolidator::new(provider, policy);
        assert_eq!(consolidator.last_metrics().sessions_processed, 0);
    }
}
