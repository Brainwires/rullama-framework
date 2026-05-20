//! Outcome metrics — per-agent aggregates with Prometheus text export.
//!
//! [`MetricsRegistry`] implements [`AnalyticsSink`] so it plugs directly into
//! the existing [`AnalyticsCollector`] pipeline and updates counters in real-time
//! as events flow through.
//!
//! The [`MetricsRegistry::prometheus_text`] method returns the full Prometheus
//! exposition format — wire it to any HTTP handler to expose a `/metrics`
//! endpoint without adding heavy dependencies.
//!
//! ## Example
//!
//! ```rust,no_run
//! use brainwires_telemetry::{AnalyticsCollector, metrics::MetricsRegistry};
//!
//! let registry = MetricsRegistry::new();
//! let collector = AnalyticsCollector::new(vec![Box::new(registry.clone())]);
//!
//! // Later, from any HTTP handler:
//! // let body = registry.prometheus_text();
//! ```

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;

use crate::{AnalyticsError, AnalyticsEvent, sink::AnalyticsSink};

// ── OutcomeMetrics ────────────────────────────────────────────────────────────

/// Aggregated outcome metrics for a single agent.
#[derive(Debug, Clone, Default)]
pub struct OutcomeMetrics {
    /// Agent identifier.
    pub agent_id: String,

    // ── Agent run counters ───────────────────────────────────────────────────
    /// Total agent run attempts.
    pub total_runs: u64,
    /// Runs that completed successfully.
    pub success_count: u64,
    /// Runs that ended in failure.
    pub failure_count: u64,
    /// Sum of iterations across all runs.
    pub total_iterations: u64,

    // ── Tool usage ───────────────────────────────────────────────────────────
    /// Total tool calls made.
    pub total_tool_calls: u64,
    /// Tool calls that produced an error.
    pub tool_error_count: u64,

    // ── Provider / cost ──────────────────────────────────────────────────────
    /// Total provider (LLM) calls.
    pub provider_call_count: u64,
    /// Prompt tokens consumed.
    pub total_tokens_prompt: u64,
    /// Completion tokens generated.
    pub total_tokens_completion: u64,
    /// Total estimated cost in USD.
    pub total_cost_usd: f64,
    /// Sum of provider call durations in milliseconds.
    pub total_provider_duration_ms: u64,
    /// Sum of agent run durations in milliseconds.
    pub total_run_duration_ms: u64,

    // ── Prompt cache (Anthropic today) ───────────────────────────────────────
    /// Cumulative tokens served from the provider's prompt cache.
    pub total_cache_read_tokens: u64,
    /// Cumulative tokens charged to populate the provider's prompt cache.
    pub total_cache_creation_tokens: u64,
}

impl OutcomeMetrics {
    /// Fraction of runs that succeeded (0.0–1.0). Returns 0.0 when no runs.
    pub fn success_rate(&self) -> f64 {
        if self.total_runs == 0 {
            0.0
        } else {
            self.success_count as f64 / self.total_runs as f64
        }
    }

    /// Average cost per run in USD. Returns 0.0 when no runs.
    pub fn avg_cost_per_run_usd(&self) -> f64 {
        if self.total_runs == 0 {
            0.0
        } else {
            self.total_cost_usd / self.total_runs as f64
        }
    }

    /// Average run duration in milliseconds. Returns 0.0 when no runs.
    pub fn avg_run_duration_ms(&self) -> f64 {
        if self.total_runs == 0 {
            0.0
        } else {
            self.total_run_duration_ms as f64 / self.total_runs as f64
        }
    }

    /// Average provider call latency in milliseconds.
    pub fn avg_provider_latency_ms(&self) -> f64 {
        if self.provider_call_count == 0 {
            0.0
        } else {
            self.total_provider_duration_ms as f64 / self.provider_call_count as f64
        }
    }

    /// Tool error rate (0.0–1.0).
    pub fn tool_error_rate(&self) -> f64 {
        if self.total_tool_calls == 0 {
            0.0
        } else {
            self.tool_error_count as f64 / self.total_tool_calls as f64
        }
    }

    /// Prompt-cache hit rate: fraction of input tokens served from cache
    /// across all tracked provider calls. `0.0` if no input tokens have been
    /// seen. Useful for validating that a `CacheStrategy` upgrade actually
    /// produced cache hits in production.
    ///
    /// (`CacheStrategy` is defined in `brainwires_core::provider`.)
    pub fn cache_hit_rate(&self) -> f64 {
        let denom = self.total_tokens_prompt + self.total_cache_read_tokens;
        if denom == 0 {
            0.0
        } else {
            self.total_cache_read_tokens as f64 / denom as f64
        }
    }
}

// ── MetricsRegistry ───────────────────────────────────────────────────────────

/// Thread-safe registry of per-agent [`OutcomeMetrics`].
///
/// Implements [`AnalyticsSink`] — register it with an [`AnalyticsCollector`](crate::collector::AnalyticsCollector)
/// and it will update counters automatically as events arrive.
#[derive(Clone, Default)]
pub struct MetricsRegistry {
    inner: Arc<Mutex<HashMap<String, OutcomeMetrics>>>,
}

impl MetricsRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a snapshot of the metrics for a specific agent.
    pub fn get(&self, agent_id: &str) -> Option<OutcomeMetrics> {
        self.inner
            .lock()
            .expect("metrics registry lock poisoned")
            .get(agent_id)
            .cloned()
    }

    /// Return snapshots for all tracked agents.
    pub fn all(&self) -> Vec<OutcomeMetrics> {
        self.inner
            .lock()
            .expect("metrics registry lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// Reset metrics for a specific agent.
    pub fn reset(&self, agent_id: &str) {
        self.inner
            .lock()
            .expect("metrics registry lock poisoned")
            .remove(agent_id);
    }

    /// Reset all tracked metrics.
    pub fn reset_all(&self) {
        self.inner
            .lock()
            .expect("metrics registry lock poisoned")
            .clear();
    }

    // ── Prometheus export ─────────────────────────────────────────────────────

    /// Render all tracked metrics as a Prometheus exposition format string.
    ///
    /// Mount this output on a `/metrics` HTTP endpoint to scrape with
    /// Prometheus, Grafana Agent, or any OpenMetrics-compatible collector.
    ///
    /// ```text
    /// # HELP brainwires_agent_runs_total Total agent runs attempted
    /// # TYPE brainwires_agent_runs_total counter
    /// brainwires_agent_runs_total{agent_id="code-review"} 42
    /// ```
    pub fn prometheus_text(&self) -> String {
        let metrics = self.inner.lock().expect("metrics registry lock poisoned");
        let mut out = String::with_capacity(metrics.len() * 512);

        // Helper closures
        let counter = |out: &mut String, name: &str, help: &str| {
            out.push_str(&format!("# HELP {name} {help}\n"));
            out.push_str(&format!("# TYPE {name} counter\n"));
        };
        let gauge = |out: &mut String, name: &str, help: &str| {
            out.push_str(&format!("# HELP {name} {help}\n"));
            out.push_str(&format!("# TYPE {name} gauge\n"));
        };

        if metrics.is_empty() {
            return out;
        }

        // ── brainwires_agent_runs_total ──────────────────────────────────────
        counter(
            &mut out,
            "brainwires_agent_runs_total",
            "Total agent runs attempted",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_runs_total",
                &m.agent_id,
                m.total_runs,
            ));
        }

        // ── brainwires_agent_runs_success ────────────────────────────────────
        counter(
            &mut out,
            "brainwires_agent_runs_success_total",
            "Agent runs that succeeded",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_runs_success_total",
                &m.agent_id,
                m.success_count,
            ));
        }

        // ── brainwires_agent_runs_failure ────────────────────────────────────
        counter(
            &mut out,
            "brainwires_agent_runs_failure_total",
            "Agent runs that failed",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_runs_failure_total",
                &m.agent_id,
                m.failure_count,
            ));
        }

        // ── brainwires_agent_success_rate ────────────────────────────────────
        gauge(
            &mut out,
            "brainwires_agent_success_rate",
            "Agent run success rate (0-1)",
        );
        for m in metrics.values() {
            out.push_str(&metric_line_f(
                "brainwires_agent_success_rate",
                &m.agent_id,
                m.success_rate(),
            ));
        }

        // ── brainwires_agent_tool_calls_total ────────────────────────────────
        counter(
            &mut out,
            "brainwires_agent_tool_calls_total",
            "Total tool calls made by agent",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_tool_calls_total",
                &m.agent_id,
                m.total_tool_calls,
            ));
        }

        // ── brainwires_agent_tool_errors_total ───────────────────────────────
        counter(
            &mut out,
            "brainwires_agent_tool_errors_total",
            "Tool calls that produced an error",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_tool_errors_total",
                &m.agent_id,
                m.tool_error_count,
            ));
        }

        // ── brainwires_agent_provider_calls_total ────────────────────────────
        counter(
            &mut out,
            "brainwires_agent_provider_calls_total",
            "Total LLM provider calls",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_provider_calls_total",
                &m.agent_id,
                m.provider_call_count,
            ));
        }

        // ── brainwires_agent_tokens_prompt_total ─────────────────────────────
        counter(
            &mut out,
            "brainwires_agent_tokens_prompt_total",
            "Total prompt tokens consumed",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_tokens_prompt_total",
                &m.agent_id,
                m.total_tokens_prompt,
            ));
        }

        // ── brainwires_agent_tokens_completion_total ─────────────────────────
        counter(
            &mut out,
            "brainwires_agent_tokens_completion_total",
            "Total completion tokens generated",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_tokens_completion_total",
                &m.agent_id,
                m.total_tokens_completion,
            ));
        }

        // ── brainwires_agent_cost_usd_total ──────────────────────────────────
        counter(
            &mut out,
            "brainwires_agent_cost_usd_total",
            "Cumulative LLM cost in USD",
        );
        for m in metrics.values() {
            out.push_str(&metric_line_f(
                "brainwires_agent_cost_usd_total",
                &m.agent_id,
                m.total_cost_usd,
            ));
        }

        // ── brainwires_agent_avg_run_duration_ms ─────────────────────────────
        gauge(
            &mut out,
            "brainwires_agent_avg_run_duration_ms",
            "Average agent run duration in milliseconds",
        );
        for m in metrics.values() {
            out.push_str(&metric_line_f(
                "brainwires_agent_avg_run_duration_ms",
                &m.agent_id,
                m.avg_run_duration_ms(),
            ));
        }

        // ── brainwires_agent_avg_provider_latency_ms ─────────────────────────
        gauge(
            &mut out,
            "brainwires_agent_avg_provider_latency_ms",
            "Average LLM provider call latency in ms",
        );
        for m in metrics.values() {
            out.push_str(&metric_line_f(
                "brainwires_agent_avg_provider_latency_ms",
                &m.agent_id,
                m.avg_provider_latency_ms(),
            ));
        }

        // ── brainwires_agent_cache_read_tokens_total ─────────────────────────
        counter(
            &mut out,
            "brainwires_agent_cache_read_tokens_total",
            "Prompt tokens served from the provider's cache",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_cache_read_tokens_total",
                &m.agent_id,
                m.total_cache_read_tokens,
            ));
        }

        // ── brainwires_agent_cache_creation_tokens_total ─────────────────────
        counter(
            &mut out,
            "brainwires_agent_cache_creation_tokens_total",
            "Prompt tokens charged to populate the provider's cache",
        );
        for m in metrics.values() {
            out.push_str(&metric_line(
                "brainwires_agent_cache_creation_tokens_total",
                &m.agent_id,
                m.total_cache_creation_tokens,
            ));
        }

        // ── brainwires_agent_cache_hit_rate ──────────────────────────────────
        gauge(
            &mut out,
            "brainwires_agent_cache_hit_rate",
            "Fraction of input tokens served from cache (0-1)",
        );
        for m in metrics.values() {
            out.push_str(&metric_line_f(
                "brainwires_agent_cache_hit_rate",
                &m.agent_id,
                m.cache_hit_rate(),
            ));
        }

        out
    }

    // ── Internal update ───────────────────────────────────────────────────────

    fn update(&self, event: &AnalyticsEvent) {
        let mut map = self.inner.lock().expect("metrics registry lock poisoned");

        match event {
            AnalyticsEvent::AgentRun {
                agent_id,
                success,
                total_tool_calls,
                tool_error_count,
                total_iterations,
                duration_ms,
                total_cost_usd,
                total_prompt_tokens,
                total_completion_tokens,
                ..
            } => {
                let m = map
                    .entry(agent_id.clone())
                    .or_insert_with(|| OutcomeMetrics {
                        agent_id: agent_id.clone(),
                        ..Default::default()
                    });
                m.total_runs += 1;
                if *success {
                    m.success_count += 1;
                } else {
                    m.failure_count += 1;
                }
                m.total_tool_calls += *total_tool_calls as u64;
                m.tool_error_count += *tool_error_count as u64;
                m.total_iterations += *total_iterations as u64;
                m.total_run_duration_ms += *duration_ms;
                m.total_cost_usd += *total_cost_usd;
                m.total_tokens_prompt += *total_prompt_tokens as u64;
                m.total_tokens_completion += *total_completion_tokens as u64;
            }

            AnalyticsEvent::ProviderCall {
                session_id,
                prompt_tokens,
                completion_tokens,
                duration_ms,
                cost_usd,
                cache_creation_input_tokens,
                cache_read_input_tokens,
                ..
            } => {
                // ProviderCall doesn't carry an agent_id directly; use session_id
                // as a best-effort key so provider stats are still tracked.
                let key = session_id
                    .clone()
                    .unwrap_or_else(|| "__global__".to_string());
                let m = map.entry(key.clone()).or_insert_with(|| OutcomeMetrics {
                    agent_id: key,
                    ..Default::default()
                });
                m.provider_call_count += 1;
                m.total_tokens_prompt += *prompt_tokens as u64;
                m.total_tokens_completion += *completion_tokens as u64;
                m.total_provider_duration_ms += *duration_ms;
                m.total_cost_usd += *cost_usd;
                m.total_cache_creation_tokens += *cache_creation_input_tokens as u64;
                m.total_cache_read_tokens += *cache_read_input_tokens as u64;
            }

            AnalyticsEvent::ToolCall {
                agent_id, is_error, ..
            } => {
                let key = agent_id.clone().unwrap_or_else(|| "__global__".to_string());
                let m = map.entry(key.clone()).or_insert_with(|| OutcomeMetrics {
                    agent_id: key,
                    ..Default::default()
                });
                m.total_tool_calls += 1;
                if *is_error {
                    m.tool_error_count += 1;
                }
            }

            // Other event types don't contribute to outcome metrics.
            _ => {}
        }
    }
}

impl std::fmt::Debug for MetricsRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let map = self.inner.lock().expect("metrics registry lock poisoned");
        f.debug_struct("MetricsRegistry")
            .field("agent_count", &map.len())
            .finish()
    }
}

#[async_trait]
impl AnalyticsSink for MetricsRegistry {
    async fn record(&self, event: AnalyticsEvent) -> Result<(), AnalyticsError> {
        self.update(&event);
        Ok(())
    }
}

// ── Prometheus formatting helpers ─────────────────────────────────────────────

fn escape_label(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn metric_line(name: &str, agent_id: &str, value: u64) -> String {
    format!(
        "{name}{{agent_id=\"{}\"}} {value}\n",
        escape_label(agent_id)
    )
}

fn metric_line_f(name: &str, agent_id: &str, value: f64) -> String {
    format!(
        "{name}{{agent_id=\"{}\"}} {value:.6}\n",
        escape_label(agent_id)
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn agent_run(agent_id: &str, success: bool, cost: f64) -> AnalyticsEvent {
        AnalyticsEvent::AgentRun {
            session_id: None,
            agent_id: agent_id.to_string(),
            task_id: "t1".to_string(),
            prompt_hash: "abc".to_string(),
            success,
            total_iterations: 3,
            total_tool_calls: 2,
            tool_error_count: 0,
            tools_used: vec![],
            total_prompt_tokens: 100,
            total_completion_tokens: 50,
            total_cost_usd: cost,
            duration_ms: 1500,
            failure_category: None,
            timestamp: Utc::now(),
            compliance: None,
        }
    }

    #[tokio::test]
    async fn counts_agent_runs() {
        let reg = MetricsRegistry::new();
        reg.record(agent_run("agent-1", true, 0.01)).await.unwrap();
        reg.record(agent_run("agent-1", true, 0.01)).await.unwrap();
        reg.record(agent_run("agent-1", false, 0.0)).await.unwrap();

        let m = reg.get("agent-1").unwrap();
        assert_eq!(m.total_runs, 3);
        assert_eq!(m.success_count, 2);
        assert_eq!(m.failure_count, 1);
        assert!((m.success_rate() - 2.0 / 3.0).abs() < 1e-9);
        assert!((m.total_cost_usd - 0.02).abs() < 1e-9);
    }

    #[tokio::test]
    async fn prometheus_text_well_formed() {
        let reg = MetricsRegistry::new();
        reg.record(agent_run("code-review", true, 0.005))
            .await
            .unwrap();

        let text = reg.prometheus_text();
        assert!(text.contains("# HELP brainwires_agent_runs_total"));
        assert!(text.contains("# TYPE brainwires_agent_runs_total counter"));
        assert!(text.contains("brainwires_agent_runs_total{agent_id=\"code-review\"} 1"));
        assert!(text.contains("brainwires_agent_success_rate{agent_id=\"code-review\"} 1.000000"));
        assert!(text.contains("brainwires_agent_cost_usd_total{agent_id=\"code-review\"}"));
    }

    #[tokio::test]
    async fn empty_registry_returns_empty_string() {
        let reg = MetricsRegistry::new();
        assert_eq!(reg.prometheus_text(), "");
    }

    #[tokio::test]
    async fn reset_clears_metrics() {
        let reg = MetricsRegistry::new();
        reg.record(agent_run("a1", true, 0.01)).await.unwrap();
        reg.reset("a1");
        assert!(reg.get("a1").is_none());
    }

    fn provider_call(
        session: &str,
        prompt: u32,
        completion: u32,
        cache_read: u32,
        cache_creation: u32,
    ) -> AnalyticsEvent {
        AnalyticsEvent::ProviderCall {
            session_id: Some(session.to_string()),
            provider: "anthropic".into(),
            model: "claude-sonnet-4-6".into(),
            prompt_tokens: prompt,
            completion_tokens: completion,
            duration_ms: 100,
            cost_usd: 0.001,
            success: true,
            timestamp: Utc::now(),
            cache_creation_input_tokens: cache_creation,
            cache_read_input_tokens: cache_read,
            compliance: None,
        }
    }

    #[tokio::test]
    async fn cache_metrics_accumulate_and_compute_hit_rate() {
        let reg = MetricsRegistry::new();
        // Call 1: 100 fresh prompt tokens, no cache.
        reg.record(provider_call("s1", 100, 50, 0, 0))
            .await
            .unwrap();
        // Call 2: 10 fresh + 90 from cache, 0 creation.
        reg.record(provider_call("s1", 10, 50, 90, 0))
            .await
            .unwrap();
        // Call 3: 5 fresh, 95 read, 200 written.
        reg.record(provider_call("s1", 5, 50, 95, 200))
            .await
            .unwrap();

        let m = reg.get("s1").expect("session tracked");
        assert_eq!(m.total_tokens_prompt, 115);
        assert_eq!(m.total_cache_read_tokens, 185);
        assert_eq!(m.total_cache_creation_tokens, 200);

        // hit_rate = cache_read / (prompt + cache_read) = 185 / 300
        let expected = 185.0 / 300.0;
        assert!((m.cache_hit_rate() - expected).abs() < 1e-9);

        let text = reg.prometheus_text();
        assert!(text.contains("brainwires_agent_cache_read_tokens_total{agent_id=\"s1\"} 185"));
        assert!(text.contains("brainwires_agent_cache_creation_tokens_total{agent_id=\"s1\"} 200"));
        assert!(text.contains("brainwires_agent_cache_hit_rate{agent_id=\"s1\"}"));
    }

    #[test]
    fn cache_hit_rate_zero_when_no_tokens() {
        let m = OutcomeMetrics {
            agent_id: "x".into(),
            ..Default::default()
        };
        assert_eq!(m.cache_hit_rate(), 0.0);
    }

    #[test]
    fn outcome_metrics_computed_fields() {
        let m = OutcomeMetrics {
            agent_id: "a".to_string(),
            total_runs: 10,
            success_count: 7,
            failure_count: 3,
            total_tool_calls: 20,
            tool_error_count: 4,
            total_cost_usd: 1.0,
            total_run_duration_ms: 5000,
            provider_call_count: 5,
            total_provider_duration_ms: 1000,
            ..Default::default()
        };
        assert!((m.success_rate() - 0.7).abs() < 1e-9);
        assert!((m.tool_error_rate() - 0.2).abs() < 1e-9);
        assert!((m.avg_cost_per_run_usd() - 0.1).abs() < 1e-9);
        assert!((m.avg_run_duration_ms() - 500.0).abs() < 1e-9);
        assert!((m.avg_provider_latency_ms() - 200.0).abs() < 1e-9);
    }
}
