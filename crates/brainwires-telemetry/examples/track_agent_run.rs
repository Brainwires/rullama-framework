//! Analytics tracking example.
//!
//! Demonstrates how to:
//! 1. Create an `AnalyticsCollector` backed by a `MemoryAnalyticsSink`
//! 2. Emit `ProviderCall`, `ToolCall`, and `AgentRun` events explicitly
//! 3. Flush and inspect the buffered events
//!
//! With the `sqlite` feature you can swap in `SqliteAnalyticsSink` for
//! persistent storage and use `AnalyticsQuery` to compute cost-by-model
//! summaries.
//!
//! ```bash
//! cargo run -p brainwires-analytics --example track_agent_run
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use brainwires_telemetry::{
    AnalyticsCollector, AnalyticsError, AnalyticsEvent, AnalyticsSink, MemoryAnalyticsSink,
};
use chrono::Utc;

// ── Shared-sink wrapper ───────────────────────────────────────────────────────
//
// The collector owns `Box<dyn AnalyticsSink>`, so we need a thin wrapper that
// keeps an `Arc` to the underlying `MemoryAnalyticsSink` so we can read
// the buffer after flushing.

struct SharedSink(Arc<MemoryAnalyticsSink>);

#[async_trait]
impl AnalyticsSink for SharedSink {
    async fn record(&self, event: AnalyticsEvent) -> Result<(), AnalyticsError> {
        self.0.record(event).await
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── 1. Build the collector ────────────────────────────────────────────────

    let mem = Arc::new(MemoryAnalyticsSink::new(1_000));
    let collector = AnalyticsCollector::new(vec![Box::new(SharedSink(Arc::clone(&mem)))]);

    // ── 2. Simulate an agent run ──────────────────────────────────────────────

    let session = Some("sess-demo-001".to_string());

    // Provider call (e.g. Anthropic Claude)
    collector.record(AnalyticsEvent::ProviderCall {
        session_id: session.clone(),
        provider: "anthropic".to_string(),
        model: "claude-sonnet-4-6".to_string(),
        prompt_tokens: 512,
        completion_tokens: 128,
        duration_ms: 843,
        cost_usd: 0.002,
        success: true,
        timestamp: Utc::now(),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        compliance: None,
    });

    // Two tool calls
    for (name, id) in [("read_file", "use-001"), ("bash", "use-002")] {
        collector.record(AnalyticsEvent::ToolCall {
            session_id: session.clone(),
            agent_id: Some("agent-main".to_string()),
            tool_name: name.to_string(),
            tool_use_id: id.to_string(),
            is_error: false,
            duration_ms: Some(12),
            timestamp: Utc::now(),
        });
    }

    // Agent run summary
    collector.record(AnalyticsEvent::AgentRun {
        session_id: session.clone(),
        agent_id: "agent-main".to_string(),
        task_id: "task-0001".to_string(),
        prompt_hash: "abc123".to_string(),
        success: true,
        total_iterations: 3,
        total_tool_calls: 2,
        tool_error_count: 0,
        tools_used: vec!["read_file".to_string(), "bash".to_string()],
        total_prompt_tokens: 512,
        total_completion_tokens: 128,
        total_cost_usd: 0.002,
        duration_ms: 1_200,
        failure_category: None,
        timestamp: Utc::now(),
        compliance: None,
    });

    // ── 3. Flush and inspect ──────────────────────────────────────────────────

    // flush() waits for all queued events to be delivered to sinks
    collector.flush().await?;

    let events = mem.snapshot();
    println!("Recorded {} analytics events:\n", events.len());

    for (i, event) in events.iter().enumerate() {
        println!("  [{}] type={}", i + 1, event.event_type());
        if let Some(sid) = event.session_id() {
            println!("       session={sid}");
        }
        match event {
            AnalyticsEvent::ProviderCall {
                provider,
                model,
                cost_usd,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                println!(
                    "       {provider}/{model}  {prompt_tokens}+{completion_tokens} tokens  ${cost_usd:.4}"
                );
            }
            AnalyticsEvent::ToolCall {
                tool_name,
                is_error,
                ..
            } => {
                println!("       tool={tool_name}  error={is_error}");
            }
            AnalyticsEvent::AgentRun {
                success,
                total_cost_usd,
                duration_ms,
                tools_used,
                ..
            } => {
                println!(
                    "       success={success}  cost=${total_cost_usd:.4}  duration={duration_ms}ms  tools={tools_used:?}"
                );
            }
            _ => {}
        }
    }

    // With `sqlite` feature you can persist and query:
    //
    //   let sink = SqliteAnalyticsSink::new_default()?;
    //   let query = AnalyticsQuery::new_default()?;
    //   query.rebuild_summaries()?;
    //   for row in query.cost_by_model(None, None)? {
    //       println!("{}: ${:.4}", row.model, row.total_cost_usd);
    //   }

    println!("\nDone.");
    Ok(())
}
