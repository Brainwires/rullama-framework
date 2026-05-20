//! OpenTelemetry export for agent execution traces
//!
//! Maps [`ExecutionGraph`] and [`RunTelemetry`] to OpenTelemetry spans for
//! integration with Jaeger, Datadog, Grafana, and other observability platforms.
//!
//! # Feature Gate
//!
//! This module is only available when the `otel` feature is enabled:
//!
//! ```toml
//! brainwires-agent = { version = "0.10", features = ["otel"] }
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use brainwires_agent::otel::export_to_otel;
//! use opentelemetry::global;
//!
//! let tracer = global::tracer("brainwires-agent");
//! export_to_otel(&execution_graph, &telemetry, &tracer);
//! ```

use opentelemetry::{
    KeyValue,
    trace::{Span, SpanKind, Status, Tracer},
};

use crate::execution_graph::{ExecutionGraph, RunTelemetry};

/// Export an agent execution trace to OpenTelemetry spans.
///
/// Creates a hierarchical span structure:
/// - Root span: `agent.run` — covers the entire execution
///   - Child spans: `agent.iteration.{N}` — one per provider call
///     - Grandchild spans: `agent.tool.{name}` — one per tool call
///
/// All token counts, costs, timing, and error information are attached as
/// span attributes.
pub fn export_to_otel<T: Tracer>(graph: &ExecutionGraph, telemetry: &RunTelemetry, tracer: &T) {
    let mut root_span = tracer
        .span_builder("agent.run")
        .with_kind(SpanKind::Internal)
        .with_attributes(vec![
            KeyValue::new("agent.prompt_hash", telemetry.prompt_hash.clone()),
            KeyValue::new("agent.total_iterations", telemetry.total_iterations as i64),
            KeyValue::new("agent.total_tool_calls", telemetry.total_tool_calls as i64),
            KeyValue::new("agent.tool_error_count", telemetry.tool_error_count as i64),
            KeyValue::new(
                "agent.total_prompt_tokens",
                telemetry.total_prompt_tokens as i64,
            ),
            KeyValue::new(
                "agent.total_completion_tokens",
                telemetry.total_completion_tokens as i64,
            ),
            KeyValue::new("agent.total_cost_usd", telemetry.total_cost_usd),
            KeyValue::new("agent.duration_ms", telemetry.duration_ms as i64),
            KeyValue::new("agent.success", telemetry.success),
            KeyValue::new("agent.tools_used", telemetry.tools_used.join(",")),
        ])
        .start(tracer);

    if !telemetry.success {
        root_span.set_status(Status::error("Agent execution failed"));
    }

    // Create child spans for each iteration
    for step in &graph.steps {
        let mut step_span = tracer
            .span_builder(format!("agent.iteration.{}", step.iteration))
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![
                KeyValue::new("iteration.number", step.iteration as i64),
                KeyValue::new("iteration.prompt_tokens", step.prompt_tokens as i64),
                KeyValue::new("iteration.completion_tokens", step.completion_tokens as i64),
                KeyValue::new(
                    "iteration.finish_reason",
                    step.finish_reason.clone().unwrap_or_default(),
                ),
                KeyValue::new("iteration.tool_count", step.tool_calls.len() as i64),
            ])
            .start(tracer);

        // Create grandchild spans for each tool call
        for tc in &step.tool_calls {
            let mut tool_span = tracer
                .span_builder(format!("agent.tool.{}", tc.tool_name))
                .with_kind(SpanKind::Internal)
                .with_attributes(vec![
                    KeyValue::new("tool.name", tc.tool_name.clone()),
                    KeyValue::new("tool.use_id", tc.tool_use_id.clone()),
                    KeyValue::new("tool.is_error", tc.is_error),
                ])
                .start(tracer);

            if tc.is_error {
                tool_span.set_status(Status::error("Tool call failed"));
            }

            tool_span.end();
        }

        step_span.end();
    }

    root_span.end();
}

/// Create OpenTelemetry span attributes from a [`RunTelemetry`] record.
///
/// Useful when you want to attach telemetry data to an existing span
/// rather than creating new spans.
pub fn telemetry_attributes(telemetry: &RunTelemetry) -> Vec<KeyValue> {
    vec![
        KeyValue::new("agent.prompt_hash", telemetry.prompt_hash.clone()),
        KeyValue::new("agent.total_iterations", telemetry.total_iterations as i64),
        KeyValue::new("agent.total_tool_calls", telemetry.total_tool_calls as i64),
        KeyValue::new("agent.tool_error_count", telemetry.tool_error_count as i64),
        KeyValue::new(
            "agent.total_prompt_tokens",
            telemetry.total_prompt_tokens as i64,
        ),
        KeyValue::new(
            "agent.total_completion_tokens",
            telemetry.total_completion_tokens as i64,
        ),
        KeyValue::new("agent.total_cost_usd", telemetry.total_cost_usd),
        KeyValue::new("agent.duration_ms", telemetry.duration_ms as i64),
        KeyValue::new("agent.success", telemetry.success),
    ]
}
