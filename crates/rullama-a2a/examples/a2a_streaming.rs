//! A2A Streaming Types — SSE event construction and serialization.
//!
//! Demonstrates building and serializing the core streaming types:
//! - `TaskStatusUpdateEvent` — task state transitions
//! - `TaskArtifactUpdateEvent` — incremental artifact delivery
//! - `StreamResponse` — the SSE envelope wrapping status/artifact/message events
//!
//! This example works entirely with in-memory data; no running server is needed.
//!
//! ```bash
//! cargo run -p rullama-a2a --example a2a_streaming --features "client,server"
//! ```

use rullama_a2a::{
    Artifact, Message, Part, Role, StreamResponse, Task, TaskArtifactUpdateEvent, TaskState,
    TaskStatus, TaskStatusUpdateEvent,
};

fn main() {
    println!("=== A2A Streaming Types Example ===\n");

    let task_id = uuid::Uuid::new_v4().to_string();
    let context_id = uuid::Uuid::new_v4().to_string();

    // -----------------------------------------------------------------------
    // 1. TaskStatusUpdateEvent — submitted
    // -----------------------------------------------------------------------
    println!("--- TaskStatusUpdateEvent (submitted) ---");

    let submitted_event = TaskStatusUpdateEvent {
        task_id: task_id.clone(),
        context_id: context_id.clone(),
        status: TaskStatus {
            state: TaskState::Submitted,
            message: None,
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
        },
        trace_id: None,
        sequence: None,
        metadata: None,
    };

    let json = serde_json::to_string_pretty(&submitted_event).unwrap();
    println!("{json}\n");

    // -----------------------------------------------------------------------
    // 2. TaskStatusUpdateEvent — working (with message)
    // -----------------------------------------------------------------------
    println!("--- TaskStatusUpdateEvent (working) ---");

    let working_event = TaskStatusUpdateEvent {
        task_id: task_id.clone(),
        context_id: context_id.clone(),
        status: TaskStatus {
            state: TaskState::Working,
            message: Some(Message::agent_text("Processing your request...")),
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
        },
        trace_id: None,
        sequence: None,
        metadata: None,
    };

    let json = serde_json::to_string_pretty(&working_event).unwrap();
    println!("{json}\n");

    // -----------------------------------------------------------------------
    // 3. TaskArtifactUpdateEvent — first chunk
    // -----------------------------------------------------------------------
    println!("--- TaskArtifactUpdateEvent (chunk 1) ---");

    let artifact_event_1 = TaskArtifactUpdateEvent {
        task_id: task_id.clone(),
        context_id: context_id.clone(),
        artifact: Artifact {
            artifact_id: "report-001".to_string(),
            name: Some("analysis-report".to_string()),
            description: Some("Code analysis report".to_string()),
            parts: vec![Part {
                text: Some("## Code Analysis\n\nAnalyzing module structure...".to_string()),
                raw: None,
                url: None,
                data: None,
                media_type: Some("text/markdown".to_string()),
                filename: Some("report.md".to_string()),
                metadata: None,
            }],
            metadata: None,
            extensions: None,
        },
        index: Some(0),
        append: Some(false),
        last_chunk: Some(false),
        trace_id: None,
        sequence: None,
        metadata: None,
    };

    let json = serde_json::to_string_pretty(&artifact_event_1).unwrap();
    println!("{json}\n");

    // -----------------------------------------------------------------------
    // 4. TaskArtifactUpdateEvent — final chunk (append)
    // -----------------------------------------------------------------------
    println!("--- TaskArtifactUpdateEvent (chunk 2, final) ---");

    let artifact_event_2 = TaskArtifactUpdateEvent {
        task_id: task_id.clone(),
        context_id: context_id.clone(),
        artifact: Artifact {
            artifact_id: "report-001".to_string(),
            name: None,
            description: None,
            parts: vec![Part {
                text: Some("\n### Summary\n\nAll checks passed. No issues found.".to_string()),
                raw: None,
                url: None,
                data: None,
                media_type: Some("text/markdown".to_string()),
                filename: None,
                metadata: None,
            }],
            metadata: None,
            extensions: None,
        },
        index: Some(0),
        append: Some(true),
        last_chunk: Some(true),
        trace_id: None,
        sequence: None,
        metadata: None,
    };

    let json = serde_json::to_string_pretty(&artifact_event_2).unwrap();
    println!("{json}\n");

    // -----------------------------------------------------------------------
    // 5. StreamResponse — wrapping a status update
    // -----------------------------------------------------------------------
    println!("--- StreamResponse (status update) ---");

    let status_stream = StreamResponse {
        task: None,
        message: None,
        status_update: Some(working_event.clone()),
        artifact_update: None,
    };

    let json = serde_json::to_string_pretty(&status_stream).unwrap();
    println!("{json}\n");

    // -----------------------------------------------------------------------
    // 6. StreamResponse — wrapping an artifact update
    // -----------------------------------------------------------------------
    println!("--- StreamResponse (artifact update) ---");

    let artifact_stream = StreamResponse {
        task: None,
        message: None,
        status_update: None,
        artifact_update: Some(artifact_event_1.clone()),
    };

    let json = serde_json::to_string_pretty(&artifact_stream).unwrap();
    println!("{json}\n");

    // -----------------------------------------------------------------------
    // 7. StreamResponse — wrapping a standalone message
    // -----------------------------------------------------------------------
    println!("--- StreamResponse (agent message) ---");

    let agent_msg = Message {
        message_id: uuid::Uuid::new_v4().to_string(),
        role: Role::Agent,
        parts: vec![Part {
            text: Some("Here is an intermediate status update.".to_string()),
            raw: None,
            url: None,
            data: None,
            media_type: None,
            filename: None,
            metadata: None,
        }],
        context_id: Some(context_id.clone()),
        task_id: Some(task_id.clone()),
        reference_task_ids: None,
        metadata: None,
        extensions: None,
    };

    let message_stream = StreamResponse {
        task: None,
        message: Some(agent_msg),
        status_update: None,
        artifact_update: None,
    };

    let json = serde_json::to_string_pretty(&message_stream).unwrap();
    println!("{json}\n");

    // -----------------------------------------------------------------------
    // 8. StreamResponse — full task snapshot (final event)
    // -----------------------------------------------------------------------
    println!("--- StreamResponse (full task snapshot) ---");

    let completed_event = TaskStatusUpdateEvent {
        task_id: task_id.clone(),
        context_id: context_id.clone(),
        status: TaskStatus {
            state: TaskState::Completed,
            message: Some(Message::agent_text("Analysis complete.")),
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
        },
        trace_id: None,
        sequence: None,
        metadata: None,
    };

    let full_task = Task {
        id: task_id.clone(),
        context_id: Some(context_id.clone()),
        status: completed_event.status.clone(),
        artifacts: Some(vec![Artifact {
            artifact_id: "report-001".to_string(),
            name: Some("analysis-report".to_string()),
            description: Some("Code analysis report".to_string()),
            parts: vec![Part {
                text: Some("## Code Analysis\n\n...full report...".to_string()),
                raw: None,
                url: None,
                data: None,
                media_type: Some("text/markdown".to_string()),
                filename: Some("report.md".to_string()),
                metadata: None,
            }],
            metadata: None,
            extensions: None,
        }]),
        history: None,
        metadata: None,
    };

    let task_stream = StreamResponse {
        task: Some(full_task),
        message: None,
        status_update: None,
        artifact_update: None,
    };

    let json = serde_json::to_string_pretty(&task_stream).unwrap();
    println!("{json}\n");

    // -----------------------------------------------------------------------
    // 9. Round-trip verification
    // -----------------------------------------------------------------------
    println!("--- Round-Trip Verification ---");

    let events: Vec<StreamResponse> =
        vec![status_stream, artifact_stream, message_stream, task_stream];

    for (i, event) in events.iter().enumerate() {
        let serialized = serde_json::to_string(event).unwrap();
        let deserialized: StreamResponse = serde_json::from_str(&serialized).unwrap();
        assert_eq!(&deserialized, event, "round-trip mismatch on event {i}");
        println!("  Event {i}: round-trip OK");
    }

    println!("\nDone.");
}
