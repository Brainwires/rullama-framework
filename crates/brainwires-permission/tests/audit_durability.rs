//! Integration tests for `AuditLogger` on-disk durability and replay.
//!
//! The audit log is the tamper-evident record for every policy-gated action.
//! Two properties matter most:
//!
//! 1. **No drop**. Important events (policy violations, human interventions,
//!    trust changes, user feedback) must hit disk *before* `log()` returns —
//!    a crash between buffer add and flush must not lose them.
//! 2. **Replay-safe**. A fresh logger pointed at an existing log must be able
//!    to read back every event that was previously written, in a
//!    newest-first order.
//!
//! These tests drive a `tempdir`-backed logger so they're deterministic and
//! parallel-safe.

use brainwires_permission::audit::{
    ActionOutcome, AuditEvent, AuditEventType, AuditLogger, AuditQuery,
};
use std::fs;
use tempfile::TempDir;

fn fresh_logger() -> (AuditLogger, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("audit.jsonl");
    let logger = AuditLogger::with_path(path).expect("logger");
    (logger, dir)
}

#[test]
fn important_event_is_written_to_disk_immediately() {
    let (logger, dir) = fresh_logger();

    logger
        .log(
            AuditEvent::new(AuditEventType::PolicyViolation)
                .with_agent("agent-1")
                .with_action("read_secret")
                .with_outcome(ActionOutcome::Denied),
        )
        .unwrap();

    // No flush() call — the important-event path must have written directly.
    let contents = fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
    assert!(
        contents.contains("\"policy_violation\""),
        "expected event on disk, got: {contents:?}",
    );
    assert_eq!(
        contents.lines().count(),
        1,
        "one event should produce exactly one JSONL line",
    );
}

#[test]
fn ordinary_event_stays_in_buffer_until_flush() {
    let (logger, dir) = fresh_logger();

    logger
        .log(
            AuditEvent::new(AuditEventType::ToolExecution)
                .with_agent("a1")
                .with_action("read_file")
                .with_outcome(ActionOutcome::Success),
        )
        .unwrap();

    // Log file either does not exist yet or is empty — event is buffered.
    let path = dir.path().join("audit.jsonl");
    let contents = fs::read_to_string(&path).unwrap_or_default();
    assert!(
        contents.is_empty(),
        "ordinary event should not flush to disk before explicit flush(), got: {contents:?}",
    );

    logger.flush().unwrap();

    let contents = fs::read_to_string(&path).unwrap();
    assert!(contents.contains("\"tool_execution\""));
    assert!(contents.contains("\"read_file\""));
}

#[test]
fn flush_preserves_insertion_order_on_disk() {
    let (logger, dir) = fresh_logger();

    for i in 0..5 {
        logger
            .log(
                AuditEvent::new(AuditEventType::ToolExecution)
                    .with_agent(&format!("a{i}"))
                    .with_action(&format!("act_{i}"))
                    .with_outcome(ActionOutcome::Success),
            )
            .unwrap();
    }
    logger.flush().unwrap();

    let contents = fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    assert_eq!(lines.len(), 5);
    for (i, line) in lines.iter().enumerate() {
        assert!(
            line.contains(&format!("\"act_{i}\"")),
            "line {i} should contain act_{i}, got {line}",
        );
    }
}

#[test]
fn query_sorts_newest_first_and_respects_limit() {
    let (logger, _dir) = fresh_logger();

    for i in 0..10 {
        logger
            .log(
                AuditEvent::new(AuditEventType::ToolExecution)
                    .with_agent("a1")
                    .with_action(&format!("op_{i}"))
                    .with_outcome(ActionOutcome::Success),
            )
            .unwrap();
        // spread timestamps so the sort is meaningful
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    logger.flush().unwrap();

    let results = logger.query(&AuditQuery::new().limit(3)).unwrap();
    assert_eq!(results.len(), 3);
    // Newest first: timestamps must be strictly non-increasing.
    for pair in results.windows(2) {
        assert!(pair[0].timestamp >= pair[1].timestamp);
    }
}

#[test]
fn query_filters_by_agent_and_outcome() {
    let (logger, _dir) = fresh_logger();

    logger
        .log_tool_execution(
            Some("alice"),
            "read_file",
            Some("/x"),
            ActionOutcome::Success,
            None,
        )
        .unwrap();
    logger
        .log_tool_execution(
            Some("bob"),
            "read_file",
            Some("/x"),
            ActionOutcome::Success,
            None,
        )
        .unwrap();
    logger
        .log_tool_execution(
            Some("alice"),
            "write_file",
            Some("/x"),
            ActionOutcome::Failure,
            None,
        )
        .unwrap();
    logger.flush().unwrap();

    let alice_ok = logger
        .query(
            &AuditQuery::new()
                .for_agent("alice")
                .with_outcome(ActionOutcome::Success),
        )
        .unwrap();
    assert_eq!(alice_ok.len(), 1);
    assert_eq!(alice_ok[0].action, "read_file");
}

#[test]
fn new_logger_replays_events_from_previous_session() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("audit.jsonl");

    // Session 1: write, flush, drop.
    {
        let logger = AuditLogger::with_path(path.clone()).unwrap();
        logger
            .log(
                AuditEvent::new(AuditEventType::TrustChange)
                    .with_agent("agent-A")
                    .with_action("trust_change")
                    .with_metadata("new_level", "3")
                    .with_outcome(ActionOutcome::Success),
            )
            .unwrap();
        // TrustChange is important → flushed immediately; no explicit flush needed.
    }

    // Session 2: fresh logger, same path — must see the old event.
    let logger2 = AuditLogger::with_path(path).unwrap();
    let events = logger2
        .query(&AuditQuery::new().of_type(AuditEventType::TrustChange))
        .unwrap();
    assert_eq!(
        events.len(),
        1,
        "new logger must replay prior-session events"
    );
    assert_eq!(events[0].agent_id.as_deref(), Some("agent-A"));
    assert_eq!(
        events[0].metadata.get("new_level").map(|s| s.as_str()),
        Some("3")
    );
}

#[test]
fn audit_log_is_valid_jsonl_after_mixed_event_types() {
    // Mix of important (immediate) and ordinary (buffered) events must
    // produce a well-formed JSONL file once everything is flushed.
    let (logger, dir) = fresh_logger();

    logger
        .log(
            AuditEvent::new(AuditEventType::ToolExecution)
                .with_action("read_file")
                .with_outcome(ActionOutcome::Success),
        )
        .unwrap();
    logger
        .log(
            AuditEvent::new(AuditEventType::PolicyViolation)
                .with_action("delete_file")
                .with_outcome(ActionOutcome::Denied),
        )
        .unwrap();
    logger
        .log(
            AuditEvent::new(AuditEventType::HumanIntervention)
                .with_action("approve_write")
                .with_outcome(ActionOutcome::Approved),
        )
        .unwrap();
    logger.flush().unwrap();

    let contents = fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
    assert_eq!(contents.lines().count(), 3);
    for line in contents.lines() {
        serde_json::from_str::<AuditEvent>(line)
            .unwrap_or_else(|e| panic!("malformed JSONL line `{line}`: {e}"));
    }
}

#[test]
fn disabled_logger_is_silent() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("audit.jsonl");
    let mut logger = AuditLogger::with_path(path.clone()).unwrap();
    logger.set_enabled(false);

    // Even important event types should not hit disk while disabled.
    logger
        .log(
            AuditEvent::new(AuditEventType::PolicyViolation)
                .with_action("read_secret")
                .with_outcome(ActionOutcome::Denied),
        )
        .unwrap();

    assert!(
        !path.exists() || fs::read_to_string(&path).unwrap().is_empty(),
        "disabled logger must not write",
    );
}
