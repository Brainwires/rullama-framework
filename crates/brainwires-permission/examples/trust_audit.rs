//! Trust & Audit Example
//!
//! Demonstrates trust level management with `TrustManager` and audit event
//! logging with `AuditLogger`. Records successes and violations for an agent,
//! observes trust level changes, then queries the audit log for statistics.
//!
//! Run: cargo run -p brainwires-permission --features native --example trust_audit

use anyhow::Result;
use brainwires_permission::{
    ActionOutcome, AuditEvent, AuditEventType, AuditLogger, AuditQuery, TrustManager,
    ViolationSeverity,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Trust & Audit Example ===\n");

    let temp_dir = tempfile::tempdir()?;

    // 1. Setup
    println!("--- 1. Setup ---\n");

    let trust_path = temp_dir.path().join("trust_store.json");
    let audit_path = temp_dir.path().join("audit.jsonl");

    let mut trust = TrustManager::with_path(trust_path)?;
    let logger = AuditLogger::with_path(audit_path)?;

    println!("  TrustManager created (persisted to temp dir)");
    println!("  AuditLogger created (persisted to temp dir)");

    // 2. Record successes and build trust
    println!("\n--- 2. Recording successes for agent-A ---\n");

    for i in 1..=10 {
        trust.record_success("agent-A");
        logger.log(
            AuditEvent::new(AuditEventType::ToolExecution)
                .with_agent("agent-A")
                .with_action("write_file")
                .with_target(&format!("src/module_{i}.rs"))
                .with_outcome(ActionOutcome::Success),
        )?;
    }

    let level = trust.get_trust_level("agent-A");
    let factor = trust.get("agent-A").unwrap();
    println!(
        "  After 10 successes: level={}, score={:.2}, ops={}",
        level, factor.score, factor.total_ops,
    );

    // 3. Record a violation and observe trust decrease
    println!("\n--- 3. Recording a major violation ---\n");

    let score_before = trust.get("agent-A").unwrap().score;

    trust.record_violation("agent-A", ViolationSeverity::Major);
    logger.log(
        AuditEvent::new(AuditEventType::PolicyViolation)
            .with_agent("agent-A")
            .with_action("write_file")
            .with_target(".env")
            .with_outcome(ActionOutcome::Denied),
    )?;

    let factor = trust.get("agent-A").unwrap();
    println!("  Score before violation: {:.2}", score_before,);
    println!(
        "  Score after violation:  {:.2}  (level: {})",
        factor.score, factor.level,
    );

    // 4. Query audit log
    println!("\n--- 4. Querying audit log ---\n");

    logger.flush()?;

    let all_events = logger.query(&AuditQuery::new())?;
    println!("  Total events logged: {}", all_events.len());

    let violations = logger.query(&AuditQuery::new().of_type(AuditEventType::PolicyViolation))?;
    println!("  Policy violations:   {}", violations.len());

    let agent_a_events = logger.query(&AuditQuery::new().for_agent("agent-A"))?;
    println!("  Events for agent-A:  {}", agent_a_events.len());

    // 5. Audit statistics
    println!("\n--- 5. Audit statistics ---\n");

    let stats = logger.statistics(None)?;
    println!("  Total events:        {}", stats.total_events);
    println!("  Tool executions:     {}", stats.tool_executions);
    println!("  Policy violations:   {}", stats.policy_violations);
    println!("  Successful actions:  {}", stats.successful_actions);
    println!("  Denied actions:      {}", stats.denied_actions);

    // 6. Trust statistics
    println!("\n--- 6. Trust statistics ---\n");

    let trust_stats = trust.statistics();
    println!("  Total agents:        {}", trust_stats.total_agents);
    println!("  Total violations:    {}", trust_stats.total_violations);
    println!("  Total operations:    {}", trust_stats.total_operations);

    println!("\n=== Done ===");
    Ok(())
}
