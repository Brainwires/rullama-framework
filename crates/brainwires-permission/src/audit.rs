//! Audit System - Comprehensive logging for security and compliance
//!
//! Provides audit logging for all permission-related events including:
//! - Tool executions (allowed, denied, approved)
//! - File access (read, write, delete)
//! - Network requests
//! - Policy evaluations
//! - Trust level changes
//! - Human interventions
//!
//! # Example
//!
//! ```rust,ignore
//! use brainwires::permissions::audit::{AuditLogger, AuditEvent, AuditEventType};
//!
//! let logger = AuditLogger::new()?;
//!
//! logger.log(AuditEvent::new(AuditEventType::ToolExecution)
//!     .with_agent("agent-123")
//!     .with_action("write_file")
//!     .with_target("/src/main.rs")
//!     .with_outcome(ActionOutcome::Success));
//! ```

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::policy::PolicyDecision;

/// Default maximum number of audit events to buffer before flushing to disk.
const DEFAULT_AUDIT_BUFFER_SIZE: usize = 100;

/// Type of audit event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventType {
    /// Tool was executed
    ToolExecution,
    /// File was accessed
    FileAccess,
    /// Network request was made
    NetworkRequest,
    /// Agent was spawned
    AgentSpawn,
    /// Policy was violated
    PolicyViolation,
    /// Trust level changed
    TrustChange,
    /// Human provided approval or override
    HumanIntervention,
    /// Session started
    SessionStart,
    /// Session ended
    SessionEnd,
    /// Configuration changed
    ConfigChange,
    /// User provided explicit feedback (thumbs up/down + optional correction) for a completed run
    UserFeedback,
}

/// Outcome of an action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionOutcome {
    /// Action completed successfully
    Success,
    /// Action failed
    Failure,
    /// Action partially completed
    Partial,
    /// Action timed out
    Timeout,
    /// Action was cancelled
    Cancelled,
    /// Action was denied by policy
    Denied,
    /// Action required approval
    PendingApproval,
    /// Action was approved by human
    Approved,
    /// Action was rejected by human
    Rejected,
}

/// Human approval record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanApproval {
    /// Who approved (if known)
    pub approver: Option<String>,
    /// When approval was given
    pub timestamp: DateTime<Utc>,
    /// What was approved
    pub action: String,
    /// Any justification provided
    pub justification: Option<String>,
    /// Whether it was approved or rejected
    pub approved: bool,
}

/// A single audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID
    pub id: String,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Type of event
    pub event_type: AuditEventType,
    /// Agent that triggered the event
    pub agent_id: Option<String>,
    /// Action that was performed
    pub action: String,
    /// Target of the action (file path, domain, etc.)
    pub target: Option<String>,
    /// Policy that was evaluated
    pub policy_id: Option<String>,
    /// Decision made by policy engine
    pub decision: Option<String>,
    /// Trust level at time of event
    pub trust_level: Option<u8>,
    /// Outcome of the action
    pub outcome: ActionOutcome,
    /// Duration in milliseconds (if applicable)
    pub duration_ms: Option<u64>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl AuditEvent {
    /// Create a new audit event
    pub fn new(event_type: AuditEventType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type,
            agent_id: None,
            action: String::new(),
            target: None,
            policy_id: None,
            decision: None,
            trust_level: None,
            outcome: ActionOutcome::Success,
            duration_ms: None,
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the agent ID
    pub fn with_agent(mut self, agent_id: &str) -> Self {
        self.agent_id = Some(agent_id.to_string());
        self
    }

    /// Set the action
    pub fn with_action(mut self, action: &str) -> Self {
        self.action = action.to_string();
        self
    }

    /// Set the target
    pub fn with_target(mut self, target: &str) -> Self {
        self.target = Some(target.to_string());
        self
    }

    /// Set the policy ID
    pub fn with_policy(mut self, policy_id: &str) -> Self {
        self.policy_id = Some(policy_id.to_string());
        self
    }

    /// Set the decision
    pub fn with_decision(mut self, decision: &PolicyDecision) -> Self {
        self.policy_id = decision.matched_policy.clone();
        self.decision = decision.reason.clone();
        self
    }

    /// Set the trust level
    pub fn with_trust_level(mut self, level: u8) -> Self {
        self.trust_level = Some(level);
        self
    }

    /// Set the outcome
    pub fn with_outcome(mut self, outcome: ActionOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Set the duration
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Set an error
    pub fn with_error(mut self, error: &str) -> Self {
        self.error = Some(error.to_string());
        self.outcome = ActionOutcome::Failure;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

impl brainwires_telemetry::anomaly::ObservedEvent for AuditEvent {
    fn timestamp_secs(&self) -> i64 {
        self.timestamp.timestamp()
    }
    fn agent_id(&self) -> Option<&str> {
        self.agent_id.as_deref()
    }
    fn category(&self) -> brainwires_telemetry::anomaly::EventCategory {
        use brainwires_telemetry::anomaly::EventCategory;
        match self.event_type {
            AuditEventType::PolicyViolation => EventCategory::PolicyViolation,
            AuditEventType::ToolExecution => EventCategory::ToolExecution,
            AuditEventType::TrustChange => EventCategory::TrustChange,
            _ => EventCategory::Other,
        }
    }
    fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }
}

/// Query parameters for searching audit logs
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Filter by agent ID
    pub agent_id: Option<String>,
    /// Filter by event type
    pub event_type: Option<AuditEventType>,
    /// Filter by action
    pub action: Option<String>,
    /// Filter by outcome
    pub outcome: Option<ActionOutcome>,
    /// Filter events after this time
    pub since: Option<DateTime<Utc>>,
    /// Filter events before this time
    pub until: Option<DateTime<Utc>>,
    /// Maximum number of results
    pub limit: Option<usize>,
}

impl AuditQuery {
    /// Create a new query
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by agent
    pub fn for_agent(mut self, agent_id: &str) -> Self {
        self.agent_id = Some(agent_id.to_string());
        self
    }

    /// Filter by event type
    pub fn of_type(mut self, event_type: AuditEventType) -> Self {
        self.event_type = Some(event_type);
        self
    }

    /// Filter by action
    pub fn with_action(mut self, action: &str) -> Self {
        self.action = Some(action.to_string());
        self
    }

    /// Filter by outcome
    pub fn with_outcome(mut self, outcome: ActionOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    /// Filter by time range
    pub fn since(mut self, since: DateTime<Utc>) -> Self {
        self.since = Some(since);
        self
    }

    /// Filter by time range
    pub fn until(mut self, until: DateTime<Utc>) -> Self {
        self.until = Some(until);
        self
    }

    /// Limit results
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check if an event matches this query
    pub fn matches(&self, event: &AuditEvent) -> bool {
        if let Some(ref agent_id) = self.agent_id
            && event.agent_id.as_ref() != Some(agent_id)
        {
            return false;
        }
        if let Some(event_type) = self.event_type
            && event.event_type != event_type
        {
            return false;
        }
        if let Some(ref action) = self.action
            && !event.action.contains(action)
        {
            return false;
        }
        if let Some(outcome) = self.outcome
            && event.outcome != outcome
        {
            return false;
        }
        if let Some(since) = self.since
            && event.timestamp < since
        {
            return false;
        }
        if let Some(until) = self.until
            && event.timestamp > until
        {
            return false;
        }
        true
    }
}

/// Audit logger for recording permission events
#[derive(Debug)]
pub struct AuditLogger {
    /// Path to the audit log file
    log_path: PathBuf,
    /// In-memory buffer for recent events
    buffer: Arc<Mutex<Vec<AuditEvent>>>,
    /// Maximum buffer size before flushing
    max_buffer_size: usize,
    /// Whether logging is enabled
    enabled: bool,
    /// Optional anomaly detector; when present, every logged event is observed.
    anomaly_detector: Option<brainwires_telemetry::anomaly::AnomalyDetector>,
}

impl AuditLogger {
    /// Create a new audit logger with the default path (~/.brainwires/audit/)
    #[cfg(feature = "native")]
    pub fn new() -> Result<Self> {
        let log_dir = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to get home directory"))?
            .join(".brainwires")
            .join("audit");
        std::fs::create_dir_all(&log_dir)?;

        let log_path = log_dir.join("audit.jsonl");

        Ok(Self {
            log_path,
            buffer: Arc::new(Mutex::new(Vec::new())),
            max_buffer_size: DEFAULT_AUDIT_BUFFER_SIZE,
            enabled: true,
            anomaly_detector: None,
        })
    }

    /// Create a logger with a custom path
    #[cfg(feature = "native")]
    pub fn with_path(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        Ok(Self {
            log_path: path,
            buffer: Arc::new(Mutex::new(Vec::new())),
            max_buffer_size: DEFAULT_AUDIT_BUFFER_SIZE,
            enabled: true,
            anomaly_detector: None,
        })
    }

    /// Attach an anomaly detector (builder pattern).
    ///
    /// Every event passed to [`Self::log`] will be fed to the detector.
    /// Call [`Self::drain_anomalies`] to retrieve any flagged events.
    pub fn with_anomaly_detection(
        mut self,
        config: brainwires_telemetry::anomaly::AnomalyConfig,
    ) -> Self {
        self.anomaly_detector = Some(brainwires_telemetry::anomaly::AnomalyDetector::new(config));
        self
    }

    /// Drain all accumulated anomaly events.
    ///
    /// Returns `None` if no anomaly detector is attached.
    pub fn drain_anomalies(&self) -> Option<Vec<brainwires_telemetry::anomaly::AnomalyEvent>> {
        self.anomaly_detector.as_ref().map(|d| d.drain_anomalies())
    }

    /// Return the number of pending anomaly events without draining.
    ///
    /// Returns `0` if no anomaly detector is attached.
    pub fn pending_anomaly_count(&self) -> usize {
        self.anomaly_detector
            .as_ref()
            .map(|d| d.pending_count())
            .unwrap_or(0)
    }

    /// Enable or disable logging
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Log an audit event
    pub fn log(&self, event: AuditEvent) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // For important events, write immediately to disk (skip buffer)
        let is_important = matches!(
            event.event_type,
            AuditEventType::PolicyViolation
                | AuditEventType::TrustChange
                | AuditEventType::HumanIntervention
                | AuditEventType::UserFeedback
        );

        // Feed to anomaly detector before potentially moving the event into the buffer
        if let Some(ref detector) = self.anomaly_detector {
            detector.observe(&event);
        }

        if is_important {
            // Write directly to disk for important events
            self.write_event(&event)?;
        } else {
            // Add to buffer for normal events
            let mut buffer = self.buffer.lock().expect("audit log buffer lock poisoned");
            buffer.push(event);

            // Flush if buffer is full
            if buffer.len() >= self.max_buffer_size {
                self.flush_buffer_internal(&mut buffer)?;
            }
        }

        Ok(())
    }

    /// Log a tool execution
    pub fn log_tool_execution(
        &self,
        agent_id: Option<&str>,
        tool_name: &str,
        target: Option<&str>,
        outcome: ActionOutcome,
        duration_ms: Option<u64>,
    ) -> Result<()> {
        let mut event = AuditEvent::new(AuditEventType::ToolExecution)
            .with_action(tool_name)
            .with_outcome(outcome);

        if let Some(agent) = agent_id {
            event = event.with_agent(agent);
        }
        if let Some(t) = target {
            event = event.with_target(t);
        }
        if let Some(d) = duration_ms {
            event = event.with_duration(d);
        }

        self.log(event)
    }

    /// Log a denied action
    pub fn log_denied(
        &self,
        agent_id: Option<&str>,
        action: &str,
        target: Option<&str>,
        reason: &str,
    ) -> Result<()> {
        let mut event = AuditEvent::new(AuditEventType::PolicyViolation)
            .with_action(action)
            .with_outcome(ActionOutcome::Denied)
            .with_metadata("reason", reason);

        if let Some(agent) = agent_id {
            event = event.with_agent(agent);
        }
        if let Some(t) = target {
            event = event.with_target(t);
        }

        self.log(event)
    }

    /// Log a human approval
    pub fn log_approval(
        &self,
        agent_id: Option<&str>,
        action: &str,
        approved: bool,
        justification: Option<&str>,
    ) -> Result<()> {
        let mut event = AuditEvent::new(AuditEventType::HumanIntervention)
            .with_action(action)
            .with_outcome(if approved {
                ActionOutcome::Approved
            } else {
                ActionOutcome::Rejected
            });

        if let Some(agent) = agent_id {
            event = event.with_agent(agent);
        }
        if let Some(j) = justification {
            event = event.with_metadata("justification", j);
        }

        self.log(event)
    }

    /// Log a trust level change
    pub fn log_trust_change(
        &self,
        agent_id: &str,
        old_level: u8,
        new_level: u8,
        reason: &str,
    ) -> Result<()> {
        let event = AuditEvent::new(AuditEventType::TrustChange)
            .with_agent(agent_id)
            .with_action("trust_change")
            .with_trust_level(new_level)
            .with_metadata("old_level", &old_level.to_string())
            .with_metadata("reason", reason);

        self.log(event)
    }

    /// Submit user feedback for a specific agent run.
    ///
    /// The feedback is persisted as an `AuditEvent` of type `UserFeedback` so it
    /// can be queried later.  The returned [`FeedbackSignal`] contains the
    /// generated feedback ID which uniquely identifies this submission.
    pub fn submit_feedback(
        &self,
        run_id: &str,
        polarity: FeedbackPolarity,
        correction: Option<&str>,
    ) -> Result<FeedbackSignal> {
        let signal = FeedbackSignal {
            id: uuid::Uuid::new_v4().to_string(),
            run_id: run_id.to_string(),
            polarity,
            correction: correction.map(str::to_string),
            submitted_at: Utc::now(),
        };

        let polarity_str = match polarity {
            FeedbackPolarity::ThumbsUp => "thumbs_up",
            FeedbackPolarity::ThumbsDown => "thumbs_down",
        };

        let mut event = AuditEvent::new(AuditEventType::UserFeedback)
            .with_action("user_feedback")
            .with_metadata("run_id", run_id)
            .with_metadata("polarity", polarity_str)
            .with_metadata("feedback_id", &signal.id)
            .with_outcome(ActionOutcome::Success);

        if let Some(c) = correction {
            event = event.with_metadata("correction", c);
        }

        self.log(event)?;
        Ok(signal)
    }

    /// Query all feedback signals associated with a specific run ID.
    ///
    /// Reconstructs [`FeedbackSignal`] values from stored `UserFeedback` audit
    /// events.  Returns an empty `Vec` when no feedback has been submitted for
    /// the run.
    pub fn get_feedback_for_run(&self, run_id: &str) -> Result<Vec<FeedbackSignal>> {
        let query = AuditQuery::new().of_type(AuditEventType::UserFeedback);
        let events = self.query(&query)?;

        let signals = events
            .into_iter()
            .filter(|e| {
                e.metadata
                    .get("run_id")
                    .map(|r| r == run_id)
                    .unwrap_or(false)
            })
            .filter_map(|e| {
                let feedback_id = e.metadata.get("feedback_id")?.clone();
                let polarity_str = e.metadata.get("polarity")?;
                let polarity = match polarity_str.as_str() {
                    "thumbs_up" => FeedbackPolarity::ThumbsUp,
                    "thumbs_down" => FeedbackPolarity::ThumbsDown,
                    _ => return None,
                };
                Some(FeedbackSignal {
                    id: feedback_id,
                    run_id: e.metadata.get("run_id")?.clone(),
                    polarity,
                    correction: e.metadata.get("correction").cloned(),
                    submitted_at: e.timestamp,
                })
            })
            .collect();

        Ok(signals)
    }

    /// Flush the buffer to disk
    pub fn flush(&self) -> Result<()> {
        let mut buffer = self.buffer.lock().expect("audit log buffer lock poisoned");
        self.flush_buffer_internal(&mut buffer)
    }

    fn flush_buffer_internal(&self, buffer: &mut Vec<AuditEvent>) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;

        for event in buffer.drain(..) {
            let json = serde_json::to_string(&event)?;
            writeln!(file, "{}", json)?;
        }

        Ok(())
    }

    fn write_event(&self, event: &AuditEvent) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;

        let json = serde_json::to_string(event)?;
        writeln!(file, "{}", json)?;

        Ok(())
    }

    /// Query audit events
    pub fn query(&self, query: &AuditQuery) -> Result<Vec<AuditEvent>> {
        let mut results = Vec::new();

        // First check buffer
        {
            let buffer = self.buffer.lock().expect("audit log buffer lock poisoned");
            for event in buffer.iter() {
                if query.matches(event) {
                    results.push(event.clone());
                }
            }
        }

        // Then read from file
        if self.log_path.exists() {
            let file = File::open(&self.log_path)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                if line.is_empty() {
                    continue;
                }
                if let Ok(event) = serde_json::from_str::<AuditEvent>(&line)
                    && query.matches(&event)
                {
                    results.push(event);
                }
            }
        }

        // Sort by timestamp (newest first)
        results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Get recent events
    pub fn recent(&self, count: usize) -> Result<Vec<AuditEvent>> {
        self.query(&AuditQuery::new().limit(count))
    }

    /// Count events matching a query
    pub fn count(&self, query: &AuditQuery) -> Result<usize> {
        // For a simple count, we can just query and count
        // A more efficient implementation would scan without deserializing
        Ok(self.query(query)?.len())
    }

    /// Export audit log to JSON
    pub fn export_json(&self, query: &AuditQuery) -> Result<String> {
        let events = self.query(query)?;
        Ok(serde_json::to_string_pretty(&events)?)
    }

    /// Export audit log to CSV
    pub fn export_csv(&self, query: &AuditQuery) -> Result<String> {
        let events = self.query(query)?;
        let mut csv =
            String::from("timestamp,event_type,agent_id,action,target,outcome,policy_id\n");

        for event in events {
            csv.push_str(&format!(
                "{},{:?},{},{},{},{:?},{}\n",
                event.timestamp.to_rfc3339(),
                event.event_type,
                event.agent_id.as_deref().unwrap_or(""),
                event.action,
                event.target.as_deref().unwrap_or(""),
                event.outcome,
                event.policy_id.as_deref().unwrap_or(""),
            ));
        }

        Ok(csv)
    }

    /// Get audit statistics
    pub fn statistics(&self, since: Option<DateTime<Utc>>) -> Result<AuditStatistics> {
        let query = if let Some(since) = since {
            AuditQuery::new().since(since)
        } else {
            AuditQuery::new()
        };

        let events = self.query(&query)?;

        let mut stats = AuditStatistics {
            total_events: events.len(),
            ..Default::default()
        };

        for event in &events {
            match event.event_type {
                AuditEventType::ToolExecution => stats.tool_executions += 1,
                AuditEventType::PolicyViolation => stats.policy_violations += 1,
                AuditEventType::HumanIntervention => stats.human_interventions += 1,
                _ => {}
            }

            match event.outcome {
                ActionOutcome::Success => stats.successful_actions += 1,
                ActionOutcome::Denied => stats.denied_actions += 1,
                ActionOutcome::Failure => stats.failed_actions += 1,
                _ => {}
            }
        }

        Ok(stats)
    }
}

impl Drop for AuditLogger {
    fn drop(&mut self) {
        // Flush any remaining events
        let _ = self.flush();
    }
}

/// Audit statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStatistics {
    /// Total number of audit events.
    pub total_events: usize,
    /// Number of tool executions.
    pub tool_executions: usize,
    /// Number of policy violations.
    pub policy_violations: usize,
    /// Number of human interventions.
    pub human_interventions: usize,
    /// Number of successful actions.
    pub successful_actions: usize,
    /// Number of denied actions.
    pub denied_actions: usize,
    /// Number of failed actions.
    pub failed_actions: usize,
}

// ── User feedback ─────────────────────────────────────────────────────────────

/// Polarity of a user feedback signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackPolarity {
    /// The user approved of the agent's output.
    ThumbsUp,
    /// The user disapproved of the agent's output.
    ThumbsDown,
}

/// A user feedback signal associated with a single agent run.
///
/// Created by [`AuditLogger::submit_feedback`] and persisted as a
/// `UserFeedback` audit event so it can be correlated with run telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSignal {
    /// Unique identifier for this feedback submission.
    pub id: String,
    /// The run UUID this feedback is associated with.
    pub run_id: String,
    /// Whether the user approved or disapproved the output.
    pub polarity: FeedbackPolarity,
    /// Optional free-text correction or comment from the user.
    pub correction: Option<String>,
    /// When the feedback was submitted.
    pub submitted_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_logger() -> (AuditLogger, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("audit.jsonl");
        let logger = AuditLogger::with_path(log_path).unwrap();
        (logger, temp_dir)
    }

    #[test]
    fn test_log_event() {
        let (logger, _temp) = create_test_logger();

        let event = AuditEvent::new(AuditEventType::ToolExecution)
            .with_agent("agent-123")
            .with_action("write_file")
            .with_target("/src/main.rs")
            .with_outcome(ActionOutcome::Success);

        logger.log(event).unwrap();
        logger.flush().unwrap();

        let events = logger.recent(10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, "write_file");
    }

    #[test]
    fn test_query_events() {
        let (logger, _temp) = create_test_logger();

        // Log some events
        logger
            .log(
                AuditEvent::new(AuditEventType::ToolExecution)
                    .with_agent("agent-1")
                    .with_action("read_file")
                    .with_outcome(ActionOutcome::Success),
            )
            .unwrap();

        logger
            .log(
                AuditEvent::new(AuditEventType::ToolExecution)
                    .with_agent("agent-2")
                    .with_action("write_file")
                    .with_outcome(ActionOutcome::Denied),
            )
            .unwrap();

        logger
            .log(
                AuditEvent::new(AuditEventType::PolicyViolation)
                    .with_agent("agent-1")
                    .with_action("delete_file")
                    .with_outcome(ActionOutcome::Denied),
            )
            .unwrap();

        logger.flush().unwrap();

        // Query by agent
        let query = AuditQuery::new().for_agent("agent-1");
        let events = logger.query(&query).unwrap();
        assert_eq!(events.len(), 2);

        // Query by outcome
        let query = AuditQuery::new().with_outcome(ActionOutcome::Denied);
        let events = logger.query(&query).unwrap();
        assert_eq!(events.len(), 2);

        // Query by type
        let query = AuditQuery::new().of_type(AuditEventType::PolicyViolation);
        let events = logger.query(&query).unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_log_denied() {
        let (logger, _temp) = create_test_logger();

        logger
            .log_denied(
                Some("agent-123"),
                "write_file",
                Some("/.env"),
                "Protected file",
            )
            .unwrap();

        logger.flush().unwrap();

        let events = logger.recent(10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, AuditEventType::PolicyViolation);
        assert_eq!(events[0].outcome, ActionOutcome::Denied);
    }

    #[test]
    fn test_statistics() {
        let (logger, _temp) = create_test_logger();

        logger
            .log(
                AuditEvent::new(AuditEventType::ToolExecution).with_outcome(ActionOutcome::Success),
            )
            .unwrap();
        logger
            .log(
                AuditEvent::new(AuditEventType::ToolExecution).with_outcome(ActionOutcome::Success),
            )
            .unwrap();
        logger
            .log(
                AuditEvent::new(AuditEventType::PolicyViolation)
                    .with_outcome(ActionOutcome::Denied),
            )
            .unwrap();

        logger.flush().unwrap();

        let stats = logger.statistics(None).unwrap();
        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.tool_executions, 2);
        assert_eq!(stats.policy_violations, 1);
        assert_eq!(stats.successful_actions, 2);
        assert_eq!(stats.denied_actions, 1);
    }

    #[test]
    fn test_export_csv() {
        let (logger, _temp) = create_test_logger();

        logger
            .log(
                AuditEvent::new(AuditEventType::ToolExecution)
                    .with_agent("agent-1")
                    .with_action("read_file")
                    .with_outcome(ActionOutcome::Success),
            )
            .unwrap();

        logger.flush().unwrap();

        let csv = logger.export_csv(&AuditQuery::new()).unwrap();
        assert!(csv.contains("timestamp,event_type,agent_id,action"));
        assert!(csv.contains("read_file"));
        assert!(csv.contains("agent-1"));
    }

    // ── Phase 8: User feedback tests ──────────────────────────────────────────

    #[test]
    fn test_submit_feedback_thumbs_up() {
        let (logger, _temp) = create_test_logger();
        let signal = logger
            .submit_feedback("run-001", FeedbackPolarity::ThumbsUp, None)
            .unwrap();
        assert_eq!(signal.run_id, "run-001");
        assert_eq!(signal.polarity, FeedbackPolarity::ThumbsUp);
        assert!(signal.correction.is_none());

        // Feedback events are written immediately (is_important), so no flush needed
        let feedback = logger.get_feedback_for_run("run-001").unwrap();
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0].polarity, FeedbackPolarity::ThumbsUp);
    }

    #[test]
    fn test_submit_feedback_with_correction() {
        let (logger, _temp) = create_test_logger();
        let signal = logger
            .submit_feedback(
                "run-002",
                FeedbackPolarity::ThumbsDown,
                Some("Wrong answer"),
            )
            .unwrap();
        assert_eq!(signal.correction.as_deref(), Some("Wrong answer"));

        let feedback = logger.get_feedback_for_run("run-002").unwrap();
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0].correction.as_deref(), Some("Wrong answer"));
    }

    #[test]
    fn test_feedback_isolated_per_run() {
        let (logger, _temp) = create_test_logger();
        logger
            .submit_feedback("run-A", FeedbackPolarity::ThumbsUp, None)
            .unwrap();
        logger
            .submit_feedback("run-B", FeedbackPolarity::ThumbsDown, None)
            .unwrap();

        let a = logger.get_feedback_for_run("run-A").unwrap();
        let b = logger.get_feedback_for_run("run-B").unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(a[0].polarity, FeedbackPolarity::ThumbsUp);
        assert_eq!(b[0].polarity, FeedbackPolarity::ThumbsDown);
    }

    #[test]
    fn test_feedback_no_results_for_unknown_run() {
        let (logger, _temp) = create_test_logger();
        let feedback = logger.get_feedback_for_run("nonexistent-run").unwrap();
        assert!(feedback.is_empty());
    }

    #[test]
    fn test_feedback_event_type_stored_correctly() {
        let (logger, _temp) = create_test_logger();
        logger
            .submit_feedback("run-X", FeedbackPolarity::ThumbsUp, None)
            .unwrap();

        let events = logger
            .query(&AuditQuery::new().of_type(AuditEventType::UserFeedback))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, AuditEventType::UserFeedback);
        assert_eq!(events[0].metadata.get("run_id").unwrap(), "run-X");
    }

    // ── Phase 7: Anomaly detection integration tests ──────────────────────────

    #[test]
    fn test_logger_with_anomaly_detection_violation() {
        use brainwires_telemetry::anomaly::AnomalyConfig;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("audit.jsonl");
        let logger = AuditLogger::with_path(log_path)
            .unwrap()
            .with_anomaly_detection(AnomalyConfig {
                violation_threshold: 2,
                ..Default::default()
            });

        logger
            .log_denied(Some("agent-x"), "write_file", None, "test")
            .unwrap();
        logger
            .log_denied(Some("agent-x"), "write_file", None, "test")
            .unwrap();

        assert_eq!(logger.pending_anomaly_count(), 1);
        let anomalies = logger.drain_anomalies().unwrap();
        assert_eq!(anomalies.len(), 1);
        assert_eq!(logger.pending_anomaly_count(), 0);
    }

    #[test]
    fn test_logger_without_anomaly_detection_returns_none() {
        let (logger, _temp) = create_test_logger();
        // No anomaly detector attached
        assert!(logger.drain_anomalies().is_none());
        assert_eq!(logger.pending_anomaly_count(), 0);
    }
}
