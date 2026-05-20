//! Feedback Bridge — AuditLogger to SEAL Learning Loop
//!
//! Reads user feedback signals (thumbs-up/down + corrections) from the
//! [`AuditLogger`] and converts them into SEAL learning signals that
//! improve prompting strategies over time.
//!
//! # Architecture
//!
//! ```text
//! AuditLogger::submit_feedback()
//!     |
//!     v
//! FeedbackSignal { polarity, correction }
//!     |  (pulled on demand)
//!     v
//! FeedbackBridge::process_feedback_for_run()
//!     |
//!     v
//! LearningCoordinator::record_outcome()
//!     |
//!     v
//! GlobalMemory patterns updated
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use brainwires_seal::FeedbackBridge;
//!
//! let bridge = FeedbackBridge::new(&audit_logger, &mut learning_coordinator);
//! let stats = bridge.process_feedback_for_run("run-123")?;
//! println!("Processed {} feedback signals", stats.processed);
//! ```

use anyhow::Result;
use brainwires_permission::audit::{
    AuditEventType, AuditLogger, AuditQuery, FeedbackPolarity, FeedbackSignal,
};
use chrono::{DateTime, Utc};

use super::learning::{LearningCoordinator, PatternHint};

/// Statistics from processing a batch of feedback signals.
#[derive(Debug, Clone, Default)]
pub struct FeedbackProcessingStats {
    /// Total feedback signals processed.
    pub processed: usize,
    /// Number of positive (thumbs-up) signals.
    pub positive: usize,
    /// Number of negative (thumbs-down) signals.
    pub negative: usize,
    /// Number of corrections applied as pattern hints.
    pub corrections_applied: usize,
    /// Number of signals skipped (e.g. duplicates or already processed).
    pub skipped: usize,
}

/// Bridge between the [`AuditLogger`] feedback system and the SEAL
/// [`LearningCoordinator`].
///
/// Uses a pull model: feedback is fetched from the audit log on demand
/// rather than requiring a background listener.
pub struct FeedbackBridge<'a> {
    audit_logger: &'a AuditLogger,
    learning: &'a mut LearningCoordinator,
}

impl<'a> FeedbackBridge<'a> {
    /// Create a new feedback bridge.
    pub fn new(audit_logger: &'a AuditLogger, learning: &'a mut LearningCoordinator) -> Self {
        Self {
            audit_logger,
            learning,
        }
    }

    /// Process all feedback signals for a specific run.
    ///
    /// Maps `ThumbsUp` → `record_outcome(success=true)` and
    /// `ThumbsDown` → `record_outcome(success=false)`. When a
    /// `ThumbsDown` includes a correction string, it is added as a
    /// [`PatternHint`] to the global memory.
    pub fn process_feedback_for_run(&mut self, run_id: &str) -> Result<FeedbackProcessingStats> {
        let signals = self.audit_logger.get_feedback_for_run(run_id)?;
        let mut stats = FeedbackProcessingStats::default();

        for signal in &signals {
            self.apply_signal(signal, &mut stats);
        }

        Ok(stats)
    }

    /// Process all feedback signals submitted since the given timestamp.
    ///
    /// Queries the audit log for `UserFeedback` events and converts each
    /// into a learning signal.
    pub fn process_recent_feedback(
        &mut self,
        since: DateTime<Utc>,
    ) -> Result<FeedbackProcessingStats> {
        let query = AuditQuery::new()
            .of_type(AuditEventType::UserFeedback)
            .since(since);

        let events = self.audit_logger.query(&query)?;
        let mut stats = FeedbackProcessingStats::default();

        // Reconstruct FeedbackSignals from audit events
        for event in &events {
            let polarity_str = match event.metadata.get("polarity") {
                Some(s) => s.as_str(),
                None => {
                    stats.skipped += 1;
                    continue;
                }
            };

            let polarity = match polarity_str {
                "thumbs_up" => FeedbackPolarity::ThumbsUp,
                "thumbs_down" => FeedbackPolarity::ThumbsDown,
                _ => {
                    stats.skipped += 1;
                    continue;
                }
            };

            let signal = FeedbackSignal {
                id: event
                    .metadata
                    .get("feedback_id")
                    .cloned()
                    .unwrap_or_default(),
                run_id: event.metadata.get("run_id").cloned().unwrap_or_default(),
                polarity,
                correction: event.metadata.get("correction").cloned(),
                submitted_at: event.timestamp,
            };

            self.apply_signal(&signal, &mut stats);
        }

        Ok(stats)
    }

    /// Apply a single feedback signal to the learning coordinator.
    fn apply_signal(&mut self, signal: &FeedbackSignal, stats: &mut FeedbackProcessingStats) {
        let success = signal.polarity == FeedbackPolarity::ThumbsUp;

        // Record as a generic outcome (no specific pattern or query core)
        self.learning.record_outcome(
            None, // no specific pattern ID
            success,
            if success { 1 } else { 0 }, // treat positive as 1 result
            None,                        // no query core context
            0,                           // no execution time
        );

        if success {
            stats.positive += 1;
        } else {
            stats.negative += 1;
        }

        // If there's a correction, add it as a pattern hint
        if let Some(ref correction) = signal.correction {
            self.apply_correction(correction, &signal.run_id);
            stats.corrections_applied += 1;
        }

        stats.processed += 1;
    }

    /// Apply a user correction as a pattern hint in global memory.
    ///
    /// The correction text is stored as a hint with the run ID as context,
    /// allowing the learning system to reference it when generating prompts
    /// for similar future queries.
    fn apply_correction(&mut self, correction: &str, run_id: &str) {
        let hint = PatternHint {
            context_pattern: format!("run:{}", run_id),
            rule: correction.to_string(),
            confidence: 1.0, // user corrections are high-confidence
            source: "user_feedback".to_string(),
        };

        self.learning.global.add_pattern_hint(hint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (AuditLogger, LearningCoordinator, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("audit.jsonl");
        let logger = AuditLogger::with_path(log_path).unwrap();
        let learning = LearningCoordinator::new("test-conv".to_string());
        (logger, learning, temp_dir)
    }

    #[test]
    fn test_process_thumbs_up() {
        let (logger, mut learning, _tmp) = setup();

        logger
            .submit_feedback("run-1", FeedbackPolarity::ThumbsUp, None)
            .unwrap();

        let mut bridge = FeedbackBridge::new(&logger, &mut learning);
        let stats = bridge.process_feedback_for_run("run-1").unwrap();

        assert_eq!(stats.processed, 1);
        assert_eq!(stats.positive, 1);
        assert_eq!(stats.negative, 0);
        assert_eq!(stats.corrections_applied, 0);
    }

    #[test]
    fn test_process_thumbs_down_with_correction() {
        let (logger, mut learning, _tmp) = setup();

        logger
            .submit_feedback(
                "run-2",
                FeedbackPolarity::ThumbsDown,
                Some("Use async instead of sync"),
            )
            .unwrap();

        let mut bridge = FeedbackBridge::new(&logger, &mut learning);
        let stats = bridge.process_feedback_for_run("run-2").unwrap();

        assert_eq!(stats.processed, 1);
        assert_eq!(stats.negative, 1);
        assert_eq!(stats.corrections_applied, 1);

        // Verify the correction was stored as a pattern hint
        let hints = bridge.learning.global.get_pattern_hints();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].rule, "Use async instead of sync");
        assert_eq!(hints[0].source, "user_feedback");
        assert!((hints[0].confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_process_multiple_feedback_for_run() {
        let (logger, mut learning, _tmp) = setup();

        logger
            .submit_feedback("run-3", FeedbackPolarity::ThumbsUp, None)
            .unwrap();
        logger
            .submit_feedback(
                "run-3",
                FeedbackPolarity::ThumbsDown,
                Some("Wrong approach"),
            )
            .unwrap();

        let mut bridge = FeedbackBridge::new(&logger, &mut learning);
        let stats = bridge.process_feedback_for_run("run-3").unwrap();

        assert_eq!(stats.processed, 2);
        assert_eq!(stats.positive, 1);
        assert_eq!(stats.negative, 1);
        assert_eq!(stats.corrections_applied, 1);
    }

    #[test]
    fn test_process_feedback_no_results() {
        let (logger, mut learning, _tmp) = setup();

        let mut bridge = FeedbackBridge::new(&logger, &mut learning);
        let stats = bridge.process_feedback_for_run("nonexistent").unwrap();

        assert_eq!(stats.processed, 0);
    }

    #[test]
    fn test_process_recent_feedback() {
        let (logger, mut learning, _tmp) = setup();
        let before = Utc::now() - chrono::Duration::seconds(1);

        logger
            .submit_feedback("run-a", FeedbackPolarity::ThumbsUp, None)
            .unwrap();
        logger
            .submit_feedback("run-b", FeedbackPolarity::ThumbsDown, Some("Fix this"))
            .unwrap();

        let mut bridge = FeedbackBridge::new(&logger, &mut learning);
        let stats = bridge.process_recent_feedback(before).unwrap();

        assert_eq!(stats.processed, 2);
        assert_eq!(stats.positive, 1);
        assert_eq!(stats.negative, 1);
        assert_eq!(stats.corrections_applied, 1);
    }

    #[test]
    fn test_feedback_isolated_between_runs() {
        let (logger, mut learning, _tmp) = setup();

        logger
            .submit_feedback("run-x", FeedbackPolarity::ThumbsUp, None)
            .unwrap();
        logger
            .submit_feedback("run-y", FeedbackPolarity::ThumbsDown, None)
            .unwrap();

        let mut bridge = FeedbackBridge::new(&logger, &mut learning);

        let stats_x = bridge.process_feedback_for_run("run-x").unwrap();
        assert_eq!(stats_x.processed, 1);
        assert_eq!(stats_x.positive, 1);

        let stats_y = bridge.process_feedback_for_run("run-y").unwrap();
        assert_eq!(stats_y.processed, 1);
        assert_eq!(stats_y.negative, 1);
    }
}
