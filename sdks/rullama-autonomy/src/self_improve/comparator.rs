//! Dual-path execution comparator.
//!
//! Compares results from direct and bridge (MCP proxy) execution paths to
//! detect bridge-specific failures and measure iteration efficiency.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Result from a single execution path (direct or bridge).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathResult {
    /// Whether the execution path succeeded.
    pub success: bool,
    /// Number of iterations used.
    pub iterations: u32,
    /// Git diff output.
    pub diff: String,
    /// Number of diff lines.
    pub diff_lines: u32,
    /// Wall-clock duration of the execution.
    pub duration: Duration,
    /// Error message, if the path failed.
    pub error: Option<String>,
}

impl PathResult {
    /// Create a failed path result with the given error and duration.
    pub fn failure(error: String, duration: Duration) -> Self {
        Self {
            success: false,
            iterations: 0,
            diff: String::new(),
            diff_lines: 0,
            duration,
            error: Some(error),
        }
    }
}

/// Comparison between direct and bridge execution paths.
pub use crate::metrics::ComparisonResult;

/// Compares two execution path results (direct vs bridge) to detect divergences.
pub struct Comparator;

impl Comparator {
    /// Compare two execution path results and produce a comparison report.
    pub fn compare(direct: &PathResult, bridge: &PathResult) -> ComparisonResult {
        let both_succeeded = direct.success && bridge.success;
        let both_failed = !direct.success && !bridge.success;

        let diffs_match = if both_succeeded {
            let d1: String = direct.diff.split_whitespace().collect();
            let d2: String = bridge.diff.split_whitespace().collect();
            d1 == d2
        } else {
            false
        };

        let iteration_delta = bridge.iterations as i32 - direct.iterations as i32;

        let mut bridge_specific_errors = Vec::new();
        if !bridge.success
            && direct.success
            && let Some(ref err) = bridge.error
        {
            bridge_specific_errors.push(format!("Bridge failed while direct succeeded: {err}"));
        }

        ComparisonResult {
            both_succeeded,
            both_failed,
            diffs_match,
            iteration_delta,
            bridge_specific_errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_both_succeed() {
        let direct = PathResult {
            success: true,
            iterations: 5,
            diff: "some diff".to_string(),
            diff_lines: 10,
            duration: Duration::from_secs(30),
            error: None,
        };
        let bridge = PathResult {
            success: true,
            iterations: 7,
            diff: "some diff".to_string(),
            diff_lines: 10,
            duration: Duration::from_secs(45),
            error: None,
        };

        let result = Comparator::compare(&direct, &bridge);
        assert!(result.both_succeeded);
        assert!(!result.both_failed);
        assert!(result.diffs_match);
        assert_eq!(result.iteration_delta, 2);
    }

    #[test]
    fn test_compare_bridge_fails() {
        let direct = PathResult {
            success: true,
            iterations: 5,
            diff: "diff".to_string(),
            diff_lines: 5,
            duration: Duration::from_secs(30),
            error: None,
        };
        let bridge = PathResult::failure("timeout".to_string(), Duration::from_secs(60));

        let result = Comparator::compare(&direct, &bridge);
        assert!(!result.both_succeeded);
        assert!(!result.both_failed);
        assert!(!result.diffs_match);
        assert_eq!(result.bridge_specific_errors.len(), 1);
    }
}
