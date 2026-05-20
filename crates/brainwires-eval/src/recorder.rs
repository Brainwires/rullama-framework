//! Tool call sequence recording and diff.
//!
//! [`ToolSequenceRecorder`] is a lightweight, thread-safe recorder that
//! captures the ordered sequence of tool calls made during an agent run.
//! Attach it to an agent's pre-execution hook and call
//! [`ToolSequenceRecorder::diff_against`] at the end of a trial to verify
//! behavioural correctness.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ── Tool call record ──────────────────────────────────────────────────────────

/// A single recorded tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallRecord {
    /// Name of the tool that was invoked.
    pub name: String,
    /// A short fingerprint of the tool's input arguments (first 16 hex chars
    /// of a FNV-style hash).  Used for lightweight argument comparison without
    /// storing the full payload.
    pub args_fingerprint: String,
    /// Wall-clock timestamp of the call in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

impl ToolCallRecord {
    fn new(name: impl Into<String>, args: &serde_json::Value) -> Self {
        let name = name.into();
        let args_fingerprint = fingerprint_json(args);
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            name,
            args_fingerprint,
            timestamp_ms,
        }
    }
}

fn fingerprint_json(v: &serde_json::Value) -> String {
    let mut h = DefaultHasher::new();
    v.to_string().hash(&mut h);
    format!("{:016x}", h.finish())
}

// ── Sequence diff ─────────────────────────────────────────────────────────────

/// Result of comparing the recorded tool sequence against an expected sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceDiff {
    /// The expected tool names (in order).
    pub expected: Vec<String>,
    /// The actual tool names recorded (in order).
    pub actual: Vec<String>,
    /// Edit distance between the two sequences (Levenshtein).
    pub edit_distance: usize,
    /// Similarity in [0, 1]: `1.0 − edit_distance / max(len_expected, len_actual)`.
    /// `1.0` means an exact match; `0.0` means maximally different.
    pub similarity: f64,
}

impl SequenceDiff {
    /// Compute the diff between `expected` and `actual` name sequences.
    pub fn compute(expected: &[&str], actual: &[String]) -> Self {
        let exp: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        let ed = levenshtein(
            expected,
            actual
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let max_len = exp.len().max(actual.len());
        let similarity = if max_len == 0 {
            1.0
        } else {
            1.0 - (ed as f64 / max_len as f64)
        };
        Self {
            expected: exp,
            actual: actual.to_vec(),
            edit_distance: ed,
            similarity,
        }
    }

    /// Returns `true` if the actual sequence exactly matches the expected one.
    pub fn is_exact_match(&self) -> bool {
        self.edit_distance == 0
    }
}

/// Compute Levenshtein edit distance between two string slices.
fn levenshtein(a: &[&str], b: &[&str]) -> usize {
    let n = a.len();
    let m = b.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate().take(n + 1) {
        row[0] = i;
    }
    for (j, val) in dp[0].iter_mut().enumerate().take(m + 1) {
        *val = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }
    dp[n][m]
}

// ── Recorder ──────────────────────────────────────────────────────────────────

/// Thread-safe recorder for tool call sequences.
///
/// Wrap in `Arc` and share across async tasks / agent hooks.
///
/// ## Example
/// ```rust,ignore
/// let recorder = ToolSequenceRecorder::new();
/// recorder.record("read_file", &json!({"path": "main.rs"}));
/// recorder.record("write_file", &json!({"path": "out.rs"}));
///
/// let diff = recorder.diff_against(&["read_file", "write_file"]);
/// assert!(diff.is_exact_match());
/// ```
#[derive(Debug, Clone, Default)]
pub struct ToolSequenceRecorder {
    inner: Arc<Mutex<Vec<ToolCallRecord>>>,
}

impl ToolSequenceRecorder {
    /// Create a new, empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a tool call.  Safe to call from multiple threads / async tasks.
    pub fn record(&self, name: impl Into<String>, args: &serde_json::Value) {
        let record = ToolCallRecord::new(name, args);
        self.inner
            .lock()
            .expect("recorder lock poisoned")
            .push(record);
    }

    /// Return a snapshot of all recorded calls in insertion order.
    pub fn calls(&self) -> Vec<ToolCallRecord> {
        self.inner.lock().expect("recorder lock poisoned").clone()
    }

    /// Return only the tool names in insertion order.
    pub fn call_names(&self) -> Vec<String> {
        self.inner
            .lock()
            .expect("recorder lock poisoned")
            .iter()
            .map(|r| r.name.clone())
            .collect()
    }

    /// Diff the recorded sequence against an expected list of tool names.
    pub fn diff_against(&self, expected: &[&str]) -> SequenceDiff {
        let actual = self.call_names();
        SequenceDiff::compute(expected, &actual)
    }

    /// Clear all recorded calls.
    pub fn reset(&self) {
        self.inner.lock().expect("recorder lock poisoned").clear();
    }

    /// Number of recorded calls.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("recorder lock poisoned").len()
    }

    /// Returns `true` if no calls have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_record_and_retrieve() {
        let recorder = ToolSequenceRecorder::new();
        recorder.record("read_file", &json!({"path": "a.rs"}));
        recorder.record("write_file", &json!({"path": "b.rs"}));

        let calls = recorder.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[1].name, "write_file");
    }

    #[test]
    fn test_call_names() {
        let recorder = ToolSequenceRecorder::new();
        recorder.record("bash", &json!({}));
        recorder.record("read_file", &json!({}));
        assert_eq!(recorder.call_names(), vec!["bash", "read_file"]);
    }

    #[test]
    fn test_diff_exact_match() {
        let recorder = ToolSequenceRecorder::new();
        recorder.record("a", &json!({}));
        recorder.record("b", &json!({}));
        recorder.record("c", &json!({}));

        let diff = recorder.diff_against(&["a", "b", "c"]);
        assert!(diff.is_exact_match());
        assert!((diff.similarity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_diff_partial_match() {
        let recorder = ToolSequenceRecorder::new();
        recorder.record("a", &json!({}));
        recorder.record("x", &json!({})); // unexpected
        recorder.record("c", &json!({}));

        let diff = recorder.diff_against(&["a", "b", "c"]);
        assert!(!diff.is_exact_match());
        assert_eq!(diff.edit_distance, 1);
        assert!(diff.similarity > 0.5);
    }

    #[test]
    fn test_diff_empty_vs_expected() {
        let recorder = ToolSequenceRecorder::new();
        let diff = recorder.diff_against(&["a", "b"]);
        assert_eq!(diff.edit_distance, 2);
        assert_eq!(diff.similarity, 0.0);
    }

    #[test]
    fn test_diff_both_empty() {
        let recorder = ToolSequenceRecorder::new();
        let diff = recorder.diff_against(&[]);
        assert!(diff.is_exact_match());
        assert!((diff.similarity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_reset_clears_calls() {
        let recorder = ToolSequenceRecorder::new();
        recorder.record("a", &json!({}));
        recorder.reset();
        assert!(recorder.is_empty());
    }

    #[test]
    fn test_args_fingerprint_differs_for_different_args() {
        let r1 = ToolCallRecord::new("tool", &json!({"a": 1}));
        let r2 = ToolCallRecord::new("tool", &json!({"a": 2}));
        assert_ne!(r1.args_fingerprint, r2.args_fingerprint);
    }

    #[test]
    fn test_args_fingerprint_same_for_same_args() {
        let r1 = ToolCallRecord::new("tool", &json!({"x": "hello"}));
        let r2 = ToolCallRecord::new("tool", &json!({"x": "hello"}));
        assert_eq!(r1.args_fingerprint, r2.args_fingerprint);
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein(&["a", "b", "c"], &["a", "b", "c"]), 0);
    }

    #[test]
    fn test_levenshtein_single_substitution() {
        assert_eq!(levenshtein(&["a", "b", "c"], &["a", "x", "c"]), 1);
    }

    #[test]
    fn test_levenshtein_insert_delete() {
        assert_eq!(levenshtein(&["a", "b"], &["a", "b", "c"]), 1);
        assert_eq!(levenshtein(&["a", "b", "c"], &["a", "b"]), 1);
    }
}
