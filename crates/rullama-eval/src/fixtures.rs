//! YAML-backed golden-prompt fixtures for the evaluation framework.
//!
//! Fixtures describe a scenario — input messages plus expected behaviour —
//! as a data file so non-Rust contributors can add tests without touching
//! the Rust crate. Each fixture is loaded as a [`FixtureCase`], which
//! implements [`EvaluationCase`] and can be fed into an
//! [`EvaluationSuite`](super::suite::EvaluationSuite).
//!
//! Running a fixture requires a [`FixtureRunner`] — the bridge between the
//! fixture description and whatever system under test you want to exercise
//! (a real agent, a cached trace, a mock). The fixture's
//! [`ExpectedBehavior`] is evaluated against the runner's
//! [`RunOutcome`] to produce a pass / fail [`TrialResult`].
//!
//! ## Fixture YAML
//!
//! ```yaml
//! name: refactor_small_func
//! category: coding
//! model: claude-opus-4-7
//! messages:
//!   - role: user
//!     content: "Rename `foo` to `bar` in lib.rs"
//! expected:
//!   tool_sequence: [read_file, edit_file]
//!   assertions:
//!     - contains: "fn bar"
//!     - tool_called: edit_file
//!     - finish_reason: end_turn
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::case::EvaluationCase;
use super::trial::TrialResult;

/// A loaded golden-prompt fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fixture {
    /// Short identifier used in reports (must be unique within a suite).
    pub name: String,
    /// Category label for grouping — e.g. `"adversarial"`, `"coding"`.
    #[serde(default = "default_category")]
    pub category: String,
    /// Optional model hint for the runner. Runners may ignore this.
    #[serde(default)]
    pub model: Option<String>,
    /// Input conversation to replay.
    pub messages: Vec<FixtureMessage>,
    /// Expected behaviour — whatever the runner produced must satisfy these.
    pub expected: ExpectedBehavior,
}

fn default_category() -> String {
    "fixture".to_string()
}

/// A single message in a fixture's input conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureMessage {
    /// Message role — `"user"`, `"system"`, or `"assistant"`.
    pub role: String,
    /// Plain text content.
    pub content: String,
}

/// Constraints a fixture imposes on the runner's output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExpectedBehavior {
    /// Exact ordered tool sequence the runner must produce. Empty means "any".
    #[serde(default)]
    pub tool_sequence: Vec<String>,
    /// Individual assertions that must all hold.
    #[serde(default)]
    pub assertions: Vec<Assertion>,
}

/// A single constraint on a fixture outcome.
///
/// YAML authors set exactly one field per list entry; all others default to
/// `None`. Multiple fields on the same entry all apply (AND'd together).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Assertion {
    /// The runner's `output_text` must contain this substring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contains: Option<String>,
    /// The runner's `output_text` must match this regex anywhere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    /// The runner must have called this tool at least once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_called: Option<String>,
    /// The runner's `finish_reason` must equal this value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Result of running a fixture through a [`FixtureRunner`].
#[derive(Debug, Clone, Default)]
pub struct RunOutcome {
    /// The final assistant text the runner produced.
    pub output_text: String,
    /// Tool calls the runner made, in the order they were dispatched.
    pub tool_sequence: Vec<String>,
    /// Reason the runner stopped — free-form, runner-specific.
    pub finish_reason: Option<String>,
    /// Wall-clock duration of the run.
    pub duration_ms: u64,
}

/// Drives a fixture to completion.
///
/// Implementations adapt whatever system you're testing (real agent, replay
/// trace, cached transcript) to produce a [`RunOutcome`] that [`FixtureCase`]
/// can evaluate against [`ExpectedBehavior`].
#[async_trait]
pub trait FixtureRunner: Send + Sync {
    /// Execute `fixture` and return the observed outcome.
    ///
    /// `trial_id` is forwarded so runners can seed randomness or choose a
    /// recorded trace for replay.
    async fn run(&self, fixture: &Fixture, trial_id: usize) -> Result<RunOutcome>;
}

/// An [`EvaluationCase`] built from a fixture + runner pair.
pub struct FixtureCase {
    fixture: Arc<Fixture>,
    runner: Arc<dyn FixtureRunner>,
}

impl FixtureCase {
    /// Build a case from a loaded fixture and a runner.
    pub fn new(fixture: Arc<Fixture>, runner: Arc<dyn FixtureRunner>) -> Self {
        Self { fixture, runner }
    }

    /// Access the wrapped fixture.
    pub fn fixture(&self) -> &Fixture {
        &self.fixture
    }
}

#[async_trait]
impl EvaluationCase for FixtureCase {
    fn name(&self) -> &str {
        &self.fixture.name
    }
    fn category(&self) -> &str {
        &self.fixture.category
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let started = std::time::Instant::now();
        let outcome = match self.runner.run(&self.fixture, trial_id).await {
            Ok(o) => o,
            Err(e) => {
                return Ok(TrialResult::failure(
                    trial_id,
                    started.elapsed().as_millis() as u64,
                    format!("runner error: {e:#}"),
                ));
            }
        };
        match evaluate(&self.fixture.expected, &outcome) {
            Ok(()) => Ok(TrialResult::success(trial_id, outcome.duration_ms)),
            Err(reason) => Ok(TrialResult::failure(trial_id, outcome.duration_ms, reason)),
        }
    }
}

/// Evaluate an outcome against the expected behaviour. Returns `Err(reason)`
/// describing the first failing assertion, or `Ok(())` if everything matches.
pub fn evaluate(expected: &ExpectedBehavior, outcome: &RunOutcome) -> Result<(), String> {
    if !expected.tool_sequence.is_empty() && expected.tool_sequence != outcome.tool_sequence {
        return Err(format!(
            "tool_sequence mismatch: expected {:?}, got {:?}",
            expected.tool_sequence, outcome.tool_sequence
        ));
    }
    for a in &expected.assertions {
        if let Some(needle) = &a.contains
            && !outcome.output_text.contains(needle.as_str())
        {
            return Err(format!(
                "output_text missing expected substring: {needle:?}"
            ));
        }
        if let Some(pat) = &a.regex {
            let re =
                Regex::new(pat).map_err(|e| format!("invalid regex in fixture: {pat:?} ({e})"))?;
            if !re.is_match(&outcome.output_text) {
                return Err(format!("output_text did not match regex: {pat:?}"));
            }
        }
        if let Some(name) = &a.tool_called
            && !outcome.tool_sequence.iter().any(|t| t == name)
        {
            return Err(format!(
                "expected tool `{name}` to be called; got {:?}",
                outcome.tool_sequence
            ));
        }
        if let Some(expected_reason) = &a.finish_reason {
            let got = outcome.finish_reason.as_deref().unwrap_or("");
            if got != expected_reason {
                return Err(format!(
                    "finish_reason mismatch: expected {expected_reason:?}, got {got:?}"
                ));
            }
        }
    }
    Ok(())
}

/// Load a single fixture YAML file.
pub fn load_fixture_file(path: impl AsRef<Path>) -> Result<Fixture> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading fixture {}", path.display()))?;
    let fixture: Fixture =
        serde_yml::from_str(&raw).with_context(|| format!("parsing fixture {}", path.display()))?;
    Ok(fixture)
}

/// Load every `.yaml` / `.yml` fixture file directly inside `dir`. Does not
/// recurse into subdirectories.
pub fn load_fixtures_from_dir(dir: impl AsRef<Path>) -> Result<Vec<Fixture>> {
    let dir = dir.as_ref();
    let mut out = Vec::new();
    let mut paths: Vec<PathBuf> = Vec::new();
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("reading fixture dir {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        match path.extension().and_then(|s| s.to_str()) {
            Some("yaml") | Some("yml") => paths.push(path),
            _ => {}
        }
    }
    // Deterministic order so test output is stable.
    paths.sort();
    for p in paths {
        out.push(load_fixture_file(&p)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn happy_outcome(seq: Vec<&str>, text: &str) -> RunOutcome {
        RunOutcome {
            output_text: text.to_string(),
            tool_sequence: seq.into_iter().map(String::from).collect(),
            finish_reason: Some("end_turn".into()),
            duration_ms: 5,
        }
    }

    fn contains(s: &str) -> Assertion {
        Assertion {
            contains: Some(s.into()),
            ..Default::default()
        }
    }
    fn tool_called(s: &str) -> Assertion {
        Assertion {
            tool_called: Some(s.into()),
            ..Default::default()
        }
    }
    fn finish_reason(s: &str) -> Assertion {
        Assertion {
            finish_reason: Some(s.into()),
            ..Default::default()
        }
    }
    fn regex_match(s: &str) -> Assertion {
        Assertion {
            regex: Some(s.into()),
            ..Default::default()
        }
    }

    #[test]
    fn evaluate_passes_when_all_assertions_hold() {
        let expected = ExpectedBehavior {
            tool_sequence: vec!["read_file".into(), "edit_file".into()],
            assertions: vec![
                contains("fn bar"),
                tool_called("edit_file"),
                finish_reason("end_turn"),
            ],
        };
        let outcome = happy_outcome(vec!["read_file", "edit_file"], "updated: fn bar() {}");
        evaluate(&expected, &outcome).expect("should pass");
    }

    #[test]
    fn evaluate_fails_on_tool_sequence_mismatch() {
        let expected = ExpectedBehavior {
            tool_sequence: vec!["read_file".into(), "edit_file".into()],
            ..Default::default()
        };
        let outcome = happy_outcome(vec!["edit_file"], "");
        let err = evaluate(&expected, &outcome).unwrap_err();
        assert!(err.contains("tool_sequence mismatch"));
    }

    #[test]
    fn evaluate_fails_on_missing_substring() {
        let expected = ExpectedBehavior {
            assertions: vec![contains("bar")],
            ..Default::default()
        };
        let outcome = happy_outcome(vec![], "only foo here");
        assert!(evaluate(&expected, &outcome).is_err());
    }

    #[test]
    fn evaluate_regex_assertion() {
        let expected = ExpectedBehavior {
            assertions: vec![regex_match(r"^updated:")],
            ..Default::default()
        };
        let outcome = happy_outcome(vec![], "updated: ok");
        evaluate(&expected, &outcome).expect("matches");
    }

    #[test]
    fn load_fixtures_from_tmpdir_in_sorted_order() {
        let dir = tempfile::tempdir().unwrap();
        let a = r#"
name: aa
category: test
messages:
  - { role: user, content: "hi" }
expected:
  assertions:
    - contains: "hi"
"#;
        let b = r#"
name: bb
category: test
messages:
  - { role: user, content: "go" }
expected:
  assertions:
    - finish_reason: end_turn
"#;
        std::fs::write(dir.path().join("a_first.yaml"), a).unwrap();
        std::fs::write(dir.path().join("b_second.yml"), b).unwrap();
        std::fs::write(dir.path().join("ignore_me.txt"), "").unwrap();

        let fixtures = load_fixtures_from_dir(dir.path()).unwrap();
        assert_eq!(fixtures.len(), 2);
        assert_eq!(fixtures[0].name, "aa");
        assert_eq!(fixtures[1].name, "bb");
    }

    struct StubRunner {
        outcome: RunOutcome,
    }
    #[async_trait]
    impl FixtureRunner for StubRunner {
        async fn run(&self, _: &Fixture, _: usize) -> Result<RunOutcome> {
            Ok(self.outcome.clone())
        }
    }

    #[tokio::test]
    async fn fixture_case_bridges_to_trial_result() {
        let fixture = Arc::new(Fixture {
            name: "f1".into(),
            category: "smoke".into(),
            model: None,
            messages: vec![FixtureMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            expected: ExpectedBehavior {
                tool_sequence: vec![],
                assertions: vec![contains("hi")],
            },
        });
        let runner = Arc::new(StubRunner {
            outcome: happy_outcome(vec![], "hi there"),
        });
        let case = FixtureCase::new(fixture.clone(), runner);
        let r = case.run(0).await.unwrap();
        assert!(r.success);

        // A failing fixture should surface as TrialResult::failure with the
        // reason string from evaluate() attached as the error.
        let fixture_bad = Arc::new(Fixture {
            expected: ExpectedBehavior {
                assertions: vec![contains("BYE")],
                ..fixture.expected.clone()
            },
            ..(*fixture).clone()
        });
        let runner = Arc::new(StubRunner {
            outcome: happy_outcome(vec![], "hi there"),
        });
        let case = FixtureCase::new(fixture_bad, runner);
        let r = case.run(0).await.unwrap();
        assert!(!r.success);
        assert!(
            r.error
                .as_deref()
                .unwrap()
                .contains("missing expected substring")
        );
    }
}
