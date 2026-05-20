//! The [`EvaluationCase`] trait — the unit of evaluation.
//!
//! Implement this trait for any scenario you want to evaluate N times.

use async_trait::async_trait;

use super::trial::TrialResult;

/// A single evaluation scenario.
///
/// Implement this trait and pass instances to
/// [`EvaluationSuite`](crate::suite::EvaluationSuite) to run N independent
/// trials and compute statistics.
///
/// ```rust,ignore
/// use brainwires_eval::{EvaluationCase, TrialResult};
/// use async_trait::async_trait;
///
/// struct MyCase;
///
/// #[async_trait]
/// impl EvaluationCase for MyCase {
///     fn name(&self) -> &str { "my_case" }
///     fn category(&self) -> &str { "smoke" }
///     async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
///         let start = std::time::Instant::now();
///         let ok = do_the_thing().await.is_ok();
///         let ms = start.elapsed().as_millis() as u64;
///         Ok(if ok {
///             TrialResult::success(trial_id, ms)
///         } else {
///             TrialResult::failure(trial_id, ms, "thing failed")
///         })
///     }
/// }
/// ```
#[async_trait]
pub trait EvaluationCase: Send + Sync {
    /// Short identifier used in reports and log output.
    fn name(&self) -> &str;

    /// Category label for grouping (e.g. `"smoke"`, `"adversarial"`,
    /// `"budget_stress"`).
    fn category(&self) -> &str;

    /// Execute one trial and return its result.
    ///
    /// The implementation is responsible for measuring wall-clock duration and
    /// encoding it in the returned [`TrialResult`].
    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult>;
}

/// A minimal no-op evaluation case useful for unit-testing the evaluation
/// infrastructure itself.
pub struct AlwaysPassCase {
    /// Short identifier for this case.
    pub name: String,
    /// Category label for grouping.
    pub category: String,
    /// Simulated duration in milliseconds returned by each trial.
    pub duration_ms: u64,
}

impl AlwaysPassCase {
    /// Create a new always-passing case with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            category: "test".into(),
            duration_ms: 0,
        }
    }

    /// Set the simulated duration in milliseconds for each trial.
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }
}

#[async_trait]
impl EvaluationCase for AlwaysPassCase {
    fn name(&self) -> &str {
        &self.name
    }
    fn category(&self) -> &str {
        &self.category
    }
    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
        Ok(TrialResult::success(trial_id, self.duration_ms))
    }
}

/// A no-op evaluation case that always fails — useful for testing failure paths.
pub struct AlwaysFailCase {
    /// Short identifier for this case.
    pub name: String,
    /// Category label for grouping.
    pub category: String,
    /// Error message returned by each trial.
    pub error_msg: String,
}

impl AlwaysFailCase {
    /// Create a new always-failing case with the given name and error message.
    pub fn new(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            category: "test".into(),
            error_msg: error.into(),
        }
    }
}

#[async_trait]
impl EvaluationCase for AlwaysFailCase {
    fn name(&self) -> &str {
        &self.name
    }
    fn category(&self) -> &str {
        &self.category
    }
    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
        Ok(TrialResult::failure(trial_id, 0, self.error_msg.clone()))
    }
}

/// A case that succeeds with a configurable probability (for testing statistics).
pub struct StochasticCase {
    /// Short identifier for this case.
    pub name: String,
    /// Probability of success per trial (0.0-1.0).
    pub success_rate: f64,
}

impl StochasticCase {
    /// Create a new stochastic case with the given name and success probability.
    pub fn new(name: impl Into<String>, success_rate: f64) -> Self {
        Self {
            name: name.into(),
            success_rate: success_rate.clamp(0.0, 1.0),
        }
    }
}

#[async_trait]
impl EvaluationCase for StochasticCase {
    fn name(&self) -> &str {
        &self.name
    }
    fn category(&self) -> &str {
        "stochastic"
    }
    async fn run(&self, trial_id: usize) -> anyhow::Result<TrialResult> {
        // Deterministic per trial_id so tests are reproducible.
        // Uses a simple LCG hash: seed = trial_id * prime, mapped to [0, 1).
        let seed = (trial_id as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map the full u64 range to [0, 1) uniformly.
        let norm = seed as f64 / u64::MAX as f64;
        if norm < self.success_rate {
            Ok(TrialResult::success(trial_id, 1))
        } else {
            Ok(TrialResult::failure(trial_id, 1, "stochastic failure"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_always_pass_case() {
        let case = AlwaysPassCase::new("test").with_duration(5);
        let result = case.run(0).await.unwrap();
        assert!(result.success);
        assert_eq!(result.trial_id, 0);
        assert_eq!(result.duration_ms, 5);
    }

    #[tokio::test]
    async fn test_always_fail_case() {
        let case = AlwaysFailCase::new("test", "oops");
        let result = case.run(3).await.unwrap();
        assert!(!result.success);
        assert_eq!(result.trial_id, 3);
        assert_eq!(result.error.as_deref(), Some("oops"));
    }

    #[tokio::test]
    async fn test_stochastic_case_reproducible() {
        let case = StochasticCase::new("test", 0.7);
        let r1 = case.run(42).await.unwrap();
        let r2 = case.run(42).await.unwrap();
        assert_eq!(
            r1.success, r2.success,
            "same trial_id must give same result"
        );
    }

    #[tokio::test]
    async fn test_stochastic_case_rate() {
        let case = StochasticCase::new("test", 0.6);
        let mut successes = 0usize;
        for i in 0..200 {
            if case.run(i).await.unwrap().success {
                successes += 1;
            }
        }
        // Allow ±15 % variance around the expected ~120 successes
        assert!(
            successes > 90 && successes < 170,
            "expected ~120 successes, got {}",
            successes
        );
    }
}
