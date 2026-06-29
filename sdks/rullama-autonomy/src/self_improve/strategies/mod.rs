//! Improvement strategies for automated code quality.
//!
//! Each strategy scans a repository for a specific class of issue (linting,
//! dead code, missing docs, test gaps, etc.) and produces [`ImprovementTask`]s
//! that the controller can execute autonomously.

pub mod clippy;
pub mod dead_code;
pub mod doc_gaps;
pub mod refactoring;
pub mod test_coverage;
pub mod todo_scanner;

#[cfg(feature = "eval-driven")]
pub mod eval_strategy;

use anyhow::Result;
use async_trait::async_trait;

use crate::config::StrategyConfig;

/// Categories of improvement tasks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImprovementCategory {
    /// Linting and clippy warnings.
    Linting,
    /// Test coverage gaps.
    Testing,
    /// Missing or incomplete documentation.
    Documentation,
    /// Code refactoring opportunities.
    Refactoring,
    /// Dead or unreachable code.
    DeadCode,
    /// Eval-suite-driven improvements.
    EvalDriven,
}

impl std::fmt::Display for ImprovementCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImprovementCategory::Linting => write!(f, "linting"),
            ImprovementCategory::Testing => write!(f, "testing"),
            ImprovementCategory::Documentation => write!(f, "documentation"),
            ImprovementCategory::Refactoring => write!(f, "refactoring"),
            ImprovementCategory::DeadCode => write!(f, "dead_code"),
            ImprovementCategory::EvalDriven => write!(f, "eval_driven"),
        }
    }
}

/// A generated improvement task describing a concrete code quality fix.
#[derive(Debug, Clone)]
pub struct ImprovementTask {
    /// Unique task identifier.
    pub id: String,
    /// Name of the strategy that generated this task.
    pub strategy: String,
    /// Category of improvement.
    pub category: ImprovementCategory,
    /// Human-readable description of the task.
    pub description: String,
    /// Files targeted by this task.
    pub target_files: Vec<String>,
    /// Priority (higher = more important).
    pub priority: u8,
    /// Estimated number of diff lines this task will produce.
    pub estimated_diff_lines: u32,
    /// Additional context for the AI agent.
    pub context: String,
}

/// Trait for improvement strategies that scan a repository and generate fix tasks.
///
/// Each strategy focuses on a single category (linting, testing, docs, etc.)
/// and is run by the [`TaskGenerator`](super::TaskGenerator) during a session.
#[async_trait]
pub trait ImprovementStrategy: Send + Sync {
    /// Return the strategy name.
    fn name(&self) -> &str;
    /// Return the improvement category.
    fn category(&self) -> ImprovementCategory;
    /// Generate improvement tasks by scanning the repository.
    async fn generate_tasks(
        &self,
        repo_path: &str,
        config: &StrategyConfig,
    ) -> Result<Vec<ImprovementTask>>;
}

/// Create the default set of all built-in strategies (clippy, todo scanner,
/// doc gaps, test coverage, refactoring, dead code).
pub fn all_strategies() -> Vec<Box<dyn ImprovementStrategy>> {
    vec![
        Box::new(clippy::ClippyStrategy),
        Box::new(todo_scanner::TodoScannerStrategy),
        Box::new(doc_gaps::DocGapsStrategy),
        Box::new(test_coverage::TestCoverageStrategy),
        Box::new(refactoring::RefactoringStrategy),
        Box::new(dead_code::DeadCodeStrategy),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_strategies_returns_expected_count() {
        let strategies = all_strategies();
        assert_eq!(strategies.len(), 6);
    }

    #[test]
    fn improvement_category_display_formatting() {
        assert_eq!(ImprovementCategory::Linting.to_string(), "linting");
        assert_eq!(ImprovementCategory::Testing.to_string(), "testing");
        assert_eq!(
            ImprovementCategory::Documentation.to_string(),
            "documentation"
        );
        assert_eq!(ImprovementCategory::Refactoring.to_string(), "refactoring");
        assert_eq!(ImprovementCategory::DeadCode.to_string(), "dead_code");
        assert_eq!(ImprovementCategory::EvalDriven.to_string(), "eval_driven");
    }
}
