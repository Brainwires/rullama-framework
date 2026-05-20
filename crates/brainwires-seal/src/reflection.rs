//! Reflection Module for Error Detection and Correction
//!
//! Provides post-execution analysis to detect issues and suggest corrections.
//! Implements error classification, root cause analysis, and correction strategies.
//!
//! ## Reflection Flow
//!
//! ```text
//! Query Execution Result
//!         │
//!         ▼
//! ┌───────────────────┐
//! │  Analyze Result   │ ◄── Check for empty results, overflow, etc.
//! └─────────┬─────────┘
//!           │
//!           ▼
//! ┌───────────────────┐
//! │  Classify Issues  │ ◄── Determine error types and severity
//! └─────────┬─────────┘
//!           │
//!           ▼
//! ┌───────────────────┐
//! │  Suggest Fixes    │ ◄── Generate correction strategies
//! └─────────┬─────────┘
//!           │
//!           ▼
//! ┌───────────────────┐
//! │ Attempt Correction│ ◄── Try fixes with retry limit
//! └───────────────────┘
//! ```
//!
//! ## Error Types
//!
//! - `EmptyResult`: Query returned no results
//! - `ResultOverflow`: Too many results to be useful
//! - `EntityNotFound`: Referenced entity doesn't exist
//! - `RelationMismatch`: Relationship type doesn't apply
//! - `CoreferenceFailure`: Could not resolve reference
//! - `SchemaAlignment`: Query structure doesn't match data
//!
//! ## Example
//!
//! ```rust,ignore
//! let mut reflection = ReflectionModule::new(ReflectionConfig::default());
//!
//! let report = reflection.analyze(&query_core, &result, &graph);
//! if !report.issues.is_empty() {
//!     if reflection.attempt_correction(&mut report, &graph, &executor) {
//!         // Use corrected_result
//!     }
//! }
//! ```

use super::learning::LearningCoordinator;
use super::query_core::{QueryCore, QueryExpr, QueryOp, QueryResult, RelationType};
use brainwires_core::graph::RelationshipGraphT;
use std::collections::HashMap;

/// Types of errors that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorType {
    /// Query returned no results
    EmptyResult,
    /// Too many results (overflow threshold exceeded)
    ResultOverflow,
    /// Referenced entity was not found in the graph
    EntityNotFound(String),
    /// Relationship type doesn't match entities
    RelationMismatch(String),
    /// Coreference resolution failed
    CoreferenceFailure(String),
    /// Query structure doesn't align with schema
    SchemaAlignment(String),
    /// Execution timeout
    Timeout,
    /// Unknown error
    Unknown(String),
}

impl ErrorType {
    /// Get a human-readable description
    pub fn description(&self) -> String {
        match self {
            ErrorType::EmptyResult => "Query returned no results".to_string(),
            ErrorType::ResultOverflow => "Query returned too many results".to_string(),
            ErrorType::EntityNotFound(name) => format!("Entity '{}' not found", name),
            ErrorType::RelationMismatch(rel) => format!("Relationship '{}' does not apply", rel),
            ErrorType::CoreferenceFailure(ref_text) => {
                format!("Could not resolve reference '{}'", ref_text)
            }
            ErrorType::SchemaAlignment(msg) => format!("Schema alignment issue: {}", msg),
            ErrorType::Timeout => "Query execution timed out".to_string(),
            ErrorType::Unknown(msg) => format!("Unknown error: {}", msg),
        }
    }
}

/// Severity level for issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational - not a problem
    Info,
    /// Warning - may indicate a problem
    Warning,
    /// Error - query failed but may be recoverable
    Error,
    /// Critical - unrecoverable error
    Critical,
}

impl Severity {
    /// Get the severity as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "error",
            Severity::Critical => "critical",
        }
    }
}

/// An issue detected during reflection
#[derive(Debug, Clone)]
pub struct Issue {
    /// Type of error
    pub error_type: ErrorType,
    /// Severity level
    pub severity: Severity,
    /// Human-readable message
    pub message: String,
    /// Suggested fixes
    pub suggested_fixes: Vec<SuggestedFix>,
    /// Source location in query (if applicable)
    pub source: Option<String>,
}

impl Issue {
    /// Create a new issue
    pub fn new(error_type: ErrorType, severity: Severity, message: &str) -> Self {
        Self {
            error_type,
            severity,
            message: message.to_string(),
            suggested_fixes: Vec::new(),
            source: None,
        }
    }

    /// Add a suggested fix
    pub fn with_fix(mut self, fix: SuggestedFix) -> Self {
        self.suggested_fixes.push(fix);
        self
    }

    /// Set the source location
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }
}

/// Suggested fix for an issue
#[derive(Debug, Clone)]
pub enum SuggestedFix {
    /// Retry with a modified query
    RetryWithQuery(QueryCore),
    /// Expand search scope by adding relationships
    ExpandScope {
        /// The relationship to expand search by.
        relation: String,
    },
    /// Narrow scope with a filter
    NarrowScope {
        /// The filter expression to apply.
        filter: String,
    },
    /// Suggest a different entity resolution
    ResolveEntity {
        /// The original entity name.
        original: String,
        /// The suggested replacement entity name.
        suggested: String,
    },
    /// Add a relationship that might be missing
    AddRelation {
        /// Source entity.
        from: String,
        /// Target entity.
        to: String,
        /// Relationship type.
        relation: String,
    },
    /// Requires manual intervention
    ManualIntervention(String),
}

impl SuggestedFix {
    /// Get a description of the fix
    pub fn description(&self) -> String {
        match self {
            SuggestedFix::RetryWithQuery(_) => "Retry with modified query".to_string(),
            SuggestedFix::ExpandScope { relation } => {
                format!("Expand scope to include {} relationships", relation)
            }
            SuggestedFix::NarrowScope { filter } => {
                format!("Narrow scope with filter: {}", filter)
            }
            SuggestedFix::ResolveEntity {
                original,
                suggested,
            } => {
                format!("Resolve '{}' as '{}'", original, suggested)
            }
            SuggestedFix::AddRelation { from, to, relation } => {
                format!("Add {} relationship from {} to {}", relation, from, to)
            }
            SuggestedFix::ManualIntervention(msg) => format!("Manual intervention: {}", msg),
        }
    }
}

/// Record of a correction attempt
#[derive(Debug, Clone)]
pub struct CorrectionRecord {
    /// The original issue
    pub issue: Issue,
    /// The fix that was attempted
    pub fix_applied: SuggestedFix,
    /// Whether the correction was successful
    pub success: bool,
    /// Timestamp
    pub timestamp: i64,
}

/// Reflection report
#[derive(Debug, Clone)]
pub struct ReflectionReport {
    /// Original query core
    pub query: QueryCore,
    /// Original result
    pub result: QueryResult,
    /// Issues detected
    pub issues: Vec<Issue>,
    /// Overall quality score (0.0 - 1.0)
    pub quality_score: f32,
    /// Whether correction was attempted
    pub correction_attempted: bool,
    /// Corrected query (if correction was attempted)
    pub corrected_query: Option<QueryCore>,
    /// Corrected result (if correction succeeded)
    pub corrected_result: Option<QueryResult>,
}

impl ReflectionReport {
    /// Create a new reflection report
    pub fn new(query: QueryCore, result: QueryResult) -> Self {
        Self {
            query,
            result,
            issues: Vec::new(),
            quality_score: 1.0,
            correction_attempted: false,
            corrected_query: None,
            corrected_result: None,
        }
    }

    /// Check if the result is acceptable
    pub fn is_acceptable(&self) -> bool {
        self.quality_score >= 0.5 && !self.issues.iter().any(|i| i.severity >= Severity::Error)
    }

    /// Get the highest severity issue
    pub fn max_severity(&self) -> Option<Severity> {
        self.issues.iter().map(|i| i.severity).max()
    }
}

/// Configuration for reflection
#[derive(Debug, Clone)]
pub struct ReflectionConfig {
    /// Maximum results before overflow warning
    pub max_results: usize,
    /// Minimum results before empty warning (if expecting results)
    pub min_results: usize,
    /// Maximum correction retries
    pub max_retries: u32,
    /// Whether to auto-apply simple corrections
    pub auto_correct: bool,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            max_results: 100,
            min_results: 1,
            max_retries: 2,
            auto_correct: true,
        }
    }
}

/// Reflection module for analyzing and correcting query results
pub struct ReflectionModule {
    /// Configuration
    config: ReflectionConfig,
    /// Error pattern counts for learning
    error_patterns: HashMap<ErrorType, u32>,
    /// History of correction attempts
    correction_history: Vec<CorrectionRecord>,
}

impl ReflectionModule {
    /// Create a new reflection module
    pub fn new(config: ReflectionConfig) -> Self {
        Self {
            config,
            error_patterns: HashMap::new(),
            correction_history: Vec::new(),
        }
    }

    /// Analyze a query result and generate a reflection report
    pub fn analyze(
        &mut self,
        query: &QueryCore,
        result: &QueryResult,
        graph: &dyn RelationshipGraphT,
    ) -> ReflectionReport {
        let mut report = ReflectionReport::new(query.clone(), result.clone());

        // Check for execution errors
        if let Some(ref error) = result.error {
            report.issues.push(Issue::new(
                ErrorType::Unknown(error.clone()),
                Severity::Error,
                error,
            ));
            report.quality_score = 0.0;
            return report;
        }

        // Check for empty results
        if result.values.is_empty() && result.count != Some(0) {
            let issue = self.analyze_empty_result(query, graph);
            report.issues.push(issue);
            report.quality_score = 0.3;
        }

        // Check for result overflow
        if result.values.len() > self.config.max_results {
            let issue = Issue::new(
                ErrorType::ResultOverflow,
                Severity::Warning,
                &format!(
                    "Query returned {} results (max: {})",
                    result.values.len(),
                    self.config.max_results
                ),
            )
            .with_fix(SuggestedFix::NarrowScope {
                filter: "Add type or name filter".to_string(),
            });
            report.issues.push(issue);
            report.quality_score = 0.6;
        }

        // Validate entities in the query
        for (entity_name, _entity_type) in &query.entities {
            if graph.get_node(entity_name).is_none() {
                let similar = self.find_similar_entities(entity_name, graph);
                let mut issue = Issue::new(
                    ErrorType::EntityNotFound(entity_name.clone()),
                    Severity::Warning,
                    &format!("Entity '{}' not found in graph", entity_name),
                );

                if let Some(suggestion) = similar.first() {
                    issue = issue.with_fix(SuggestedFix::ResolveEntity {
                        original: entity_name.clone(),
                        suggested: suggestion.clone(),
                    });
                }

                report.issues.push(issue);
                report.quality_score = (report.quality_score - 0.2).max(0.0);
            }
        }

        // Validate relationships
        self.validate_relationships(query, graph, &mut report);

        // Update error pattern counts
        for issue in &report.issues {
            *self
                .error_patterns
                .entry(issue.error_type.clone())
                .or_insert(0) += 1;
        }

        report
    }

    /// Analyze why a result is empty
    fn analyze_empty_result(&self, query: &QueryCore, graph: &dyn RelationshipGraphT) -> Issue {
        // Check if entities exist
        for (entity_name, _) in &query.entities {
            if graph.get_node(entity_name).is_none() {
                return Issue::new(
                    ErrorType::EntityNotFound(entity_name.clone()),
                    Severity::Error,
                    &format!(
                        "Entity '{}' not found - query cannot return results",
                        entity_name
                    ),
                );
            }
        }

        // Check if the relationship type applies
        if let Some(relation_msg) = self.check_relationship_applicability(&query.root, graph) {
            return Issue::new(
                ErrorType::RelationMismatch(relation_msg.clone()),
                Severity::Error,
                &relation_msg,
            )
            .with_fix(SuggestedFix::ExpandScope {
                relation: "CoOccurs".to_string(),
            });
        }

        // Generic empty result
        Issue::new(
            ErrorType::EmptyResult,
            Severity::Warning,
            "Query returned no results",
        )
        .with_fix(SuggestedFix::ExpandScope {
            relation: "All".to_string(),
        })
    }

    /// Check if a relationship type applies to the entities
    fn check_relationship_applicability(
        &self,
        expr: &QueryExpr,
        graph: &dyn RelationshipGraphT,
    ) -> Option<String> {
        match expr {
            QueryExpr::Op(QueryOp::Join {
                relation,
                subject,
                object,
            }) => {
                // Get entity names
                let subject_name = match subject.as_ref() {
                    QueryExpr::Constant(name, _) => Some(name.as_str()),
                    _ => None,
                };
                let object_name = match object.as_ref() {
                    QueryExpr::Constant(name, _) => Some(name.as_str()),
                    _ => None,
                };

                // Check if any edges of this type exist for the entities
                if let Some(name) = subject_name.or(object_name) {
                    let edges = graph.get_edges(name);
                    if let Some(edge_type) = relation.to_edge_type()
                        && !edges.iter().any(|e| e.edge_type == edge_type)
                    {
                        return Some(format!(
                            "No {:?} relationships found for '{}'",
                            relation, name
                        ));
                    }
                }

                None
            }
            _ => None,
        }
    }

    /// Find similar entity names
    fn find_similar_entities(&self, name: &str, graph: &dyn RelationshipGraphT) -> Vec<String> {
        let candidates = graph.search(name, 5);
        candidates
            .iter()
            .map(|node| node.entity_name.clone())
            .collect()
    }

    /// Validate relationships in the query
    fn validate_relationships(
        &self,
        query: &QueryCore,
        graph: &dyn RelationshipGraphT,
        report: &mut ReflectionReport,
    ) {
        self.validate_expr(&query.root, graph, report);
    }

    #[allow(clippy::only_used_in_recursion)]
    fn validate_expr(
        &self,
        expr: &QueryExpr,
        graph: &dyn RelationshipGraphT,
        report: &mut ReflectionReport,
    ) {
        match expr {
            QueryExpr::Op(QueryOp::Join {
                relation,
                subject,
                object,
            }) => {
                // Check if the relation type is valid for the entity types
                if relation.to_edge_type().is_none()
                    && !matches!(
                        relation,
                        RelationType::HasType
                            | RelationType::HasError
                            | RelationType::CreatedAt
                            | RelationType::ModifiedAt
                    )
                    && let RelationType::Custom(name) = relation
                {
                    report.issues.push(
                        Issue::new(
                            ErrorType::RelationMismatch(name.clone()),
                            Severity::Warning,
                            &format!("Custom relationship '{}' may not exist", name),
                        )
                        .with_source(&format!("{:?}", relation)),
                    );
                }

                // Recursively validate sub-expressions
                self.validate_expr(subject, graph, report);
                self.validate_expr(object, graph, report);
            }
            QueryExpr::Op(QueryOp::And(exprs)) | QueryExpr::Op(QueryOp::Or(exprs)) => {
                for e in exprs {
                    self.validate_expr(e, graph, report);
                }
            }
            QueryExpr::Op(QueryOp::Filter { source, .. }) => {
                self.validate_expr(source, graph, report);
            }
            QueryExpr::Op(QueryOp::Count(inner)) => {
                self.validate_expr(inner, graph, report);
            }
            QueryExpr::Op(QueryOp::Superlative { source, .. }) => {
                self.validate_expr(source, graph, report);
            }
            _ => {}
        }
    }

    /// Validate a query core structure (before execution)
    pub fn validate_query_core(&self, query: &QueryCore) -> Vec<Issue> {
        let mut issues = Vec::new();

        // Check for missing entities
        if query.entities.is_empty() {
            issues.push(Issue::new(
                ErrorType::SchemaAlignment("No entities in query".to_string()),
                Severity::Warning,
                "Query does not reference any entities",
            ));
        }

        // Check for valid question type
        if matches!(
            query.question_type,
            super::query_core::QuestionType::Unknown
        ) {
            issues.push(Issue::new(
                ErrorType::SchemaAlignment("Unknown question type".to_string()),
                Severity::Info,
                "Could not determine question type",
            ));
        }

        issues
    }

    /// Attempt to correct issues in a report
    pub fn attempt_correction(
        &mut self,
        report: &mut ReflectionReport,
        graph: &dyn RelationshipGraphT,
        _executor: &super::query_core::QueryExecutor,
    ) -> bool {
        if !self.config.auto_correct {
            return false;
        }

        if report.issues.is_empty() {
            return true; // Nothing to correct
        }

        report.correction_attempted = true;

        // Try to apply fixes for each issue
        for issue in &report.issues {
            if issue.severity < Severity::Warning {
                continue;
            }

            for fix in &issue.suggested_fixes {
                match fix {
                    SuggestedFix::ResolveEntity {
                        original,
                        suggested,
                    } => {
                        // Try resolving to a different entity
                        if graph.get_node(suggested).is_some() {
                            // Create a corrected query by substituting the entity
                            let corrected =
                                self.substitute_entity(&report.query, original, suggested);
                            if let Some(corrected) = corrected {
                                report.corrected_query = Some(corrected);
                                // Note: Would re-execute here, but we don't have executor access
                                // in a way that allows us to return the result
                                self.record_correction(issue.clone(), fix.clone(), true);
                                return true;
                            }
                        }
                    }
                    SuggestedFix::ExpandScope { .. } => {
                        // Would need to modify query to use different relationship
                        // This is more complex and requires query rewriting
                    }
                    _ => {}
                }
            }
        }

        false
    }

    /// Substitute an entity in a query
    fn substitute_entity(
        &self,
        query: &QueryCore,
        original: &str,
        replacement: &str,
    ) -> Option<QueryCore> {
        let mut corrected = query.clone();

        // Update entities list
        for (name, _) in &mut corrected.entities {
            if name == original {
                *name = replacement.to_string();
            }
        }

        // Update the expression tree
        corrected.root = Self::substitute_in_expr(&query.root, original, replacement);

        Some(corrected)
    }

    fn substitute_in_expr(expr: &QueryExpr, original: &str, replacement: &str) -> QueryExpr {
        match expr {
            QueryExpr::Constant(name, entity_type) => {
                if name == original {
                    QueryExpr::Constant(replacement.to_string(), entity_type.clone())
                } else {
                    expr.clone()
                }
            }
            QueryExpr::Op(op) => QueryExpr::Op(match op {
                QueryOp::Join {
                    relation,
                    subject,
                    object,
                } => QueryOp::Join {
                    relation: relation.clone(),
                    subject: Box::new(Self::substitute_in_expr(subject, original, replacement)),
                    object: Box::new(Self::substitute_in_expr(object, original, replacement)),
                },
                QueryOp::And(exprs) => QueryOp::And(
                    exprs
                        .iter()
                        .map(|e| Self::substitute_in_expr(e, original, replacement))
                        .collect(),
                ),
                QueryOp::Or(exprs) => QueryOp::Or(
                    exprs
                        .iter()
                        .map(|e| Self::substitute_in_expr(e, original, replacement))
                        .collect(),
                ),
                QueryOp::Filter { source, predicate } => QueryOp::Filter {
                    source: Box::new(Self::substitute_in_expr(source, original, replacement)),
                    predicate: predicate.clone(),
                },
                QueryOp::Count(inner) => QueryOp::Count(Box::new(Self::substitute_in_expr(
                    inner,
                    original,
                    replacement,
                ))),
                QueryOp::Superlative {
                    source,
                    property,
                    direction,
                } => QueryOp::Superlative {
                    source: Box::new(Self::substitute_in_expr(source, original, replacement)),
                    property: property.clone(),
                    direction: direction.clone(),
                },
                _ => op.clone(),
            }),
            _ => expr.clone(),
        }
    }

    /// Record a correction attempt
    fn record_correction(&mut self, issue: Issue, fix: SuggestedFix, success: bool) {
        self.correction_history.push(CorrectionRecord {
            issue,
            fix_applied: fix,
            success,
            timestamp: chrono::Utc::now().timestamp(),
        });
    }

    /// Provide feedback to the learning coordinator
    pub fn provide_feedback(
        &self,
        report: &ReflectionReport,
        coordinator: &mut LearningCoordinator,
    ) {
        // Record the query outcome
        let success = report.is_acceptable();
        let result_count = report.result.values.len();

        coordinator.record_outcome(
            None, // Pattern ID not tracked here
            success,
            result_count,
            Some(&report.query),
            0, // Reflection doesn't track query execution timing
        );

        // Track specific error patterns for future avoidance
        for issue in &report.issues {
            if issue.severity >= Severity::Error {
                // This could be used to create negative patterns
                // that the learning system avoids
            }
        }
    }

    /// Get error statistics
    pub fn get_error_stats(&self) -> HashMap<ErrorType, u32> {
        self.error_patterns.clone()
    }

    /// Get correction success rate
    pub fn correction_success_rate(&self) -> f32 {
        if self.correction_history.is_empty() {
            return 0.0;
        }

        let successes = self.correction_history.iter().filter(|r| r.success).count();
        successes as f32 / self.correction_history.len() as f32
    }
}

impl Default for ReflectionModule {
    fn default() -> Self {
        Self::new(ReflectionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_core::{QueryExpr, QueryResultValue, QuestionType};
    use brainwires_core::graph::EntityType;
    use brainwires_knowledge::RelationshipGraph;

    fn create_test_query() -> QueryCore {
        QueryCore::new(
            QuestionType::Definition,
            QueryExpr::var("x"),
            vec![("main.rs".to_string(), EntityType::File)],
            "What is main.rs?".to_string(),
        )
    }

    #[test]
    fn test_analyze_empty_result() {
        let mut reflection = ReflectionModule::new(ReflectionConfig::default());
        let query = create_test_query();
        let result = QueryResult::empty();
        let graph = RelationshipGraph::new();

        let report = reflection.analyze(&query, &result, &graph);

        assert!(!report.issues.is_empty());
        assert!(report.issues.iter().any(|i| matches!(
            i.error_type,
            ErrorType::EmptyResult | ErrorType::EntityNotFound(_)
        )));
    }

    #[test]
    fn test_analyze_overflow_result() {
        let mut reflection = ReflectionModule::new(ReflectionConfig {
            max_results: 10,
            ..Default::default()
        });
        let query = create_test_query();

        // Create result with many values
        let mut values = Vec::new();
        for i in 0..20 {
            values.push(QueryResultValue {
                value: format!("entity_{}", i),
                entity_type: Some(EntityType::File),
                score: 0.8,
                metadata: std::collections::HashMap::new(),
            });
        }
        let result = crate::query_core::QueryResult::with_values(values);
        let graph = RelationshipGraph::new();

        let report = reflection.analyze(&query, &result, &graph);

        assert!(
            report
                .issues
                .iter()
                .any(|i| i.error_type == ErrorType::ResultOverflow)
        );
    }

    #[test]
    fn test_validate_query_core() {
        let reflection = ReflectionModule::new(ReflectionConfig::default());

        // Query with no entities
        let query = QueryCore::new(
            QuestionType::Unknown,
            QueryExpr::var("x"),
            vec![],
            "Test".to_string(),
        );

        let issues = reflection.validate_query_core(&query);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_issue_creation() {
        let issue = Issue::new(ErrorType::EmptyResult, Severity::Warning, "No results")
            .with_fix(SuggestedFix::ExpandScope {
                relation: "All".to_string(),
            })
            .with_source("query_root");

        assert_eq!(issue.severity, Severity::Warning);
        assert_eq!(issue.suggested_fixes.len(), 1);
        assert!(issue.source.is_some());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_reflection_report_acceptable() {
        let query = create_test_query();
        let result = QueryResult::empty();
        let mut report = ReflectionReport::new(query, result);

        report.quality_score = 0.7;
        assert!(report.is_acceptable());

        report.quality_score = 0.3;
        assert!(!report.is_acceptable());

        report.quality_score = 0.7;
        report
            .issues
            .push(Issue::new(ErrorType::EmptyResult, Severity::Error, "Error"));
        assert!(!report.is_acceptable());
    }

    #[test]
    fn test_suggested_fix_description() {
        let fix = SuggestedFix::ResolveEntity {
            original: "main".to_string(),
            suggested: "main.rs".to_string(),
        };

        let desc = fix.description();
        assert!(desc.contains("main"));
        assert!(desc.contains("main.rs"));
    }

    #[test]
    fn test_error_type_description() {
        let error = ErrorType::EntityNotFound("test.rs".to_string());
        let desc = error.description();
        assert!(desc.contains("test.rs"));
    }

    #[test]
    fn test_substitute_entity() {
        let reflection = ReflectionModule::new(ReflectionConfig::default());

        let query = QueryCore::new(
            QuestionType::Definition,
            QueryExpr::constant("main", EntityType::File),
            vec![("main".to_string(), EntityType::File)],
            "What is main?".to_string(),
        );

        let corrected = reflection.substitute_entity(&query, "main", "main.rs");
        assert!(corrected.is_some());

        let corrected = corrected.unwrap();
        assert!(corrected.entities.iter().any(|(name, _)| name == "main.rs"));
    }

    #[test]
    fn test_correction_success_rate() {
        let mut reflection = ReflectionModule::new(ReflectionConfig::default());

        assert_eq!(reflection.correction_success_rate(), 0.0);

        reflection.record_correction(
            Issue::new(ErrorType::EmptyResult, Severity::Warning, "test"),
            SuggestedFix::ExpandScope {
                relation: "All".to_string(),
            },
            true,
        );

        reflection.record_correction(
            Issue::new(ErrorType::EmptyResult, Severity::Warning, "test"),
            SuggestedFix::ExpandScope {
                relation: "All".to_string(),
            },
            false,
        );

        assert_eq!(reflection.correction_success_rate(), 0.5);
    }
}
