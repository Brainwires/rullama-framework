//! Self-Evolving Learning Mechanism
//!
//! Enables the system to learn from successful interactions without retraining.
//! Implements both local (per-session) and global (cross-session) memory.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │         Learning Coordinator        │
//! │                                     │
//! │  ┌─────────────┐  ┌─────────────┐  │
//! │  │Local Memory │  │Global Memory│  │
//! │  │ (Session)   │  │ (Persisted) │  │
//! │  └─────────────┘  └─────────────┘  │
//! └─────────────────────────────────────┘
//! ```
//!
//! ## Local Memory (Per-Session)
//!
//! - Tracks entities mentioned in the current conversation
//! - Maintains coreference resolution history
//! - Stores query patterns used in this session
//! - Focus stack for active entities
//!
//! ## Global Memory (Cross-Session)
//!
//! - Template library organized by question type
//! - Query patterns with success/failure statistics
//! - Resolution patterns that worked well
//! - Persisted to LanceDB for retrieval
//!
//! ## Learning Flow
//!
//! 1. User query is processed
//! 2. Query core is extracted
//! 3. Check global memory for similar patterns
//! 4. Execute query
//! 5. Record outcome (success/failure, result count)
//! 6. If successful: generalize pattern, add to global memory
//! 7. If failed: record failure for pattern avoidance

use super::query_core::{QueryCore, QuestionType};
use brainwires_core::confidence::ResponseConfidence;
use brainwires_core::graph::EntityType;
use brainwires_tool_runtime::{ToolErrorCategory, ToolOutcome};
use chrono::Utc;
use std::collections::HashMap;

/// A tracked entity in local memory
#[derive(Debug, Clone)]
pub struct TrackedEntity {
    /// Entity name
    pub name: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Turn numbers when mentioned
    pub mention_turns: Vec<u32>,
    /// Whether this entity was queried about
    pub was_queried: bool,
    /// Whether this entity was modified
    pub was_modified: bool,
    /// Relationships discovered for this entity
    pub discovered_relations: Vec<(String, String)>, // (relation_type, target)
}

impl TrackedEntity {
    /// Create a new tracked entity
    pub fn new(name: String, entity_type: EntityType, turn: u32) -> Self {
        Self {
            name,
            entity_type,
            mention_turns: vec![turn],
            was_queried: false,
            was_modified: false,
            discovered_relations: Vec::new(),
        }
    }

    /// Record a mention
    pub fn record_mention(&mut self, turn: u32) {
        if !self.mention_turns.contains(&turn) {
            self.mention_turns.push(turn);
        }
    }

    /// Get the frequency of mentions
    pub fn frequency(&self) -> usize {
        self.mention_turns.len()
    }
}

/// Record of a coreference resolution
#[derive(Debug, Clone)]
pub struct CoreferenceRecord {
    /// The original reference text
    pub reference: String,
    /// The resolved entity
    pub resolved_to: String,
    /// Confidence of the resolution
    pub confidence: f32,
    /// Turn when resolved
    pub turn: u32,
    /// Whether the resolution was confirmed correct
    pub confirmed: Option<bool>,
}

/// Record of a query execution
#[derive(Debug, Clone)]
pub struct QueryRecord {
    /// Original query text
    pub original: String,
    /// Resolved query (after coreference)
    pub resolved: String,
    /// Question type
    pub question_type: QuestionType,
    /// Query core S-expression
    pub query_sexp: Option<String>,
    /// Turn when executed
    pub turn: u32,
    /// Whether successful
    pub success: bool,
    /// Number of results
    pub result_count: usize,
    /// Execution time in ms
    pub execution_time_ms: u64,
}

/// Local memory for a single conversation session
#[derive(Debug)]
pub struct LocalMemory {
    /// Conversation ID
    pub conversation_id: String,
    /// Tracked entities
    pub entities: HashMap<String, TrackedEntity>,
    /// Coreference resolution history
    pub coreference_log: Vec<CoreferenceRecord>,
    /// Query history
    pub query_history: Vec<QueryRecord>,
    /// Current focus stack (entity names)
    pub focus_stack: Vec<String>,
    /// Current turn number
    pub current_turn: u32,
}

impl LocalMemory {
    /// Create new local memory for a conversation
    pub fn new(conversation_id: String) -> Self {
        Self {
            conversation_id,
            entities: HashMap::new(),
            coreference_log: Vec::new(),
            query_history: Vec::new(),
            focus_stack: Vec::new(),
            current_turn: 0,
        }
    }

    /// Advance to the next turn
    pub fn next_turn(&mut self) {
        self.current_turn += 1;
    }

    /// Track an entity mention
    pub fn track_entity(&mut self, name: &str, entity_type: EntityType) {
        if let Some(entity) = self.entities.get_mut(name) {
            entity.record_mention(self.current_turn);
        } else {
            self.entities.insert(
                name.to_string(),
                TrackedEntity::new(name.to_string(), entity_type, self.current_turn),
            );
        }

        // Update focus stack
        self.focus_stack.retain(|n| n != name);
        self.focus_stack.insert(0, name.to_string());
        if self.focus_stack.len() > 20 {
            self.focus_stack.truncate(20);
        }
    }

    /// Record a coreference resolution
    pub fn record_coreference(&mut self, reference: &str, resolved_to: &str, confidence: f32) {
        self.coreference_log.push(CoreferenceRecord {
            reference: reference.to_string(),
            resolved_to: resolved_to.to_string(),
            confidence,
            turn: self.current_turn,
            confirmed: None,
        });
    }

    /// Record a query execution
    #[allow(clippy::too_many_arguments)]
    pub fn record_query(
        &mut self,
        original: &str,
        resolved: &str,
        question_type: QuestionType,
        query_sexp: Option<String>,
        success: bool,
        result_count: usize,
        execution_time_ms: u64,
    ) {
        self.query_history.push(QueryRecord {
            original: original.to_string(),
            resolved: resolved.to_string(),
            question_type,
            query_sexp,
            turn: self.current_turn,
            success,
            result_count,
            execution_time_ms,
        });
    }

    /// Get entities by frequency (most frequent first)
    pub fn get_frequent_entities(&self, limit: usize) -> Vec<&TrackedEntity> {
        let mut entities: Vec<_> = self.entities.values().collect();
        entities.sort_by_key(|e| std::cmp::Reverse(e.frequency()));
        entities.into_iter().take(limit).collect()
    }

    /// Get recent coreference resolutions
    pub fn get_recent_coreferences(&self, count: usize) -> Vec<&CoreferenceRecord> {
        self.coreference_log.iter().rev().take(count).collect()
    }

    /// Get success rate for a question type
    pub fn get_success_rate(&self, question_type: &QuestionType) -> f32 {
        let relevant: Vec<_> = self
            .query_history
            .iter()
            .filter(|q| &q.question_type == question_type)
            .collect();

        if relevant.is_empty() {
            return 0.5; // No data, neutral assumption
        }

        let successes = relevant.iter().filter(|q| q.success).count();
        successes as f32 / relevant.len() as f32
    }
}

/// A learned query pattern
#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// Unique pattern ID
    pub id: String,
    /// Question type this pattern applies to
    pub question_type: QuestionType,
    /// Template for the query (with placeholders)
    pub template: String,
    /// Required entity types
    pub required_types: Vec<EntityType>,
    /// Number of successful uses
    pub success_count: u32,
    /// Number of failed uses
    pub failure_count: u32,
    /// Average number of results
    pub avg_results: f32,
    /// When this pattern was created
    pub created_at: i64,
    /// When this pattern was last used
    pub last_used_at: i64,
}

impl QueryPattern {
    /// Create a new query pattern
    pub fn new(
        question_type: QuestionType,
        template: String,
        required_types: Vec<EntityType>,
    ) -> Self {
        let now = Utc::now().timestamp();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            question_type,
            template,
            required_types,
            success_count: 0,
            failure_count: 0,
            avg_results: 0.0,
            created_at: now,
            last_used_at: now,
        }
    }

    /// Compute reliability score (0.0 - 1.0)
    pub fn reliability(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 0.5; // No data, neutral
        }
        self.success_count as f32 / total as f32
    }

    /// Record a successful use
    pub fn record_success(&mut self, result_count: usize) {
        self.success_count += 1;
        self.last_used_at = Utc::now().timestamp();

        // Update average results with exponential moving average
        let alpha = 0.3;
        self.avg_results = alpha * result_count as f32 + (1.0 - alpha) * self.avg_results;
    }

    /// Record a failed use
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_used_at = Utc::now().timestamp();
    }

    /// Check if this pattern matches the given entity types
    pub fn matches_types(&self, types: &[EntityType]) -> bool {
        self.required_types.iter().all(|rt| types.contains(rt))
    }
}

/// A learned coreference resolution pattern
#[derive(Debug, Clone)]
pub struct ResolutionPattern {
    /// Reference type (e.g., "it", "the file")
    pub reference_type: String,
    /// Entity type that was resolved
    pub entity_type: EntityType,
    /// Context pattern (what typically precedes this reference)
    pub context_pattern: Option<String>,
    /// Success count
    pub success_count: u32,
    /// Failure count
    pub failure_count: u32,
}

/// A learned tool error pattern for avoiding repeated failures
#[derive(Debug, Clone)]
pub struct ToolErrorPattern {
    /// Tool name this pattern applies to
    pub tool_name: String,
    /// Error category (serialized for storage)
    pub error_category: String,
    /// Number of times this error occurred
    pub occurrence_count: u32,
    /// Last occurrence timestamp
    pub last_occurred: i64,
    /// Suggested fix or avoidance strategy
    pub suggested_fix: Option<String>,
    /// Input patterns that led to this error (for prevention)
    pub input_patterns: Vec<String>,
}

impl ToolErrorPattern {
    /// Create a new error pattern
    pub fn new(tool_name: &str, error_category: &ToolErrorCategory) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            error_category: error_category.category_name().to_string(),
            occurrence_count: 1,
            last_occurred: Utc::now().timestamp(),
            suggested_fix: error_category.get_suggestion(),
            input_patterns: Vec::new(),
        }
    }

    /// Record another occurrence
    pub fn record_occurrence(&mut self) {
        self.occurrence_count += 1;
        self.last_occurred = Utc::now().timestamp();
    }

    /// Check if this pattern is frequent (warrants attention)
    pub fn is_frequent(&self) -> bool {
        self.occurrence_count >= 3
    }
}

/// Tool execution statistics for learning
#[derive(Debug, Clone, Default)]
pub struct ToolStats {
    /// Number of successful executions
    pub success_count: u32,
    /// Number of failed executions
    pub failure_count: u32,
    /// Total retries needed
    pub total_retries: u32,
    /// Average execution time in ms
    pub avg_execution_time_ms: f64,
    /// Last used timestamp
    pub last_used: i64,
}

impl ToolStats {
    /// Record a successful execution
    pub fn record_success(&mut self, retries: u32, execution_time_ms: u64) {
        self.success_count += 1;
        self.total_retries += retries;
        self.last_used = Utc::now().timestamp();

        // Update average execution time with exponential moving average
        let alpha = 0.3;
        self.avg_execution_time_ms =
            alpha * execution_time_ms as f64 + (1.0 - alpha) * self.avg_execution_time_ms;
    }

    /// Record a failed execution
    pub fn record_failure(&mut self, retries: u32, execution_time_ms: u64) {
        self.failure_count += 1;
        self.total_retries += retries;
        self.last_used = Utc::now().timestamp();

        let alpha = 0.3;
        self.avg_execution_time_ms =
            alpha * execution_time_ms as f64 + (1.0 - alpha) * self.avg_execution_time_ms;
    }

    /// Get the success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.5 // Neutral when no data
        } else {
            self.success_count as f64 / total as f64
        }
    }

    /// Get average retries per execution
    pub fn avg_retries(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.total_retries as f64 / total as f64
        }
    }
}

/// Response confidence statistics for learning prompt patterns
#[derive(Debug, Clone, Default)]
pub struct ConfidenceStats {
    /// Total samples recorded
    pub sample_count: u32,
    /// Sum of confidence scores
    pub confidence_sum: f64,
    /// Number of low confidence responses
    pub low_confidence_count: u32,
    /// Number of high confidence responses
    pub high_confidence_count: u32,
}

impl ConfidenceStats {
    /// Record a confidence sample
    pub fn record_sample(&mut self, confidence: &ResponseConfidence) {
        self.sample_count += 1;
        self.confidence_sum += confidence.score;

        if confidence.is_low_confidence() {
            self.low_confidence_count += 1;
        } else if confidence.is_high_confidence() {
            self.high_confidence_count += 1;
        }
    }

    /// Get average confidence
    pub fn avg_confidence(&self) -> f64 {
        if self.sample_count == 0 {
            0.5
        } else {
            self.confidence_sum / self.sample_count as f64
        }
    }

    /// Get the ratio of low confidence responses
    pub fn low_confidence_ratio(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.low_confidence_count as f64 / self.sample_count as f64
        }
    }
}

/// A structured hint derived from behavioral knowledge (BKS)
#[derive(Debug, Clone)]
pub struct PatternHint {
    /// Context pattern describing when this hint applies
    pub context_pattern: String,
    /// The learned rule or guideline
    pub rule: String,
    /// Confidence of the source truth (0.0-1.0)
    pub confidence: f64,
    /// Source system that produced this hint (e.g. "bks", "seal")
    pub source: String,
}

/// Global memory for cross-session learning
#[derive(Debug, Default)]
pub struct GlobalMemory {
    /// Query patterns organized by question type
    pub query_patterns: HashMap<QuestionType, Vec<QueryPattern>>,
    /// Coreference resolution patterns
    pub resolution_patterns: Vec<ResolutionPattern>,
    /// Tool error patterns for learning failure modes
    pub tool_error_patterns: HashMap<String, ToolErrorPattern>,
    /// Tool execution statistics
    pub tool_stats: HashMap<String, ToolStats>,
    /// Response confidence statistics
    pub confidence_stats: ConfidenceStats,
    /// Structured hints from behavioral knowledge
    pub pattern_hints: Vec<PatternHint>,
}

impl GlobalMemory {
    /// Create new global memory
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a structured pattern hint from behavioral knowledge
    pub fn add_pattern_hint(&mut self, hint: PatternHint) {
        self.pattern_hints.push(hint);
    }

    /// Get all stored pattern hints
    pub fn get_pattern_hints(&self) -> &[PatternHint] {
        &self.pattern_hints
    }

    /// Add a query pattern
    pub fn add_pattern(&mut self, pattern: QueryPattern) {
        self.query_patterns
            .entry(pattern.question_type.clone())
            .or_default()
            .push(pattern);
    }

    /// Get patterns for a question type, sorted by reliability
    pub fn get_patterns(&self, question_type: &QuestionType) -> Vec<&QueryPattern> {
        if let Some(patterns) = self.query_patterns.get(question_type) {
            let mut sorted: Vec<_> = patterns.iter().collect();
            sorted.sort_by(|a, b| {
                b.reliability()
                    .partial_cmp(&a.reliability())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted
        } else {
            Vec::new()
        }
    }

    /// Get the best pattern for a question type and entity types
    pub fn get_best_pattern(
        &self,
        question_type: &QuestionType,
        entity_types: &[EntityType],
    ) -> Option<&QueryPattern> {
        self.get_patterns(question_type)
            .into_iter()
            .find(|p| p.matches_types(entity_types))
    }

    /// Get a pattern by ID
    pub fn get_pattern_mut(&mut self, id: &str) -> Option<&mut QueryPattern> {
        for patterns in self.query_patterns.values_mut() {
            if let Some(pattern) = patterns.iter_mut().find(|p| p.id == id) {
                return Some(pattern);
            }
        }
        None
    }

    /// Remove low-reliability patterns
    pub fn prune_patterns(&mut self, min_reliability: f32, min_uses: u32) {
        for patterns in self.query_patterns.values_mut() {
            patterns.retain(|p| {
                let total_uses = p.success_count + p.failure_count;
                total_uses < min_uses || p.reliability() >= min_reliability
            });
        }
    }

    /// Record a tool outcome for learning
    pub fn record_tool_outcome(&mut self, outcome: &ToolOutcome) {
        let stats = self
            .tool_stats
            .entry(outcome.tool_name.clone())
            .or_default();

        if outcome.success {
            stats.record_success(outcome.retries, outcome.execution_time_ms);
        } else {
            stats.record_failure(outcome.retries, outcome.execution_time_ms);

            // Also record error pattern if we have error info
            if let Some(ref error_category) = outcome.error_category {
                let key = format!("{}:{}", outcome.tool_name, error_category.category_name());

                if let Some(pattern) = self.tool_error_patterns.get_mut(&key) {
                    pattern.record_occurrence();
                } else {
                    self.tool_error_patterns.insert(
                        key,
                        ToolErrorPattern::new(&outcome.tool_name, error_category),
                    );
                }
            }
        }
    }

    /// Record a response confidence sample
    pub fn record_confidence(&mut self, confidence: &ResponseConfidence) {
        self.confidence_stats.record_sample(confidence);
    }

    /// Get common errors for a tool
    pub fn get_common_errors(&self, tool_name: &str) -> Vec<&ToolErrorPattern> {
        self.tool_error_patterns
            .values()
            .filter(|p| p.tool_name == tool_name && p.is_frequent())
            .collect()
    }

    /// Get error prevention hints for prompts
    pub fn get_error_prevention_hints(&self, tool_name: &str) -> Option<String> {
        let common_errors = self.get_common_errors(tool_name);
        if common_errors.is_empty() {
            return None;
        }

        let hints: Vec<String> = common_errors
            .iter()
            .filter_map(|e| e.suggested_fix.clone())
            .collect();

        if hints.is_empty() {
            None
        } else {
            Some(format!(
                "Common pitfalls for {}: {}",
                tool_name,
                hints.join("; ")
            ))
        }
    }

    /// Get tool reliability score
    pub fn get_tool_reliability(&self, tool_name: &str) -> Option<f64> {
        self.tool_stats.get(tool_name).map(|s| s.success_rate())
    }
}

/// Learning coordinator that manages both local and global memory
#[derive(Debug)]
pub struct LearningCoordinator {
    /// Local memory for current session
    pub local: LocalMemory,
    /// Global memory for cross-session patterns
    pub global: GlobalMemory,
    /// Learning rate for pattern updates
    _learning_rate: f32,
    /// Minimum successes before pattern is trusted
    min_successes: u32,
}

impl LearningCoordinator {
    /// Create a new learning coordinator
    pub fn new(conversation_id: String) -> Self {
        Self {
            local: LocalMemory::new(conversation_id),
            global: GlobalMemory::new(),
            _learning_rate: 0.3,
            min_successes: 3,
        }
    }

    /// Process a query through the learning system
    pub fn process_query(
        &mut self,
        _original: &str,
        _resolved: &str,
        core: Option<QueryCore>,
        turn: u32,
    ) -> Option<&QueryPattern> {
        self.local.current_turn = turn;

        if let Some(ref c) = core {
            // Get entity types from the core
            let entity_types: Vec<_> = c.entities.iter().map(|(_, t)| t.clone()).collect();

            // Check for matching pattern in global memory
            if let Some(pattern) = self
                .global
                .get_best_pattern(&c.question_type, &entity_types)
            {
                return Some(pattern);
            }
        }

        None
    }

    /// Record the outcome of a query execution
    pub fn record_outcome(
        &mut self,
        pattern_id: Option<&str>,
        success: bool,
        result_count: usize,
        query_core: Option<&QueryCore>,
        execution_time_ms: u64,
    ) {
        // Update pattern statistics if we used one
        if let Some(id) = pattern_id
            && let Some(pattern) = self.global.get_pattern_mut(id)
        {
            if success {
                pattern.record_success(result_count);
            } else {
                pattern.record_failure();
            }
        }

        // Record in local memory
        if let Some(core) = query_core {
            self.local.record_query(
                &core.original,
                core.resolved.as_deref().unwrap_or(&core.original),
                core.question_type.clone(),
                Some(core.to_sexp()),
                success,
                result_count,
                execution_time_ms,
            );

            // If successful and we don't have a pattern, create one
            if success && pattern_id.is_none() && result_count > 0 {
                let _ = self.learn_pattern(core, result_count);
            }
        }
    }

    /// Learn a new pattern from a successful query
    pub fn learn_pattern(&mut self, query: &QueryCore, result_count: usize) -> Option<String> {
        // Only learn from queries with reasonable results
        if result_count == 0 || result_count > 100 {
            return None;
        }

        // Generalize the query to a template
        let template = self.generalize_query(query);

        // Get required entity types
        let required_types: Vec<_> = query.entities.iter().map(|(_, t)| t.clone()).collect();

        // Check if we already have a similar pattern
        if let Some(existing) = self
            .global
            .get_best_pattern(&query.question_type, &required_types)
            && existing.template == template
        {
            return None; // Already have this pattern
        }

        // Create and add the new pattern
        let mut pattern = QueryPattern::new(query.question_type.clone(), template, required_types);
        pattern.record_success(result_count);

        let id = pattern.id.clone();
        self.global.add_pattern(pattern);

        Some(id)
    }

    /// Generalize a query to a template (replace specific entities with placeholders)
    fn generalize_query(&self, query: &QueryCore) -> String {
        let mut template = query.to_sexp();

        // Replace entity names with type placeholders
        for (name, entity_type) in &query.entities {
            let placeholder = format!("${{{}}}", entity_type.as_str().to_uppercase());
            template = template.replace(&format!("\"{}\"", name), &placeholder);
        }

        template
    }

    /// Get context for prompt injection
    pub fn get_context_for_prompt(&self) -> String {
        let mut context = String::new();

        // Add frequently used entities
        let frequent = self.local.get_frequent_entities(5);
        if !frequent.is_empty() {
            context.push_str("Frequently referenced entities:\n");
            for entity in frequent {
                context.push_str(&format!(
                    "- {} ({}): {} mentions\n",
                    entity.name,
                    entity.entity_type.as_str(),
                    entity.frequency()
                ));
            }
            context.push('\n');
        }

        // Add recent successful patterns
        for question_type in [
            QuestionType::Definition,
            QuestionType::Location,
            QuestionType::Dependency,
        ] {
            let patterns = self.global.get_patterns(&question_type);
            let good_patterns: Vec<_> = patterns
                .iter()
                .filter(|p| p.reliability() > 0.7 && p.success_count >= self.min_successes)
                .take(2)
                .collect();

            if !good_patterns.is_empty() {
                context.push_str(&format!("Effective {:?} patterns:\n", question_type));
                for pattern in good_patterns {
                    context.push_str(&format!(
                        "- {} ({}% reliable)\n",
                        pattern.template,
                        (pattern.reliability() * 100.0) as u32
                    ));
                }
                context.push('\n');
            }
        }

        context
    }

    /// Get all promotable patterns (high reliability, enough uses)
    ///
    /// Returns patterns that meet the criteria for promotion to BKS
    pub fn get_promotable_patterns(
        &self,
        min_reliability: f32,
        min_uses: u32,
    ) -> Vec<&QueryPattern> {
        let mut promotable = Vec::new();

        for patterns in self.global.query_patterns.values() {
            for pattern in patterns {
                let total_uses = pattern.success_count + pattern.failure_count;
                if pattern.reliability() >= min_reliability && total_uses >= min_uses {
                    promotable.push(pattern);
                }
            }
        }

        // Sort by reliability descending
        promotable.sort_by(|a, b| {
            b.reliability()
                .partial_cmp(&a.reliability())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        promotable
    }

    /// Get learning statistics
    pub fn get_stats(&self) -> LearningStats {
        let total_patterns: usize = self.global.query_patterns.values().map(|v| v.len()).sum();

        let mut total_successes = 0u32;
        let mut total_failures = 0u32;
        for patterns in self.global.query_patterns.values() {
            for pattern in patterns {
                total_successes += pattern.success_count;
                total_failures += pattern.failure_count;
            }
        }

        LearningStats {
            session_queries: self.local.query_history.len(),
            session_entities: self.local.entities.len(),
            session_coreferences: self.local.coreference_log.len(),
            global_patterns: total_patterns,
            global_successes: total_successes,
            global_failures: total_failures,
            overall_reliability: if total_successes + total_failures > 0 {
                total_successes as f32 / (total_successes + total_failures) as f32
            } else {
                0.5
            },
        }
    }

    // =====================
    // Tool Learning Methods
    // =====================

    /// Record a tool execution outcome (delegates to global memory)
    pub fn record_tool_outcome(&mut self, outcome: &ToolOutcome) {
        self.global.record_tool_outcome(outcome);
    }

    /// Record a response confidence sample (delegates to global memory)
    pub fn record_confidence(&mut self, confidence: &ResponseConfidence) {
        self.global.record_confidence(confidence);
    }

    /// Get error prevention hints for a tool (delegates to global memory)
    pub fn get_error_prevention_hints(&self, tool_name: &str) -> Option<String> {
        self.global.get_error_prevention_hints(tool_name)
    }

    /// Get tool reliability score (delegates to global memory)
    pub fn get_tool_reliability(&self, tool_name: &str) -> Option<f64> {
        self.global.get_tool_reliability(tool_name)
    }

    /// Get common errors for a tool (delegates to global memory)
    pub fn get_common_errors(&self, tool_name: &str) -> Vec<&ToolErrorPattern> {
        self.global.get_common_errors(tool_name)
    }

    /// Get the average response confidence
    pub fn get_avg_confidence(&self) -> f64 {
        self.global.confidence_stats.avg_confidence()
    }

    /// Check if responses are frequently low confidence
    pub fn has_confidence_issues(&self) -> bool {
        self.global.confidence_stats.low_confidence_ratio() > 0.3
    }
}

/// Learning statistics
#[derive(Debug, Clone)]
pub struct LearningStats {
    /// Number of queries in current session
    pub session_queries: usize,
    /// Number of entities tracked in session
    pub session_entities: usize,
    /// Number of coreferences resolved in session
    pub session_coreferences: usize,
    /// Number of patterns in global memory
    pub global_patterns: usize,
    /// Total successful pattern uses
    pub global_successes: u32,
    /// Total failed pattern uses
    pub global_failures: u32,
    /// Overall reliability score
    pub overall_reliability: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracked_entity() {
        let mut entity = TrackedEntity::new("main.rs".to_string(), EntityType::File, 1);
        assert_eq!(entity.frequency(), 1);

        entity.record_mention(2);
        entity.record_mention(3);
        assert_eq!(entity.frequency(), 3);

        // Duplicate mention should not increase frequency
        entity.record_mention(2);
        assert_eq!(entity.frequency(), 3);
    }

    #[test]
    fn test_local_memory() {
        let mut local = LocalMemory::new("test-conv".to_string());

        local.track_entity("main.rs", EntityType::File);
        local.next_turn();
        local.track_entity("config.toml", EntityType::File);
        local.track_entity("main.rs", EntityType::File); // Mention again

        assert_eq!(local.entities.len(), 2);
        assert_eq!(local.entities["main.rs"].frequency(), 2);

        // Focus stack should have config.toml first (most recent)
        assert_eq!(local.focus_stack[0], "main.rs"); // Re-mentioned
    }

    #[test]
    fn test_query_pattern_reliability() {
        let mut pattern =
            QueryPattern::new(QuestionType::Definition, "template".to_string(), vec![]);

        assert_eq!(pattern.reliability(), 0.5); // No data

        pattern.record_success(5);
        pattern.record_success(3);
        pattern.record_failure();

        // 2 successes, 1 failure = 2/3 reliability
        assert!((pattern.reliability() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_global_memory_patterns() {
        let mut global = GlobalMemory::new();

        let mut pattern1 =
            QueryPattern::new(QuestionType::Definition, "template1".to_string(), vec![]);
        pattern1.record_success(5);
        pattern1.record_success(5);

        let mut pattern2 =
            QueryPattern::new(QuestionType::Definition, "template2".to_string(), vec![]);
        pattern2.record_failure();

        global.add_pattern(pattern1);
        global.add_pattern(pattern2);

        // Get patterns should return them sorted by reliability
        let patterns = global.get_patterns(&QuestionType::Definition);
        assert_eq!(patterns.len(), 2);
        assert!(patterns[0].reliability() > patterns[1].reliability());
    }

    #[test]
    fn test_learning_coordinator() {
        let mut coordinator = LearningCoordinator::new("test-conv".to_string());

        // Record some queries
        let core = QueryCore::new(
            QuestionType::Definition,
            crate::query_core::QueryExpr::var("x"),
            vec![("main.rs".to_string(), EntityType::File)],
            "What is main.rs?".to_string(),
        );

        coordinator.record_outcome(None, true, 1, Some(&core), 0);

        let stats = coordinator.get_stats();
        assert_eq!(stats.session_queries, 1);
        assert_eq!(stats.global_patterns, 1); // Should have learned a pattern
    }

    #[test]
    fn test_pattern_matching() {
        let pattern = QueryPattern::new(
            QuestionType::Definition,
            "template".to_string(),
            vec![EntityType::File],
        );

        assert!(pattern.matches_types(&[EntityType::File]));
        assert!(pattern.matches_types(&[EntityType::File, EntityType::Function]));
        assert!(!pattern.matches_types(&[EntityType::Function]));
    }

    #[test]
    fn test_prune_patterns() {
        let mut global = GlobalMemory::new();

        let mut good_pattern =
            QueryPattern::new(QuestionType::Definition, "good".to_string(), vec![]);
        for _ in 0..10 {
            good_pattern.record_success(5);
        }

        let mut bad_pattern =
            QueryPattern::new(QuestionType::Definition, "bad".to_string(), vec![]);
        for _ in 0..10 {
            bad_pattern.record_failure();
        }

        global.add_pattern(good_pattern);
        global.add_pattern(bad_pattern);

        assert_eq!(global.get_patterns(&QuestionType::Definition).len(), 2);

        global.prune_patterns(0.5, 5);

        // Bad pattern should be removed
        assert_eq!(global.get_patterns(&QuestionType::Definition).len(), 1);
    }

    #[test]
    fn test_get_context_for_prompt() {
        let mut coordinator = LearningCoordinator::new("test".to_string());

        coordinator.local.track_entity("main.rs", EntityType::File);
        coordinator.local.track_entity("main.rs", EntityType::File);
        coordinator
            .local
            .track_entity("config.toml", EntityType::File);

        let context = coordinator.get_context_for_prompt();

        // Should mention main.rs as frequently referenced
        assert!(context.contains("main.rs") || context.contains("Frequently"));
    }
}
