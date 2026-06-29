//! Local cache for behavioral truths with SQLite persistence
//!
//! Maintains a local copy of truths synced from the server, with offline
//! queue support for when the server is unavailable.

use super::truth::{BehavioralTruth, PendingTruthSubmission, TruthCategory, TruthFeedback};
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Local cache of behavioral truths synced from server
pub struct BehavioralKnowledgeCache {
    /// SQLite connection
    conn: Arc<Mutex<Connection>>,

    /// In-memory cache for fast access
    truths: HashMap<String, BehavioralTruth>,

    /// Timestamp of last successful sync with server
    pub last_sync: i64,

    /// Queue of truths waiting to be submitted to server
    pending_submissions: Vec<PendingTruthSubmission>,

    /// Queue of feedback waiting to be sent to server
    pending_feedback: Vec<TruthFeedback>,

    /// Maximum size of offline queue
    max_queue_size: usize,
}

impl BehavioralKnowledgeCache {
    /// Create a new cache with SQLite persistence
    pub fn new<P: AsRef<Path>>(db_path: P, max_queue_size: usize) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        Self::init_schema(&conn)?;

        let mut cache = Self {
            conn: Arc::new(Mutex::new(conn)),
            truths: HashMap::new(),
            last_sync: 0,
            pending_submissions: Vec::new(),
            pending_feedback: Vec::new(),
            max_queue_size,
        };

        // Load existing data from database
        cache.load_from_db()?;

        Ok(cache)
    }

    /// Create an in-memory cache (for testing)
    pub fn in_memory(max_queue_size: usize) -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            truths: HashMap::new(),
            last_sync: 0,
            pending_submissions: Vec::new(),
            pending_feedback: Vec::new(),
            max_queue_size,
        })
    }

    /// Initialize database schema
    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS truths (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                context_pattern TEXT NOT NULL,
                rule TEXT NOT NULL,
                rationale TEXT NOT NULL,
                confidence REAL NOT NULL,
                reinforcements INTEGER NOT NULL DEFAULT 0,
                contradictions INTEGER NOT NULL DEFAULT 0,
                last_used INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                created_by TEXT,
                source TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                deleted INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_truths_context ON truths(context_pattern);
            CREATE INDEX IF NOT EXISTS idx_truths_category ON truths(category);
            CREATE INDEX IF NOT EXISTS idx_truths_confidence ON truths(confidence);

            CREATE TABLE IF NOT EXISTS pending_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                truth_json TEXT NOT NULL,
                queued_at INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            );

            CREATE TABLE IF NOT EXISTS pending_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                truth_id TEXT NOT NULL,
                is_reinforcement INTEGER NOT NULL,
                context TEXT,
                timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sync_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;

        Ok(())
    }

    /// Load truths and state from database
    fn load_from_db(&mut self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .expect("knowledge cache connection lock poisoned");

        // Load last sync timestamp
        self.last_sync = conn
            .query_row(
                "SELECT value FROM sync_state WHERE key = 'last_sync'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // Load truths
        let mut stmt = conn.prepare(
            "SELECT id, category, context_pattern, rule, rationale, confidence,
                    reinforcements, contradictions, last_used, created_at,
                    created_by, source, version, deleted
             FROM truths WHERE deleted = 0",
        )?;

        let truths = stmt.query_map([], |row| {
            Ok(BehavioralTruth {
                id: row.get(0)?,
                category: serde_json::from_str(&format!("\"{}\"", row.get::<_, String>(1)?))
                    .unwrap_or(TruthCategory::CommandUsage),
                context_pattern: row.get(2)?,
                rule: row.get(3)?,
                rationale: row.get(4)?,
                confidence: row.get(5)?,
                reinforcements: row.get(6)?,
                contradictions: row.get(7)?,
                last_used: row.get(8)?,
                created_at: row.get(9)?,
                created_by: row.get(10)?,
                source: serde_json::from_str(&format!("\"{}\"", row.get::<_, String>(11)?))
                    .unwrap_or(super::truth::TruthSource::ExplicitCommand),
                version: row.get::<_, i64>(12)? as u64,
                deleted: row.get::<_, i32>(13)? != 0,
            })
        })?;

        for truth in truths {
            let truth = truth?;
            self.truths.insert(truth.id.clone(), truth);
        }

        // Load pending submissions
        let mut stmt = conn.prepare(
            "SELECT truth_json, queued_at, attempts, last_error FROM pending_submissions",
        )?;

        let submissions = stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            let truth: BehavioralTruth = serde_json::from_str(&json).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            Ok(PendingTruthSubmission {
                truth,
                queued_at: row.get(1)?,
                attempts: row.get(2)?,
                last_error: row.get(3)?,
            })
        })?;

        for submission in submissions {
            self.pending_submissions.push(submission?);
        }

        // Load pending feedback
        let mut stmt = conn.prepare(
            "SELECT truth_id, is_reinforcement, context, timestamp FROM pending_feedback",
        )?;

        let feedback = stmt.query_map([], |row| {
            Ok(TruthFeedback {
                truth_id: row.get(0)?,
                is_reinforcement: row.get::<_, i32>(1)? != 0,
                context: row.get(2)?,
                timestamp: row.get(3)?,
            })
        })?;

        for fb in feedback {
            self.pending_feedback.push(fb?);
        }

        Ok(())
    }

    /// Save a truth to the database
    fn save_truth_to_db(&self, truth: &BehavioralTruth) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .expect("knowledge cache connection lock poisoned");
        let category = serde_json::to_string(&truth.category)?
            .trim_matches('"')
            .to_string();
        let source = serde_json::to_string(&truth.source)?
            .trim_matches('"')
            .to_string();

        conn.execute(
            r#"INSERT OR REPLACE INTO truths
               (id, category, context_pattern, rule, rationale, confidence,
                reinforcements, contradictions, last_used, created_at,
                created_by, source, version, deleted)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)"#,
            params![
                truth.id,
                category,
                truth.context_pattern,
                truth.rule,
                truth.rationale,
                truth.confidence,
                truth.reinforcements,
                truth.contradictions,
                truth.last_used,
                truth.created_at,
                truth.created_by,
                source,
                truth.version as i64,
                truth.deleted as i32,
            ],
        )?;

        Ok(())
    }

    /// Update last sync timestamp
    pub fn set_last_sync(&mut self, timestamp: i64) -> Result<()> {
        self.last_sync = timestamp;
        let conn = self
            .conn
            .lock()
            .expect("knowledge cache connection lock poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO sync_state (key, value) VALUES ('last_sync', ?1)",
            params![timestamp.to_string()],
        )?;
        Ok(())
    }

    /// Add a new truth to the cache
    pub fn add_truth(&mut self, truth: BehavioralTruth) -> Result<()> {
        self.save_truth_to_db(&truth)?;
        self.truths.insert(truth.id.clone(), truth);
        Ok(())
    }

    /// Update an existing truth
    pub fn update_truth(&mut self, truth: BehavioralTruth) -> Result<()> {
        self.save_truth_to_db(&truth)?;
        self.truths.insert(truth.id.clone(), truth);
        Ok(())
    }

    /// Get a truth by ID
    pub fn get_truth(&self, id: &str) -> Option<&BehavioralTruth> {
        self.truths.get(id)
    }

    /// Get a mutable reference to a truth by ID
    pub fn get_truth_mut(&mut self, id: &str) -> Option<&mut BehavioralTruth> {
        self.truths.get_mut(id)
    }

    /// Remove a truth (soft delete)
    pub fn remove_truth(&mut self, id: &str) -> Result<bool> {
        if let Some(truth) = self.truths.get_mut(id) {
            truth.delete();
        } else {
            return Ok(false);
        }

        // Save after releasing the mutable borrow
        if let Some(truth) = self.truths.get(id) {
            self.save_truth_to_db(truth)?;
        }
        Ok(true)
    }

    /// Get all active truths
    pub fn all_truths(&self) -> impl Iterator<Item = &BehavioralTruth> {
        self.truths.values().filter(|t| !t.deleted)
    }

    /// Get truths by category
    pub fn truths_by_category(&self, category: TruthCategory) -> Vec<&BehavioralTruth> {
        self.truths
            .values()
            .filter(|t| !t.deleted && t.category == category)
            .collect()
    }

    /// Get truths matching a context pattern (simple substring match)
    pub fn get_matching_truths(&self, context: &str) -> Vec<&BehavioralTruth> {
        let context_lower = context.to_lowercase();
        self.truths
            .values()
            .filter(|t| {
                !t.deleted
                    && t.context_pattern
                        .to_lowercase()
                        .split_whitespace()
                        .any(|word| context_lower.contains(word))
            })
            .collect()
    }

    /// Get truths matching a context pattern with relevance scores
    /// Returns (truth, score) tuples sorted by relevance
    pub fn get_matching_truths_with_scores(
        &self,
        context: &str,
        min_confidence: f32,
        limit: usize,
    ) -> Result<Vec<(&BehavioralTruth, f32)>> {
        let context_lower = context.to_lowercase();
        let context_words: Vec<&str> = context_lower.split_whitespace().collect();

        let mut matches: Vec<(&BehavioralTruth, f32)> = self
            .truths
            .values()
            .filter(|t| !t.deleted && t.confidence >= min_confidence)
            .filter_map(|truth| {
                // Calculate relevance score based on word overlap
                let pattern_lower = truth.context_pattern.to_lowercase();
                let pattern_words: Vec<&str> = pattern_lower.split_whitespace().collect();

                let mut score = 0.0f32;
                for pattern_word in &pattern_words {
                    for context_word in &context_words {
                        if context_word.contains(pattern_word)
                            || pattern_word.contains(context_word)
                        {
                            score += 1.0;
                        }
                    }
                }

                if score > 0.0 {
                    // Normalize by pattern length and boost by confidence
                    let normalized_score = (score / pattern_words.len() as f32) * truth.confidence;
                    Some((truth, normalized_score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        matches.truncate(limit);

        Ok(matches)
    }

    /// Get truths above a confidence threshold
    pub fn get_reliable_truths(
        &self,
        min_confidence: f32,
        decay_days: u32,
    ) -> Vec<&BehavioralTruth> {
        self.truths
            .values()
            .filter(|t| !t.deleted && t.is_reliable(min_confidence, decay_days))
            .collect()
    }

    /// Queue a truth for submission to server
    pub fn queue_submission(&mut self, truth: BehavioralTruth) -> Result<bool> {
        if self.pending_submissions.len() >= self.max_queue_size {
            return Ok(false);
        }

        let submission = PendingTruthSubmission::new(truth);
        let json = serde_json::to_string(&submission.truth)?;

        let conn = self
            .conn
            .lock()
            .expect("knowledge cache connection lock poisoned");
        conn.execute(
            "INSERT INTO pending_submissions (truth_json, queued_at, attempts) VALUES (?1, ?2, ?3)",
            params![json, submission.queued_at, submission.attempts],
        )?;

        self.pending_submissions.push(submission);
        Ok(true)
    }

    /// Get pending submissions
    pub fn pending_submissions(&self) -> &[PendingTruthSubmission] {
        &self.pending_submissions
    }

    /// Clear all pending submissions (after successful sync)
    pub fn clear_pending_submissions(&mut self) -> Result<()> {
        self.pending_submissions.clear();
        let conn = self
            .conn
            .lock()
            .expect("knowledge cache connection lock poisoned");
        conn.execute("DELETE FROM pending_submissions", [])?;
        Ok(())
    }

    /// Queue feedback for sending to server
    pub fn queue_feedback(&mut self, feedback: TruthFeedback) -> Result<bool> {
        if self.pending_feedback.len() >= self.max_queue_size {
            return Ok(false);
        }

        let conn = self
            .conn
            .lock()
            .expect("knowledge cache connection lock poisoned");
        conn.execute(
            "INSERT INTO pending_feedback (truth_id, is_reinforcement, context, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                feedback.truth_id,
                feedback.is_reinforcement as i32,
                feedback.context,
                feedback.timestamp,
            ],
        )?;

        self.pending_feedback.push(feedback);
        Ok(true)
    }

    /// Get pending feedback
    pub fn pending_feedback(&self) -> &[TruthFeedback] {
        &self.pending_feedback
    }

    /// Clear all pending feedback (after successful sync)
    pub fn clear_pending_feedback(&mut self) -> Result<()> {
        self.pending_feedback.clear();
        let conn = self
            .conn
            .lock()
            .expect("knowledge cache connection lock poisoned");
        conn.execute("DELETE FROM pending_feedback", [])?;
        Ok(())
    }

    /// Merge truths from server (handles version conflicts)
    pub fn merge_from_server(
        &mut self,
        server_truths: Vec<BehavioralTruth>,
    ) -> Result<MergeResult> {
        let mut added = 0;
        let mut updated = 0;
        let mut conflicts = 0;

        for server_truth in server_truths {
            if let Some(local_truth) = self.truths.get(&server_truth.id) {
                // Check for version conflict
                if server_truth.version > local_truth.version {
                    // Server wins - update local
                    self.save_truth_to_db(&server_truth)?;
                    self.truths.insert(server_truth.id.clone(), server_truth);
                    updated += 1;
                } else if server_truth.version < local_truth.version {
                    // Local is newer - conflict (should be rare)
                    conflicts += 1;
                }
                // Equal versions - no action needed
            } else {
                // New truth from server
                self.save_truth_to_db(&server_truth)?;
                self.truths.insert(server_truth.id.clone(), server_truth);
                added += 1;
            }
        }

        Ok(MergeResult {
            added,
            updated,
            conflicts,
        })
    }

    /// Apply decay to all truths
    pub fn apply_decay(&mut self, decay_start_days: u32) -> Result<u32> {
        let mut decayed = 0;

        for truth in self.truths.values_mut() {
            let old_confidence = truth.confidence;
            truth.apply_decay(decay_start_days);
            if (truth.confidence - old_confidence).abs() > 0.001 {
                decayed += 1;
            }
        }

        // Save decayed truths to database
        if decayed > 0 {
            for truth in self.truths.values() {
                self.save_truth_to_db(truth)?;
            }
        }

        Ok(decayed)
    }

    /// Get statistics about the cache
    pub fn stats(&self) -> CacheStats {
        let mut by_category: HashMap<TruthCategory, u32> = HashMap::new();
        let mut total_confidence = 0.0f32;
        let mut count = 0u32;

        for truth in self.truths.values().filter(|t| !t.deleted) {
            *by_category.entry(truth.category).or_insert(0) += 1;
            total_confidence += truth.confidence;
            count += 1;
        }

        CacheStats {
            total_truths: count,
            by_category,
            avg_confidence: if count > 0 {
                total_confidence / count as f32
            } else {
                0.0
            },
            pending_submissions: self.pending_submissions.len(),
            pending_feedback: self.pending_feedback.len(),
            last_sync: self.last_sync,
        }
    }
}

/// Result of merging truths from server
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Number of new truths added.
    pub added: u32,
    /// Number of existing truths updated.
    pub updated: u32,
    /// Number of merge conflicts.
    pub conflicts: u32,
}

/// Statistics about the cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of cached truths.
    pub total_truths: u32,
    /// Counts by category.
    pub by_category: HashMap<TruthCategory, u32>,
    /// Average confidence score.
    pub avg_confidence: f32,
    /// Number of pending truth submissions.
    pub pending_submissions: usize,
    /// Number of pending feedback reports.
    pub pending_feedback: usize,
    /// Unix timestamp of last sync.
    pub last_sync: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::bks_pks::truth::TruthSource;

    fn create_test_truth(context: &str, rule: &str) -> BehavioralTruth {
        BehavioralTruth::new(
            TruthCategory::CommandUsage,
            context.to_string(),
            rule.to_string(),
            "Test rationale".to_string(),
            TruthSource::ExplicitCommand,
            None,
        )
    }

    #[test]
    fn test_cache_creation() {
        let cache = BehavioralKnowledgeCache::in_memory(100).unwrap();
        assert_eq!(cache.last_sync, 0);
        assert_eq!(cache.all_truths().count(), 0);
    }

    #[test]
    fn test_add_and_get_truth() {
        let mut cache = BehavioralKnowledgeCache::in_memory(100).unwrap();
        let truth = create_test_truth("pm2 logs", "Use --nostream");

        let id = truth.id.clone();
        cache.add_truth(truth).unwrap();

        let retrieved = cache.get_truth(&id).unwrap();
        assert_eq!(retrieved.rule, "Use --nostream");
    }

    #[test]
    fn test_matching_truths() {
        let mut cache = BehavioralKnowledgeCache::in_memory(100).unwrap();

        cache
            .add_truth(create_test_truth("pm2 logs", "Use --nostream"))
            .unwrap();
        cache
            .add_truth(create_test_truth("cargo build", "Use cargo-watch"))
            .unwrap();

        let matches = cache.get_matching_truths("pm2 logs myapp");
        assert_eq!(matches.len(), 1);
        assert!(matches[0].rule.contains("--nostream"));
    }

    #[test]
    fn test_truths_by_category() {
        let mut cache = BehavioralKnowledgeCache::in_memory(100).unwrap();

        cache
            .add_truth(create_test_truth("test1", "rule1"))
            .unwrap();

        let mut task_truth = create_test_truth("test2", "rule2");
        task_truth.category = TruthCategory::TaskStrategy;
        cache.add_truth(task_truth).unwrap();

        let cmd_truths = cache.truths_by_category(TruthCategory::CommandUsage);
        assert_eq!(cmd_truths.len(), 1);

        let task_truths = cache.truths_by_category(TruthCategory::TaskStrategy);
        assert_eq!(task_truths.len(), 1);
    }

    #[test]
    fn test_queue_submission() {
        let mut cache = BehavioralKnowledgeCache::in_memory(2).unwrap();

        let truth1 = create_test_truth("test1", "rule1");
        let truth2 = create_test_truth("test2", "rule2");
        let truth3 = create_test_truth("test3", "rule3");

        assert!(cache.queue_submission(truth1).unwrap());
        assert!(cache.queue_submission(truth2).unwrap());
        assert!(!cache.queue_submission(truth3).unwrap()); // Queue full

        assert_eq!(cache.pending_submissions().len(), 2);
    }

    #[test]
    fn test_merge_from_server() {
        let mut cache = BehavioralKnowledgeCache::in_memory(100).unwrap();

        // Add local truth
        let mut local = create_test_truth("local", "local rule");
        local.version = 1;
        let local_id = local.id.clone();
        cache.add_truth(local).unwrap();

        // Create server truths
        let new_truth = create_test_truth("new", "new rule");

        let mut updated = create_test_truth("local", "updated rule");
        updated.id = local_id.clone();
        updated.version = 2;

        let result = cache.merge_from_server(vec![new_truth, updated]).unwrap();

        assert_eq!(result.added, 1);
        assert_eq!(result.updated, 1);
        assert_eq!(result.conflicts, 0);

        // Verify update applied
        let truth = cache.get_truth(&local_id).unwrap();
        assert_eq!(truth.rule, "updated rule");
    }

    #[test]
    fn test_stats() {
        let mut cache = BehavioralKnowledgeCache::in_memory(100).unwrap();

        cache
            .add_truth(create_test_truth("test1", "rule1"))
            .unwrap();
        cache
            .add_truth(create_test_truth("test2", "rule2"))
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.total_truths, 2);
        assert_eq!(
            *stats.by_category.get(&TruthCategory::CommandUsage).unwrap(),
            2
        );
        assert!(stats.avg_confidence > 0.0);
    }
}
