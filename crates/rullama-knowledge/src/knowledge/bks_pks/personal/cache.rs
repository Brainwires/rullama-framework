//! Local cache for personal facts with SQLite persistence
//!
//! Maintains a local copy of personal facts synced from the server, with offline
//! queue support for when the server is unavailable.

use super::fact::{
    PendingFactSubmission, PersonalFact, PersonalFactCategory, PersonalFactFeedback,
};
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Local cache of personal facts synced from server
pub struct PersonalKnowledgeCache {
    /// SQLite connection
    conn: Arc<Mutex<Connection>>,

    /// In-memory cache for fast access
    facts: HashMap<String, PersonalFact>,

    /// Index by key for fast lookup
    facts_by_key: HashMap<String, String>, // key -> id

    /// Timestamp of last successful sync with server
    pub last_sync: i64,

    /// Queue of facts waiting to be submitted to server
    pending_submissions: Vec<PendingFactSubmission>,

    /// Queue of feedback waiting to be sent to server
    pending_feedback: Vec<PersonalFactFeedback>,

    /// Maximum size of offline queue
    max_queue_size: usize,
}

impl PersonalKnowledgeCache {
    /// Create a new cache with SQLite persistence
    pub fn new<P: AsRef<Path>>(db_path: P, max_queue_size: usize) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        Self::init_schema(&conn)?;

        let mut cache = Self {
            conn: Arc::new(Mutex::new(conn)),
            facts: HashMap::new(),
            facts_by_key: HashMap::new(),
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
            facts: HashMap::new(),
            facts_by_key: HashMap::new(),
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
            CREATE TABLE IF NOT EXISTS personal_facts (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                context TEXT,
                confidence REAL NOT NULL,
                reinforcements INTEGER NOT NULL DEFAULT 0,
                contradictions INTEGER NOT NULL DEFAULT 0,
                last_used INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                source TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                deleted INTEGER NOT NULL DEFAULT 0,
                local_only INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_personal_facts_key ON personal_facts(key);
            CREATE INDEX IF NOT EXISTS idx_personal_facts_category ON personal_facts(category);
            CREATE INDEX IF NOT EXISTS idx_personal_facts_confidence ON personal_facts(confidence);

            CREATE TABLE IF NOT EXISTS pending_fact_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_json TEXT NOT NULL,
                queued_at INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            );

            CREATE TABLE IF NOT EXISTS pending_fact_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id TEXT NOT NULL,
                is_reinforcement INTEGER NOT NULL,
                context TEXT,
                timestamp INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS personal_sync_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;

        Ok(())
    }

    /// Load facts and state from database
    fn load_from_db(&mut self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .expect("personal knowledge cache connection lock poisoned");

        // Load last sync timestamp
        self.last_sync = conn
            .query_row(
                "SELECT value FROM personal_sync_state WHERE key = 'last_sync'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // Load facts
        let mut stmt = conn.prepare(
            "SELECT id, category, key, value, context, confidence,
                    reinforcements, contradictions, last_used, created_at,
                    updated_at, source, version, deleted, local_only
             FROM personal_facts WHERE deleted = 0",
        )?;

        let facts = stmt.query_map([], |row| {
            Ok(PersonalFact {
                id: row.get(0)?,
                category: serde_json::from_str(&format!("\"{}\"", row.get::<_, String>(1)?))
                    .unwrap_or(PersonalFactCategory::Preference),
                key: row.get(2)?,
                value: row.get(3)?,
                context: row.get(4)?,
                confidence: row.get(5)?,
                reinforcements: row.get(6)?,
                contradictions: row.get(7)?,
                last_used: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
                source: serde_json::from_str(&format!("\"{}\"", row.get::<_, String>(11)?))
                    .unwrap_or(super::fact::PersonalFactSource::ExplicitStatement),
                version: row.get::<_, i64>(12)? as u64,
                deleted: row.get::<_, i32>(13)? != 0,
                local_only: row.get::<_, i32>(14)? != 0,
            })
        })?;

        for fact in facts {
            let fact = fact?;
            self.facts_by_key.insert(fact.key.clone(), fact.id.clone());
            self.facts.insert(fact.id.clone(), fact);
        }

        // Load pending submissions
        let mut stmt = conn.prepare(
            "SELECT fact_json, queued_at, attempts, last_error FROM pending_fact_submissions",
        )?;

        let submissions = stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            let fact: PersonalFact = serde_json::from_str(&json).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            Ok(PendingFactSubmission {
                fact,
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
            "SELECT fact_id, is_reinforcement, context, timestamp FROM pending_fact_feedback",
        )?;

        let feedback = stmt.query_map([], |row| {
            Ok(PersonalFactFeedback {
                fact_id: row.get(0)?,
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

    /// Save a fact to the database
    fn save_fact_to_db(&self, fact: &PersonalFact) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .expect("personal knowledge cache connection lock poisoned");
        let category = serde_json::to_string(&fact.category)?
            .trim_matches('"')
            .to_string();
        let source = serde_json::to_string(&fact.source)?
            .trim_matches('"')
            .to_string();

        conn.execute(
            r#"INSERT OR REPLACE INTO personal_facts
               (id, category, key, value, context, confidence,
                reinforcements, contradictions, last_used, created_at,
                updated_at, source, version, deleted, local_only)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)"#,
            params![
                fact.id,
                category,
                fact.key,
                fact.value,
                fact.context,
                fact.confidence,
                fact.reinforcements,
                fact.contradictions,
                fact.last_used,
                fact.created_at,
                fact.updated_at,
                source,
                fact.version as i64,
                fact.deleted as i32,
                fact.local_only as i32,
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
            .expect("personal knowledge cache connection lock poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO personal_sync_state (key, value) VALUES ('last_sync', ?1)",
            params![timestamp.to_string()],
        )?;
        Ok(())
    }

    /// Add or update a fact (upsert by key)
    pub fn upsert_fact(&mut self, mut fact: PersonalFact) -> Result<()> {
        // Check if we already have a fact with this key
        if let Some(existing_id) = self.facts_by_key.get(&fact.key)
            && let Some(existing) = self.facts.get(existing_id)
        {
            // Update existing fact
            fact.id = existing.id.clone();
            fact.reinforcements = existing.reinforcements + 1;
            fact.confidence = fact.confidence.max(existing.confidence);
        }

        self.save_fact_to_db(&fact)?;
        self.facts_by_key.insert(fact.key.clone(), fact.id.clone());
        self.facts.insert(fact.id.clone(), fact);
        Ok(())
    }

    /// Add a new fact to the cache
    pub fn add_fact(&mut self, fact: PersonalFact) -> Result<()> {
        self.save_fact_to_db(&fact)?;
        self.facts_by_key.insert(fact.key.clone(), fact.id.clone());
        self.facts.insert(fact.id.clone(), fact);
        Ok(())
    }

    /// Add or update a fact with simplified interface
    pub fn upsert_fact_simple(
        &mut self,
        key: &str,
        value: &str,
        _confidence: f32,
        local_only: bool,
    ) -> Result<()> {
        use super::fact::{PersonalFactCategory, PersonalFactSource};

        let fact = PersonalFact::new(
            PersonalFactCategory::Context,
            key.to_string(),
            value.to_string(),
            None,
            PersonalFactSource::SystemObserved,
            local_only,
        );

        self.upsert_fact(fact)
    }

    /// Get all non-deleted facts
    pub fn get_all_facts(&self) -> Vec<&PersonalFact> {
        self.facts.values().filter(|f| !f.deleted).collect()
    }

    /// Get facts by key prefix (e.g., "recent_entity:" gets all recent entity facts)
    pub fn get_facts_by_key_prefix(&self, prefix: &str) -> Result<Vec<&PersonalFact>> {
        Ok(self
            .facts
            .values()
            .filter(|f| !f.deleted && f.key.starts_with(prefix))
            .collect())
    }

    /// Update an existing fact
    pub fn update_fact(&mut self, fact: PersonalFact) -> Result<()> {
        self.save_fact_to_db(&fact)?;
        self.facts_by_key.insert(fact.key.clone(), fact.id.clone());
        self.facts.insert(fact.id.clone(), fact);
        Ok(())
    }

    /// Get a fact by ID
    pub fn get_fact(&self, id: &str) -> Option<&PersonalFact> {
        self.facts.get(id)
    }

    /// Get a fact by key
    pub fn get_fact_by_key(&self, key: &str) -> Option<&PersonalFact> {
        self.facts_by_key.get(key).and_then(|id| self.facts.get(id))
    }

    /// Get a mutable reference to a fact by ID
    pub fn get_fact_mut(&mut self, id: &str) -> Option<&mut PersonalFact> {
        self.facts.get_mut(id)
    }

    /// Remove a fact (soft delete)
    pub fn remove_fact(&mut self, id: &str) -> Result<bool> {
        // First check if fact exists and get key
        let key_to_remove = {
            if let Some(fact) = self.facts.get_mut(id) {
                fact.delete();
                Some(fact.key.clone())
            } else {
                None
            }
        };

        // Now save and remove from index
        if let Some(key) = key_to_remove {
            if let Some(fact) = self.facts.get(id) {
                self.save_fact_to_db(fact)?;
            }
            self.facts_by_key.remove(&key);
            return Ok(true);
        }
        Ok(false)
    }

    /// Remove a fact by key (soft delete)
    pub fn remove_fact_by_key(&mut self, key: &str) -> Result<bool> {
        if let Some(id) = self.facts_by_key.get(key).cloned() {
            return self.remove_fact(&id);
        }
        Ok(false)
    }

    /// Get all active facts
    pub fn all_facts(&self) -> impl Iterator<Item = &PersonalFact> {
        self.facts.values().filter(|f| !f.deleted)
    }

    /// Get facts by category
    pub fn facts_by_category(&self, category: PersonalFactCategory) -> Vec<&PersonalFact> {
        self.facts
            .values()
            .filter(|f| !f.deleted && f.category == category)
            .collect()
    }

    /// Get facts matching a search query (simple substring match)
    pub fn search_facts(&self, query: &str) -> Vec<&PersonalFact> {
        let query_lower = query.to_lowercase();
        self.facts
            .values()
            .filter(|f| {
                !f.deleted
                    && (f.key.to_lowercase().contains(&query_lower)
                        || f.value.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Get facts above a confidence threshold
    pub fn get_reliable_facts(&self, min_confidence: f32) -> Vec<&PersonalFact> {
        self.facts
            .values()
            .filter(|f| !f.deleted && f.is_reliable(min_confidence))
            .collect()
    }

    /// Get facts that should be synced to server (not local_only)
    pub fn get_syncable_facts(&self) -> Vec<&PersonalFact> {
        self.facts
            .values()
            .filter(|f| !f.deleted && !f.local_only)
            .collect()
    }

    /// Queue a fact for submission to server
    pub fn queue_submission(&mut self, fact: PersonalFact) -> Result<bool> {
        if fact.local_only {
            return Ok(false); // Never sync local-only facts
        }

        if self.pending_submissions.len() >= self.max_queue_size {
            return Ok(false);
        }

        let submission = PendingFactSubmission::new(fact);
        let json = serde_json::to_string(&submission.fact)?;

        let conn = self
            .conn
            .lock()
            .expect("personal knowledge cache connection lock poisoned");
        conn.execute(
            "INSERT INTO pending_fact_submissions (fact_json, queued_at, attempts) VALUES (?1, ?2, ?3)",
            params![json, submission.queued_at, submission.attempts],
        )?;

        self.pending_submissions.push(submission);
        Ok(true)
    }

    /// Get pending submissions
    pub fn pending_submissions(&self) -> &[PendingFactSubmission] {
        &self.pending_submissions
    }

    /// Clear all pending submissions (after successful sync)
    pub fn clear_pending_submissions(&mut self) -> Result<()> {
        self.pending_submissions.clear();
        let conn = self
            .conn
            .lock()
            .expect("personal knowledge cache connection lock poisoned");
        conn.execute("DELETE FROM pending_fact_submissions", [])?;
        Ok(())
    }

    /// Queue feedback for sending to server
    pub fn queue_feedback(&mut self, feedback: PersonalFactFeedback) -> Result<bool> {
        // Check if the fact is local-only
        if let Some(fact) = self.facts.get(&feedback.fact_id)
            && fact.local_only
        {
            return Ok(false); // Don't sync feedback for local-only facts
        }

        if self.pending_feedback.len() >= self.max_queue_size {
            return Ok(false);
        }

        let conn = self
            .conn
            .lock()
            .expect("personal knowledge cache connection lock poisoned");
        conn.execute(
            "INSERT INTO pending_fact_feedback (fact_id, is_reinforcement, context, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                feedback.fact_id,
                feedback.is_reinforcement as i32,
                feedback.context,
                feedback.timestamp,
            ],
        )?;

        self.pending_feedback.push(feedback);
        Ok(true)
    }

    /// Get pending feedback
    pub fn pending_feedback(&self) -> &[PersonalFactFeedback] {
        &self.pending_feedback
    }

    /// Clear all pending feedback (after successful sync)
    pub fn clear_pending_feedback(&mut self) -> Result<()> {
        self.pending_feedback.clear();
        let conn = self
            .conn
            .lock()
            .expect("personal knowledge cache connection lock poisoned");
        conn.execute("DELETE FROM pending_fact_feedback", [])?;
        Ok(())
    }

    /// Merge facts from server (handles version conflicts)
    pub fn merge_from_server(&mut self, server_facts: Vec<PersonalFact>) -> Result<MergeResult> {
        let mut added = 0;
        let mut updated = 0;
        let mut conflicts = 0;

        for server_fact in server_facts {
            // Skip local-only facts that somehow got synced
            if server_fact.local_only {
                continue;
            }

            if let Some(local_fact) = self.facts.get(&server_fact.id) {
                // Check for version conflict
                if server_fact.version > local_fact.version {
                    // Server wins - update local
                    self.save_fact_to_db(&server_fact)?;
                    self.facts_by_key
                        .insert(server_fact.key.clone(), server_fact.id.clone());
                    self.facts.insert(server_fact.id.clone(), server_fact);
                    updated += 1;
                } else if server_fact.version < local_fact.version {
                    // Local is newer - conflict (should be rare)
                    conflicts += 1;
                }
                // Equal versions - no action needed
            } else {
                // New fact from server
                self.save_fact_to_db(&server_fact)?;
                self.facts_by_key
                    .insert(server_fact.key.clone(), server_fact.id.clone());
                self.facts.insert(server_fact.id.clone(), server_fact);
                added += 1;
            }
        }

        Ok(MergeResult {
            added,
            updated,
            conflicts,
        })
    }

    /// Apply decay to all facts based on category
    pub fn apply_decay(&mut self) -> Result<u32> {
        let mut decayed = 0;

        for fact in self.facts.values_mut() {
            let old_confidence = fact.confidence;
            fact.apply_decay();
            if (fact.confidence - old_confidence).abs() > 0.001 {
                decayed += 1;
            }
        }

        // Save decayed facts to database
        if decayed > 0 {
            for fact in self.facts.values() {
                self.save_fact_to_db(fact)?;
            }
        }

        Ok(decayed)
    }

    /// Get statistics about the cache
    pub fn stats(&self) -> CacheStats {
        let mut by_category: HashMap<PersonalFactCategory, u32> = HashMap::new();
        let mut total_confidence = 0.0f32;
        let mut count = 0u32;
        let mut local_only_count = 0u32;

        for fact in self.facts.values().filter(|f| !f.deleted) {
            *by_category.entry(fact.category).or_insert(0) += 1;
            total_confidence += fact.confidence;
            count += 1;
            if fact.local_only {
                local_only_count += 1;
            }
        }

        CacheStats {
            total_facts: count,
            by_category,
            avg_confidence: if count > 0 {
                total_confidence / count as f32
            } else {
                0.0
            },
            local_only_facts: local_only_count,
            pending_submissions: self.pending_submissions.len(),
            pending_feedback: self.pending_feedback.len(),
            last_sync: self.last_sync,
        }
    }

    /// Export all facts as JSON (for /profile export)
    pub fn export_json(&self) -> Result<String> {
        let facts: Vec<&PersonalFact> = self.facts.values().filter(|f| !f.deleted).collect();
        Ok(serde_json::to_string_pretty(&facts)?)
    }

    /// Import facts from JSON (for /profile import)
    pub fn import_json(&mut self, json: &str) -> Result<ImportResult> {
        let facts: Vec<PersonalFact> = serde_json::from_str(json)?;
        let mut imported = 0;
        let mut updated = 0;

        for mut fact in facts {
            if let Some(existing_id) = self.facts_by_key.get(&fact.key) {
                // Update existing by key
                fact.id = existing_id.clone();
                updated += 1;
            } else {
                imported += 1;
            }
            self.upsert_fact(fact)?;
        }

        Ok(ImportResult { imported, updated })
    }
}

/// Result of merging facts from server
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Number of new facts added.
    pub added: u32,
    /// Number of existing facts updated.
    pub updated: u32,
    /// Number of merge conflicts.
    pub conflicts: u32,
}

/// Result of importing facts
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Number of facts imported.
    pub imported: u32,
    /// Number of existing facts updated.
    pub updated: u32,
}

/// Statistics about the cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of cached facts.
    pub total_facts: u32,
    /// Counts by category.
    pub by_category: HashMap<PersonalFactCategory, u32>,
    /// Average confidence score.
    pub avg_confidence: f32,
    /// Facts that exist only locally.
    pub local_only_facts: u32,
    /// Number of pending fact submissions.
    pub pending_submissions: usize,
    /// Number of pending feedback reports.
    pub pending_feedback: usize,
    /// Unix timestamp of last sync.
    pub last_sync: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::bks_pks::personal::fact::PersonalFactSource;

    fn create_test_fact(key: &str, value: &str) -> PersonalFact {
        PersonalFact::new(
            PersonalFactCategory::Preference,
            key.to_string(),
            value.to_string(),
            None,
            PersonalFactSource::ExplicitStatement,
            false,
        )
    }

    #[test]
    fn test_cache_creation() {
        let cache = PersonalKnowledgeCache::in_memory(100).unwrap();
        assert_eq!(cache.last_sync, 0);
        assert_eq!(cache.all_facts().count(), 0);
    }

    #[test]
    fn test_add_and_get_fact() {
        let mut cache = PersonalKnowledgeCache::in_memory(100).unwrap();
        let fact = create_test_fact("language", "Rust");

        let id = fact.id.clone();
        cache.add_fact(fact).unwrap();

        let retrieved = cache.get_fact(&id).unwrap();
        assert_eq!(retrieved.value, "Rust");
    }

    #[test]
    fn test_get_by_key() {
        let mut cache = PersonalKnowledgeCache::in_memory(100).unwrap();
        cache
            .add_fact(create_test_fact("language", "Rust"))
            .unwrap();

        let retrieved = cache.get_fact_by_key("language").unwrap();
        assert_eq!(retrieved.value, "Rust");
    }

    #[test]
    fn test_upsert_fact() {
        let mut cache = PersonalKnowledgeCache::in_memory(100).unwrap();

        // Add initial fact
        cache
            .upsert_fact(create_test_fact("language", "Python"))
            .unwrap();
        assert_eq!(cache.get_fact_by_key("language").unwrap().value, "Python");

        // Upsert with same key should update
        cache
            .upsert_fact(create_test_fact("language", "Rust"))
            .unwrap();
        let fact = cache.get_fact_by_key("language").unwrap();
        assert_eq!(fact.value, "Rust");
        assert_eq!(fact.reinforcements, 1); // Should be incremented
    }

    #[test]
    fn test_facts_by_category() {
        let mut cache = PersonalKnowledgeCache::in_memory(100).unwrap();

        cache.add_fact(create_test_fact("lang", "Rust")).unwrap();

        let mut identity_fact = create_test_fact("name", "John");
        identity_fact.category = PersonalFactCategory::Identity;
        cache.add_fact(identity_fact).unwrap();

        let pref_facts = cache.facts_by_category(PersonalFactCategory::Preference);
        assert_eq!(pref_facts.len(), 1);

        let id_facts = cache.facts_by_category(PersonalFactCategory::Identity);
        assert_eq!(id_facts.len(), 1);
    }

    #[test]
    fn test_search_facts() {
        let mut cache = PersonalKnowledgeCache::in_memory(100).unwrap();

        cache
            .add_fact(create_test_fact("language", "Rust"))
            .unwrap();
        cache
            .add_fact(create_test_fact("framework", "Actix"))
            .unwrap();

        let results = cache.search_facts("rust");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "language");
    }

    #[test]
    fn test_local_only_facts() {
        let mut cache = PersonalKnowledgeCache::in_memory(100).unwrap();

        let mut local_fact = create_test_fact("secret", "value");
        local_fact.local_only = true;
        cache.add_fact(local_fact.clone()).unwrap();

        // Should not be able to queue for sync
        assert!(!cache.queue_submission(local_fact).unwrap());

        // Should not be in syncable facts
        assert!(cache.get_syncable_facts().is_empty());
    }

    #[test]
    fn test_export_import_json() {
        let mut cache = PersonalKnowledgeCache::in_memory(100).unwrap();

        cache.add_fact(create_test_fact("lang", "Rust")).unwrap();
        cache
            .add_fact(create_test_fact("editor", "VSCode"))
            .unwrap();

        let json = cache.export_json().unwrap();

        // Create new cache and import
        let mut new_cache = PersonalKnowledgeCache::in_memory(100).unwrap();
        let result = new_cache.import_json(&json).unwrap();

        assert_eq!(result.imported, 2);
        assert_eq!(result.updated, 0);
        assert_eq!(new_cache.all_facts().count(), 2);
    }

    #[test]
    fn test_stats() {
        let mut cache = PersonalKnowledgeCache::in_memory(100).unwrap();

        cache.add_fact(create_test_fact("lang", "Rust")).unwrap();

        let mut local_fact = create_test_fact("secret", "value");
        local_fact.local_only = true;
        cache.add_fact(local_fact).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.total_facts, 2);
        assert_eq!(stats.local_only_facts, 1);
    }
}
