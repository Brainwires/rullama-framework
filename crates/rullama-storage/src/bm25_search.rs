use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument, doc};

/// BM25-based keyword search using Tantivy
pub struct BM25Search {
    index: Index,
    id_field: Field,
    string_id_field: Field,
    content_field: Field,
    file_path_field: Field,
    /// Path to the index directory (needed for lock cleanup)
    index_path: std::path::PathBuf,
    /// Mutex to ensure only one IndexWriter is created at a time
    writer_lock: Mutex<()>,
}

/// Search result from BM25
#[derive(Debug, Clone)]
pub struct BM25Result {
    /// Legacy numeric document identifier (Tantivy internal).
    pub id: u64,
    /// Stable composite key matching the LanceDB `id` column (`"{file_path}:{start_line}"`).
    pub string_id: String,
    /// BM25 relevance score.
    pub score: f32,
}

impl BM25Search {
    /// Create a new BM25 search index
    pub fn new<P: AsRef<Path>>(index_path: P) -> Result<Self> {
        let index_path = index_path.as_ref().to_path_buf();

        // Create schema with ID, content, and file_path fields.
        // content is TEXT | STORED so documents can be retrieved after indexing.
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_u64_field("id", STORED | INDEXED);
        let string_id_field = schema_builder.add_text_field("string_id", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let file_path_field = schema_builder.add_text_field("file_path", STRING | STORED);
        let schema = schema_builder.build();

        // Create or open index, validating schema on reopen to detect drift.
        std::fs::create_dir_all(&index_path).context("Failed to create BM25 index directory")?;

        let index = if index_path.join("meta.json").exists() {
            let existing =
                Index::open_in_dir(&index_path).context("Failed to open existing BM25 index")?;
            // Validate that the on-disk schema matches the expected fields.
            // If it doesn't (e.g. after a schema change), recreate the index.
            let schema_ok = existing.schema().get_field("id").is_ok()
                && existing.schema().get_field("string_id").is_ok()
                && existing.schema().get_field("content").is_ok()
                && existing.schema().get_field("file_path").is_ok();
            if schema_ok {
                existing
            } else {
                tracing::warn!(
                    "BM25 index schema mismatch at {:?} — recreating index",
                    index_path
                );
                std::fs::remove_dir_all(&index_path)
                    .context("Failed to remove stale BM25 index")?;
                std::fs::create_dir_all(&index_path)
                    .context("Failed to recreate BM25 index directory")?;
                Index::create_in_dir(&index_path, schema.clone())
                    .context("Failed to recreate BM25 index")?
            }
        } else {
            Index::create_in_dir(&index_path, schema.clone())
                .context("Failed to create BM25 index")?
        };

        Ok(Self {
            index,
            id_field,
            string_id_field,
            content_field,
            file_path_field,
            index_path,
            writer_lock: Mutex::new(()),
        })
    }

    /// Check if a lock file is stale (older than 5 minutes with no recent activity)
    fn is_lock_stale(lock_path: &Path) -> bool {
        if !lock_path.exists() {
            return false;
        }

        // Check file modification time
        if let Ok(metadata) = std::fs::metadata(lock_path)
            && let Ok(modified) = metadata.modified()
            && let Ok(elapsed) = modified.elapsed()
        {
            // Consider lock stale if older than 5 minutes
            return elapsed.as_secs() > 300;
        }

        false
    }

    /// Try to clean up stale lock files only if they appear to be from crashed processes
    fn try_cleanup_stale_locks(index_path: &Path) -> Result<bool> {
        let writer_lock = index_path.join(".tantivy-writer.lock");
        let meta_lock = index_path.join(".tantivy-meta.lock");

        let writer_stale = Self::is_lock_stale(&writer_lock);
        let meta_stale = Self::is_lock_stale(&meta_lock);

        if !writer_stale && !meta_stale {
            return Ok(false); // Locks appear to be active
        }

        if writer_stale && writer_lock.exists() {
            tracing::warn!(
                "Removing stale Tantivy writer lock file (>5min old): {:?}",
                writer_lock
            );
            std::fs::remove_file(&writer_lock)
                .context("Failed to remove stale writer lock file")?;
        }

        if meta_stale && meta_lock.exists() {
            tracing::warn!(
                "Removing stale Tantivy meta lock file (>5min old): {:?}",
                meta_lock
            );
            std::fs::remove_file(&meta_lock).context("Failed to remove stale meta lock file")?;
        }

        Ok(true) // Cleaned up stale locks
    }

    /// Add documents to the index
    ///
    /// Arguments:
    /// * `documents` - Vec of (id, string_id, content, file_path) tuples
    ///   - `id`: sequential u64 for Tantivy internal use
    ///   - `string_id`: stable composite key (`"{file_path}:{start_line}"`) for cross-index fusion
    ///   - `content`: document text
    ///   - `file_path`: source file path
    pub fn add_documents(&self, documents: Vec<(u64, String, String, String)>) -> Result<()> {
        // Lock to ensure only one writer at a time (within this process)
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        // Try to create the index writer
        let mut index_writer: IndexWriter<TantivyDocument> = match self.index.writer(50_000_000) {
            Ok(writer) => writer,
            Err(e) => {
                // Check if this is a lock error
                let error_msg = format!("{}", e);
                if error_msg.contains("lock") || error_msg.contains("Lock") {
                    tracing::warn!(
                        "Index writer creation failed (possibly locked), checking for stale locks..."
                    );

                    // Try to cleanup stale locks
                    match Self::try_cleanup_stale_locks(&self.index_path) {
                        Ok(true) => {
                            // Stale locks were cleaned up, retry once
                            tracing::info!("Stale locks cleaned up, retrying writer creation...");
                            self.index.writer(50_000_000).context(
                                "Failed to create index writer after cleaning stale locks",
                            )?
                        }
                        Ok(false) => {
                            // Locks exist but are not stale (another process is actively using the index)
                            return Err(anyhow::anyhow!(
                                "BM25 index is currently being used by another process. Please wait and try again later."
                            ));
                        }
                        Err(cleanup_err) => {
                            // Failed to cleanup locks
                            return Err(anyhow::anyhow!(
                                "Failed to create index writer (locked) and failed to cleanup stale locks: {}. Original error: {}",
                                cleanup_err,
                                e
                            ));
                        }
                    }
                } else {
                    // Not a lock error, propagate original error
                    return Err(e).context("Failed to create index writer");
                }
            }
        };

        for (id, string_id, content, file_path) in documents {
            let doc = doc!(
                self.id_field => id,
                self.string_id_field => string_id,
                self.content_field => content,
                self.file_path_field => file_path,
            );
            index_writer
                .add_document(doc)
                .context("Failed to add document")?;
        }

        index_writer
            .commit()
            .context("Failed to commit documents")?;

        Ok(())
    }

    /// Search the index with BM25 scoring
    pub fn search(&self, query_text: &str, limit: usize) -> Result<Vec<BM25Result>> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("Failed to create index reader")?;

        let searcher = reader.searcher();

        // Parse query using lenient mode to handle special characters like :: in code
        // (e.g., "Tool::new" would fail strict parsing since : is a field separator)
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        let (query, errors) = query_parser.parse_query_lenient(query_text);
        if !errors.is_empty() {
            tracing::warn!(
                "BM25 query parse issues for {:?} (terms may have been dropped): {:?}",
                query_text,
                errors
            );
        }

        // Search with BM25
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .context("Failed to execute search")?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher
                .doc(doc_address)
                .context("Failed to retrieve document")?;

            let id = retrieved_doc
                .get_first(self.id_field)
                .and_then(|v| v.as_u64());
            let string_id = retrieved_doc
                .get_first(self.string_id_field)
                .and_then(|v| match v {
                    tantivy::schema::OwnedValue::Str(s) => Some(s.to_string()),
                    _ => None,
                });

            match (id, string_id) {
                (Some(id), Some(string_id)) => {
                    results.push(BM25Result {
                        id,
                        string_id,
                        score,
                    });
                }
                _ => tracing::warn!(
                    "BM25: document at {:?} is missing id or string_id field — skipping",
                    doc_address
                ),
            }
        }

        Ok(results)
    }

    /// Delete all documents for a specific ID
    pub fn delete_by_id(&self, id: u64) -> Result<()> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        let term = Term::from_field_u64(self.id_field, id);
        index_writer.delete_term(term);

        index_writer.commit().context("Failed to commit deletion")?;

        Ok(())
    }

    /// Delete all documents with a specific file_path
    ///
    /// This is used for incremental updates when files are deleted or modified.
    pub fn delete_by_file_path(&self, file_path: &str) -> Result<usize> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        let term = Term::from_field_text(self.file_path_field, file_path);
        index_writer.delete_term(term);

        index_writer
            .commit()
            .context("Failed to commit file_path deletion")?;

        // Note: Tantivy doesn't return count of deleted documents
        // Return 0 as placeholder
        Ok(0)
    }

    /// Clear the entire index
    pub fn clear(&self) -> Result<()> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        index_writer
            .delete_all_documents()
            .context("Failed to delete all documents")?;

        index_writer.commit().context("Failed to commit clear")?;

        Ok(())
    }

    /// Get index statistics
    pub fn get_stats(&self) -> Result<BM25Stats> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("Failed to create index reader")?;

        let searcher = reader.searcher();
        let total_docs = searcher.num_docs() as usize;

        Ok(BM25Stats {
            total_documents: total_docs,
        })
    }
}

/// Statistics about the BM25 index
#[derive(Debug, Clone)]
pub struct BM25Stats {
    /// Total number of indexed documents.
    pub total_documents: usize,
}

/// Trait for custom search scoring/fusion strategies.
///
/// Implement this trait to replace the default Reciprocal Rank Fusion (RRF) with
/// your own fusion algorithm (e.g., weighted linear combination, learned fusion,
/// cross-encoder reranking).
///
/// # Example
///
/// ```rust,ignore
/// use crate::bm25_search::{SearchScorer, BM25Result};
///
/// struct WeightedFusion { vector_weight: f32, keyword_weight: f32 }
///
/// impl SearchScorer for WeightedFusion {
///     fn fuse(
///         &self,
///         vector_results: Vec<(String, f32)>,
///         bm25_results: Vec<BM25Result>,
///         limit: usize,
///     ) -> Vec<(String, f32)> {
///         // Your custom fusion logic here
///         vec![]
///     }
/// }
/// ```
pub trait SearchScorer: Send + Sync {
    /// Combine vector search and BM25 keyword results into a single ranked list.
    ///
    /// - `vector_results`: (string_id, similarity_score) pairs from vector search, sorted by score desc
    /// - `bm25_results`: keyword search results with raw BM25 scores and string IDs
    /// - `limit`: maximum number of combined results to return
    ///
    /// Returns (string_id, combined_score) pairs sorted by score descending.
    fn fuse(
        &self,
        vector_results: Vec<(String, f32)>,
        bm25_results: Vec<BM25Result>,
        limit: usize,
    ) -> Vec<(String, f32)>;
}

/// Standard RRF constant (60.0 is the commonly used value from the RRF paper)
pub const RRF_K_CONSTANT: f32 = 60.0;

/// Default scorer using Reciprocal Rank Fusion (RRF).
///
/// The standard RRF approach from the paper, using k=60.
pub struct RrfScorer;

impl SearchScorer for RrfScorer {
    fn fuse(
        &self,
        vector_results: Vec<(String, f32)>,
        bm25_results: Vec<BM25Result>,
        limit: usize,
    ) -> Vec<(String, f32)> {
        reciprocal_rank_fusion(vector_results, bm25_results, limit)
    }
}

/// Reciprocal Rank Fusion (RRF) for combining vector and BM25 results
///
/// Uses stable string IDs (`"{file_path}:{start_line}"`) so vector and BM25
/// results share the same ID space and fuse correctly.
pub fn reciprocal_rank_fusion(
    vector_results: Vec<(String, f32)>,
    bm25_results: Vec<BM25Result>,
    k: usize,
) -> Vec<(String, f32)> {
    // Convert BM25 results to the same format as vector results using string_id
    let bm25_tuples: Vec<(String, f32)> = bm25_results
        .into_iter()
        .map(|r| (r.string_id, r.score))
        .collect();

    // Use the generic implementation
    reciprocal_rank_fusion_generic([vector_results, bm25_tuples], k)
}

/// Generic Reciprocal Rank Fusion (RRF) for combining arbitrary ranked lists
///
/// This is a generic version that works with any type that implements Eq + Hash + Clone.
/// Useful for combining results from different search systems.
///
/// # Arguments
/// * `ranked_lists` - Iterator of ranked result lists, each containing (id, original_score)
/// * `limit` - Maximum results to return
///
/// # Returns
/// Vec of (id, combined_rrf_score) sorted by score descending
pub fn reciprocal_rank_fusion_generic<T, I, L>(ranked_lists: I, limit: usize) -> Vec<(T, f32)>
where
    T: Eq + std::hash::Hash + Clone,
    I: IntoIterator<Item = L>,
    L: IntoIterator<Item = (T, f32)>,
{
    let mut score_map: HashMap<T, f32> = HashMap::new();

    for list in ranked_lists {
        for (rank, (id, _score)) in list.into_iter().enumerate() {
            let rrf_score = 1.0 / (RRF_K_CONSTANT + (rank + 1) as f32);
            *score_map.entry(id).or_insert(0.0) += rrf_score;
        }
    }

    let mut combined: Vec<(T, f32)> = score_map.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined.truncate(limit);

    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── reciprocal_rank_fusion ────────────────────────────────────────────

    #[test]
    fn rrf_empty_inputs_returns_empty() {
        let result = reciprocal_rank_fusion(vec![], vec![], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn rrf_vector_only_result_ranked_first_gets_highest_score() {
        let vector_results = vec![
            ("a:1".to_string(), 0.9),
            ("b:2".to_string(), 0.8),
            ("c:3".to_string(), 0.7),
        ];
        let result = reciprocal_rank_fusion(vector_results, vec![], 3);
        let scores: Vec<&str> = result.iter().map(|(id, _)| id.as_str()).collect();
        assert!(scores.contains(&"a:1"));
        assert!(scores.contains(&"b:2"));
        assert!(scores.contains(&"c:3"));
        let id1_score = result.iter().find(|(id, _)| id == "a:1").unwrap().1;
        let id2_score = result.iter().find(|(id, _)| id == "b:2").unwrap().1;
        assert!(id1_score > id2_score);
    }

    #[test]
    fn rrf_limit_caps_result_count() {
        let vector_results = vec![
            ("a:1".to_string(), 1.0),
            ("b:2".to_string(), 0.9),
            ("c:3".to_string(), 0.8),
            ("d:4".to_string(), 0.7),
        ];
        let result = reciprocal_rank_fusion(vector_results, vec![], 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn rrf_item_in_both_lists_ranks_higher() {
        // Item "x:10" appears in both vector and bm25 results
        let vector_results = vec![("x:10".to_string(), 0.9), ("y:20".to_string(), 0.8)];
        let bm25_results = vec![
            BM25Result {
                id: 10,
                string_id: "x:10".to_string(),
                score: 0.9,
            },
            BM25Result {
                id: 30,
                string_id: "z:30".to_string(),
                score: 0.7,
            },
        ];
        let result = reciprocal_rank_fusion(vector_results, bm25_results, 10);
        let score_10 = result.iter().find(|(id, _)| id == "x:10").unwrap().1;
        let score_20 = result.iter().find(|(id, _)| id == "y:20").unwrap().1;
        let score_30 = result.iter().find(|(id, _)| id == "z:30").unwrap().1;
        assert!(
            score_10 > score_20,
            "item in both lists should beat vector-only"
        );
        assert!(
            score_10 > score_30,
            "item in both lists should beat bm25-only"
        );
    }

    #[test]
    fn rrf_generic_string_ids_work() {
        let list1 = vec![("a".to_string(), 1.0f32), ("b".to_string(), 0.5)];
        let list2 = vec![("b".to_string(), 1.0f32), ("c".to_string(), 0.5)];
        let result = reciprocal_rank_fusion_generic([list1, list2], 10);
        // "b" appears in both, should have higher score
        let score_b = result.iter().find(|(id, _)| id == "b").unwrap().1;
        let score_a = result.iter().find(|(id, _)| id == "a").unwrap().1;
        let score_c = result.iter().find(|(id, _)| id == "c").unwrap().1;
        assert!(score_b > score_a);
        assert!(score_b > score_c);
    }

    #[test]
    fn rrf_k_constant_is_60() {
        assert_eq!(RRF_K_CONSTANT, 60.0);
    }

    #[test]
    fn rrf_score_for_rank_zero_is_one_over_61() {
        // At rank 0 (first item): 1 / (60 + 1) = 1/61
        let vector_results = vec![("doc:42".to_string(), 1.0)];
        let result = reciprocal_rank_fusion(vector_results, vec![], 1);
        let score = result[0].1;
        let expected = 1.0 / 61.0f32;
        assert!(
            (score - expected).abs() < 1e-6,
            "score={score}, expected={expected}"
        );
    }

    // ── BM25Search ────────────────────────────────────────────────────────

    #[test]
    fn bm25search_creates_index_in_temp_dir() {
        let dir = TempDir::new().unwrap();
        let search = BM25Search::new(dir.path()).unwrap();
        let stats = search.get_stats().unwrap();
        assert_eq!(stats.total_documents, 0);
    }

    #[test]
    fn bm25search_add_and_count_documents() {
        let dir = TempDir::new().unwrap();
        let search = BM25Search::new(dir.path()).unwrap();
        search
            .add_documents(vec![
                (
                    1,
                    "file1.rs:1".to_string(),
                    "the quick brown fox".to_string(),
                    "file1.rs".to_string(),
                ),
                (
                    2,
                    "file2.rs:1".to_string(),
                    "jumps over the lazy dog".to_string(),
                    "file2.rs".to_string(),
                ),
            ])
            .unwrap();
        let stats = search.get_stats().unwrap();
        assert_eq!(stats.total_documents, 2);
    }

    #[test]
    fn bm25search_returns_relevant_results() {
        let dir = TempDir::new().unwrap();
        let search = BM25Search::new(dir.path()).unwrap();
        search
            .add_documents(vec![
                (
                    1,
                    "auth.rs:1".to_string(),
                    "authentication login user password".to_string(),
                    "auth.rs".to_string(),
                ),
                (
                    2,
                    "db.rs:1".to_string(),
                    "database storage connection pool".to_string(),
                    "db.rs".to_string(),
                ),
                (
                    3,
                    "oauth.rs:1".to_string(),
                    "authentication oauth token".to_string(),
                    "oauth.rs".to_string(),
                ),
            ])
            .unwrap();

        let results = search.search("authentication", 10).unwrap();
        assert!(
            !results.is_empty(),
            "should find results for 'authentication'"
        );
        for r in &results {
            assert!(r.score > 0.0);
            assert!(!r.string_id.is_empty(), "string_id should be populated");
        }
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&1) || ids.contains(&3));
    }

    #[test]
    fn bm25search_search_returns_empty_for_unknown_term() {
        let dir = TempDir::new().unwrap();
        let search = BM25Search::new(dir.path()).unwrap();
        search
            .add_documents(vec![(
                1,
                "f.rs:1".to_string(),
                "some content".to_string(),
                "f.rs".to_string(),
            )])
            .unwrap();
        let results = search.search("xyzabsolutelynotinindex", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn bm25search_clear_removes_all_documents() {
        let dir = TempDir::new().unwrap();
        let search = BM25Search::new(dir.path()).unwrap();
        search
            .add_documents(vec![(
                1,
                "f.rs:1".to_string(),
                "content".to_string(),
                "f.rs".to_string(),
            )])
            .unwrap();
        search.clear().unwrap();
        let stats = search.get_stats().unwrap();
        assert_eq!(stats.total_documents, 0);
    }

    #[test]
    fn bm25search_delete_by_id() {
        let dir = TempDir::new().unwrap();
        let search = BM25Search::new(dir.path()).unwrap();
        search
            .add_documents(vec![
                (
                    1,
                    "a.rs:1".to_string(),
                    "hello world".to_string(),
                    "a.rs".to_string(),
                ),
                (
                    2,
                    "b.rs:1".to_string(),
                    "goodbye world".to_string(),
                    "b.rs".to_string(),
                ),
            ])
            .unwrap();
        search.delete_by_id(1).unwrap();
        // After deletion, searching for doc 1's unique term should not return id 1
        let results = search.search("hello", 10).unwrap();
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert!(!ids.contains(&1), "id 1 should be deleted");
    }

    #[test]
    fn bm25search_reopen_existing_index() {
        let dir = TempDir::new().unwrap();
        // Create and index
        {
            let search = BM25Search::new(dir.path()).unwrap();
            search
                .add_documents(vec![(
                    1,
                    "p.rs:1".to_string(),
                    "persistent content".to_string(),
                    "p.rs".to_string(),
                )])
                .unwrap();
        }
        // Reopen and verify docs persist
        let search2 = BM25Search::new(dir.path()).unwrap();
        let stats = search2.get_stats().unwrap();
        assert_eq!(stats.total_documents, 1);
    }
}
