//! # Tenant scoping
//!
//! This module supports optional per-owner (tenant) scoping of thoughts via
//! the [`Thought::owner_id`] field and the matching request-side `owner_id`
//! fields on the request types in [`crate::knowledge::types`].
//!
//! Behavior is deliberately asymmetric to preserve single-tenant backward
//! compatibility:
//!
//! - If a **write** request has `owner_id: None`, the thought is stored with
//!   `owner_id = None` (unscoped row).
//! - If a **read** request has `owner_id: None`, **no owner filter is added**
//!   to the storage query. Unscoped callers see every row regardless of
//!   owner, matching the pre-tenant-scoping single-tenant semantics.
//! - If a **read** request has `owner_id: Some(x)`, an equality filter
//!   `owner_id = x` is appended to the existing filter composition, so only
//!   thoughts owned by `x` are returned (and unscoped `None` rows are NOT
//!   surfaced — opt-in callers get an isolated view).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;
use tracing;

use crate::knowledge::bks_pks::{
    BehavioralKnowledgeCache, PersonalFactCollector, PersonalKnowledgeCache,
};
use brainwires_storage::{
    CachedEmbeddingProvider, FieldDef, FieldType, FieldValue, Filter, Record, StorageBackend,
    record_get,
};

#[cfg(feature = "knowledge")]
use brainwires_storage::LanceDatabase;

use crate::knowledge::config::MemoryBankConfig;
use crate::knowledge::fact_extractor;
use crate::knowledge::thought::{Thought, ThoughtCategory, ThoughtSource};
use crate::knowledge::types::*;

/// Central orchestrator for all Open Brain storage operations.
pub struct BrainClient {
    backend: Arc<dyn StorageBackend>,
    embeddings: Arc<CachedEmbeddingProvider>,
    pks_cache: PersonalKnowledgeCache,
    bks_cache: BehavioralKnowledgeCache,
    fact_collector: PersonalFactCollector,
    /// Optional memory bank configuration (mission, directives, disposition).
    config: MemoryBankConfig,
}

const THOUGHTS_TABLE: &str = "thoughts";

/// EMA alpha for confidence updates on corroboration/contradiction.
const EVIDENCE_EMA_ALPHA: f32 = 0.3;
/// Score threshold above which a similar thought is a corroboration.
const CORROBORATION_THRESHOLD: f32 = 0.85;
/// Score threshold above which a similar thought may be a contradiction.
const CONTRADICTION_THRESHOLD: f32 = 0.70;

impl BrainClient {
    /// Create a new BrainClient with default paths.
    ///
    /// - LanceDB: `~/.brainwires/brain/`
    /// - PKS:     `~/.brainwires/pks.db`
    /// - BKS:     `~/.brainwires/bks.db`
    pub async fn new() -> Result<Self> {
        let base = dirs::home_dir()
            .context("Cannot determine home directory")?
            .join(".brainwires");

        std::fs::create_dir_all(&base)?;

        let lance_path = base.join("brain");
        let pks_path = base.join("pks.db");
        let bks_path = base.join("bks.db");

        Self::with_paths(
            lance_path
                .to_str()
                .context("lance path is not valid UTF-8")?,
            pks_path.to_str().context("pks path is not valid UTF-8")?,
            bks_path.to_str().context("bks path is not valid UTF-8")?,
        )
        .await
    }

    /// Create with explicit paths (useful for testing).
    ///
    /// Creates a LanceDatabase internally as the default backend.
    pub async fn with_paths(lance_path: &str, pks_path: &str, bks_path: &str) -> Result<Self> {
        let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
        let backend: Arc<dyn StorageBackend> = Arc::new(LanceDatabase::new(lance_path).await?);

        Self::with_backend(backend, embeddings, pks_path, bks_path).await
    }

    /// Create with an externally-provided storage backend.
    ///
    /// This is the primary constructor for dependency injection — any
    /// [`StorageBackend`] implementation can be used (LanceDB, Postgres, etc.).
    pub async fn with_backend(
        backend: Arc<dyn StorageBackend>,
        embeddings: Arc<CachedEmbeddingProvider>,
        pks_path: &str,
        bks_path: &str,
    ) -> Result<Self> {
        // Ensure the thoughts table exists
        Self::ensure_thoughts_table(&*backend, embeddings.dimension()).await?;

        let pks_cache = PersonalKnowledgeCache::new(pks_path, 1000)?;
        let bks_cache = BehavioralKnowledgeCache::new(bks_path, 1000)?;
        let fact_collector = PersonalFactCollector::default();

        Ok(Self {
            backend,
            embeddings,
            pks_cache,
            bks_cache,
            fact_collector,
            config: MemoryBankConfig::default(),
        })
    }

    /// Create with default paths and a custom [`MemoryBankConfig`].
    pub async fn with_bank_config(config: MemoryBankConfig) -> Result<Self> {
        let mut client = Self::new().await?;
        client.config = config;
        Ok(client)
    }

    /// Set or replace the [`MemoryBankConfig`] on an existing client.
    pub fn set_config(&mut self, config: MemoryBankConfig) {
        self.config = config;
    }

    /// Return a reference to the active [`MemoryBankConfig`].
    pub fn config(&self) -> &MemoryBankConfig {
        &self.config
    }

    // ── Table management ─────────────────────────────────────────────────

    async fn ensure_thoughts_table(backend: &dyn StorageBackend, dim: usize) -> Result<()> {
        backend
            .ensure_table(
                THOUGHTS_TABLE,
                &[
                    FieldDef::required("vector", FieldType::Vector(dim)),
                    FieldDef::required("id", FieldType::Utf8),
                    FieldDef::required("content", FieldType::Utf8),
                    FieldDef::required("category", FieldType::Utf8),
                    FieldDef::required("tags", FieldType::Utf8),
                    FieldDef::required("source", FieldType::Utf8),
                    FieldDef::required("importance", FieldType::Float32),
                    FieldDef::required("created_at", FieldType::Int64),
                    FieldDef::required("updated_at", FieldType::Int64),
                    FieldDef::required("deleted", FieldType::Boolean),
                    FieldDef::optional("confidence", FieldType::Float32),
                    FieldDef::optional("evidence_chain", FieldType::Utf8),
                    FieldDef::optional("reinforcement_count", FieldType::Int64),
                    FieldDef::optional("contradiction_count", FieldType::Int64),
                    FieldDef::optional("owner_id", FieldType::Utf8),
                ],
            )
            .await
            .context("Failed to create thoughts table")?;

        tracing::info!("Ensured thoughts table exists");
        Ok(())
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Default importance score based on thought category.
    fn importance_for_category(cat: &ThoughtCategory) -> f32 {
        match cat {
            ThoughtCategory::Decision => 0.85,
            ThoughtCategory::ActionItem => 0.8,
            ThoughtCategory::Insight => 0.75,
            ThoughtCategory::Idea => 0.65,
            ThoughtCategory::Person => 0.6,
            ThoughtCategory::MeetingNote => 0.55,
            ThoughtCategory::Reference => 0.5,
            ThoughtCategory::Conversation => 0.45,
            ThoughtCategory::General => 0.4,
        }
    }

    // ── Capture ──────────────────────────────────────────────────────────

    /// Capture a new thought, embed it, detect category, extract PKS facts.
    pub async fn capture_thought(
        &mut self,
        req: CaptureThoughtRequest,
    ) -> Result<CaptureThoughtResponse> {
        // Build the Thought
        let category = match &req.category {
            Some(c) => ThoughtCategory::parse(c),
            None => fact_extractor::detect_category(&req.content),
        };

        let mut auto_tags = fact_extractor::extract_tags(&req.content);
        if let Some(ref user_tags) = req.tags {
            for t in user_tags {
                let lower = t.to_lowercase();
                if !auto_tags.contains(&lower) {
                    auto_tags.push(lower);
                }
            }
        }

        // Auto-tag with the mission slug when a mission is configured.
        if let Some(mission_tag) = self.config.mission_tag()
            && !auto_tags.contains(&mission_tag)
        {
            auto_tags.push(mission_tag);
        }

        let source = req
            .source
            .as_deref()
            .map(ThoughtSource::parse)
            .unwrap_or(ThoughtSource::ManualCapture);

        let thought = Thought::new(req.content.clone())
            .with_category(category)
            .with_tags(auto_tags.clone())
            .with_source(source)
            .with_importance(
                req.importance
                    .unwrap_or_else(|| Self::importance_for_category(&category)),
            )
            .with_owner_id(req.owner_id.clone());

        // Embed
        let embedding = self.embeddings.embed(&thought.content)?;

        // Store via backend
        let record = Self::thought_to_record(&thought, &embedding);
        self.backend
            .insert(THOUGHTS_TABLE, vec![record])
            .await
            .context("Failed to store thought")?;

        // Extract PKS facts
        let facts = self.fact_collector.process_message(&req.content);
        let facts_count = facts.len();
        for fact in facts {
            if let Err(e) = self.pks_cache.upsert_fact(fact) {
                tracing::warn!("Failed to upsert PKS fact: {}", e);
            }
        }

        // Run evidence check: find corroborations / contradictions among existing thoughts.
        let evidence = self
            .apply_evidence_check(&thought.id, &req.content)
            .await
            .unwrap_or_default();

        // Compute initial confidence for the new thought based on corroboration count.
        let initial_confidence = (0.5 + 0.05 * evidence.corroborations.len() as f32
            - 0.05 * evidence.contradictions.len() as f32)
            .clamp(0.0, 1.0);

        // Persist updated confidence + evidence_chain for the new thought itself.
        if !evidence.corroborations.is_empty() || !evidence.contradictions.is_empty() {
            let mut all_evidence = evidence.corroborations.clone();
            all_evidence.extend(evidence.contradictions.iter().cloned());

            // Update the newly inserted thought record with its evidence data.
            let delete_filter = Filter::Eq("id".into(), FieldValue::Utf8(Some(thought.id.clone())));
            let _ = self.backend.delete(THOUGHTS_TABLE, &delete_filter).await;
            let mut updated_thought = thought.clone();
            updated_thought.confidence = initial_confidence;
            updated_thought.evidence_chain = all_evidence;
            let embedding = self.embeddings.embed_cached(&updated_thought.content)?;
            let record = Self::thought_to_record(&updated_thought, &embedding);
            let _ = self.backend.insert(THOUGHTS_TABLE, vec![record]).await;
        }

        tracing::info!(
            id = %thought.id,
            category = %category,
            facts = facts_count,
            corroborations = evidence.corroborations.len(),
            contradictions = evidence.contradictions.len(),
            "Captured thought"
        );

        Ok(CaptureThoughtResponse {
            id: thought.id,
            category: category.to_string(),
            tags: auto_tags,
            importance: thought.importance,
            facts_extracted: facts_count,
            corroborations: evidence.corroborations,
            contradictions: evidence.contradictions,
            confidence: initial_confidence,
        })
    }

    /// Batch-capture multiple thoughts in a single embed + insert pass.
    ///
    /// Skips per-message evidence checks for speed. PKS extraction still runs per message.
    /// Returns the number of thoughts stored.
    pub async fn capture_thoughts_batch(
        &mut self,
        requests: Vec<CaptureThoughtRequest>,
    ) -> Result<usize> {
        if requests.is_empty() {
            return Ok(0);
        }

        // Build Thought objects with category detection + importance
        let thoughts: Vec<Thought> = requests
            .iter()
            .map(|req| {
                let category = match &req.category {
                    Some(c) => ThoughtCategory::parse(c),
                    None => fact_extractor::detect_category(&req.content),
                };
                let mut auto_tags = fact_extractor::extract_tags(&req.content);
                if let Some(ref user_tags) = req.tags {
                    for t in user_tags {
                        if !auto_tags.contains(t) {
                            auto_tags.push(t.clone());
                        }
                    }
                }
                let source = req
                    .source
                    .as_deref()
                    .map(ThoughtSource::parse)
                    .unwrap_or(ThoughtSource::ManualCapture);
                Thought::new(req.content.clone())
                    .with_category(category)
                    .with_tags(auto_tags)
                    .with_source(source)
                    .with_importance(
                        req.importance
                            .unwrap_or_else(|| Self::importance_for_category(&category)),
                    )
                    .with_owner_id(req.owner_id.clone())
            })
            .collect();

        // Single batch embed
        let contents: Vec<String> = thoughts.iter().map(|t| t.content.clone()).collect();
        let embeddings = self.embeddings.embed_batch(&contents)?;

        // Build records
        let records: Vec<Record> = thoughts
            .iter()
            .zip(embeddings.iter())
            .map(|(thought, emb)| Self::thought_to_record(thought, emb))
            .collect();

        let count = records.len();
        self.backend
            .insert(THOUGHTS_TABLE, records)
            .await
            .context("Failed to batch-store thoughts")?;

        // PKS extraction per message (sync, fast)
        for req in &requests {
            let facts = self.fact_collector.process_message(&req.content);
            for fact in facts {
                if let Err(e) = self.pks_cache.upsert_fact(fact) {
                    tracing::warn!("Failed to upsert PKS fact: {}", e);
                }
            }
        }

        tracing::info!("Batch-captured {} thoughts", count);
        Ok(count)
    }

    // ── Search (semantic) ────────────────────────────────────────────────

    /// Semantic search across thoughts and optionally PKS facts.
    pub async fn search_memory(&self, req: SearchMemoryRequest) -> Result<SearchMemoryResponse> {
        let search_thoughts = req
            .sources
            .as_ref()
            .is_none_or(|s| s.iter().any(|x| x == "thoughts"));
        let search_facts = req
            .sources
            .as_ref()
            .is_none_or(|s| s.iter().any(|x| x == "facts"));

        let mut results = Vec::new();

        // 1. Thought vector search
        if search_thoughts {
            let query_embedding = self.embeddings.embed_cached(&req.query)?;

            // Build filter: deleted = false, optional category
            let mut filters = vec![Filter::Eq(
                "deleted".into(),
                FieldValue::Boolean(Some(false)),
            )];

            if let Some(ref cat) = req.category {
                let cat_str = ThoughtCategory::parse(cat).as_str().to_string();
                filters.push(Filter::Eq(
                    "category".into(),
                    FieldValue::Utf8(Some(cat_str)),
                ));
            }

            // Tenant scoping: only apply an owner filter when the caller opted in.
            if let Some(ref owner) = req.owner_id {
                filters.push(Filter::Eq(
                    "owner_id".into(),
                    FieldValue::Utf8(Some(owner.clone())),
                ));
            }

            let filter = Filter::And(filters);

            let scored_records = self
                .backend
                .vector_search(
                    THOUGHTS_TABLE,
                    "vector",
                    query_embedding,
                    req.limit,
                    Some(&filter),
                )
                .await?;

            for sr in scored_records {
                let score = sr.score;
                if score >= req.min_score {
                    let thought = Self::record_to_thought(&sr.record)?;
                    results.push(MemorySearchResult {
                        content: thought.content,
                        score,
                        source: "thoughts".into(),
                        thought_id: Some(thought.id),
                        category: Some(thought.category.to_string()),
                        tags: Some(thought.tags),
                        created_at: Some(thought.created_at),
                    });
                }
            }
        }

        // 2. PKS keyword search
        if search_facts {
            let pks_results = self.pks_cache.search_facts(&req.query);
            for fact in pks_results {
                let score = 0.7; // Flat relevance for keyword matches
                if score >= req.min_score {
                    results.push(MemorySearchResult {
                        content: format!("{}: {}", fact.key, fact.value),
                        score,
                        source: "facts".into(),
                        thought_id: None,
                        category: Some(format!("{:?}", fact.category)),
                        tags: None,
                        created_at: Some(fact.created_at),
                    });
                }
            }
        }

        // Apply memory bank config: directive filtering + disposition scoring.
        if !self.config.is_noop() {
            results.retain(|r| !self.config.blocks_content(&r.content));
            for r in &mut results {
                let delta = self.config.disposition_score_delta(&r.content);
                r.score = (r.score + delta).clamp(0.0, 1.0);
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(req.limit);

        let total = results.len();
        Ok(SearchMemoryResponse { results, total })
    }

    // ── List recent ──────────────────────────────────────────────────────

    /// List recent thoughts, optionally filtered by category and time range.
    pub async fn list_recent(&self, req: ListRecentRequest) -> Result<ListRecentResponse> {
        let since_ts = match &req.since {
            Some(s) => chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.timestamp())
                .unwrap_or_else(|_| Utc::now().timestamp() - 7 * 86400),
            None => Utc::now().timestamp() - 7 * 86400,
        };

        let mut filters = vec![
            Filter::Eq("deleted".into(), FieldValue::Boolean(Some(false))),
            Filter::Gte("created_at".into(), FieldValue::Int64(Some(since_ts))),
        ];

        if let Some(ref cat) = req.category {
            let cat_str = ThoughtCategory::parse(cat).as_str().to_string();
            filters.push(Filter::Eq(
                "category".into(),
                FieldValue::Utf8(Some(cat_str)),
            ));
        }

        // Tenant scoping: only apply an owner filter when the caller opted in.
        if let Some(ref owner) = req.owner_id {
            filters.push(Filter::Eq(
                "owner_id".into(),
                FieldValue::Utf8(Some(owner.clone())),
            ));
        }

        let filter = Filter::And(filters);

        let records = self
            .backend
            .query(THOUGHTS_TABLE, Some(&filter), Some(req.limit))
            .await?;

        let mut thoughts = Self::records_to_thoughts(&records)?;
        thoughts.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        thoughts.truncate(req.limit);

        let total = thoughts.len();
        let summaries = thoughts
            .into_iter()
            .map(|t| ThoughtSummary {
                id: t.id,
                content: t.content,
                category: t.category.to_string(),
                tags: t.tags,
                importance: t.importance,
                created_at: t.created_at,
            })
            .collect();

        Ok(ListRecentResponse {
            thoughts: summaries,
            total,
        })
    }

    /// Query thought content strings matching a filter. Used for deduplication.
    pub async fn query_thought_contents(
        &self,
        filter: &Filter,
        limit: usize,
    ) -> Result<Vec<String>> {
        let records = self
            .backend
            .query(THOUGHTS_TABLE, Some(filter), Some(limit))
            .await?;
        let thoughts = Self::records_to_thoughts(&records)?;
        Ok(thoughts.into_iter().map(|t| t.content).collect())
    }

    // ── Get by ID ────────────────────────────────────────────────────────

    /// Get a single thought by ID, optionally scoped to an owner.
    ///
    /// When `owner_id` is `None`, the lookup is unscoped (matches pre-tenant
    /// behavior). When `Some(x)`, only a thought whose stored owner_id equals
    /// `x` is returned; otherwise this returns `None`.
    pub async fn get_thought(
        &self,
        id: &str,
        owner_id: Option<&str>,
    ) -> Result<Option<GetThoughtResponse>> {
        let mut parts = vec![
            Filter::Eq("id".into(), FieldValue::Utf8(Some(id.to_string()))),
            Filter::Eq("deleted".into(), FieldValue::Boolean(Some(false))),
        ];
        if let Some(owner) = owner_id {
            parts.push(Filter::Eq(
                "owner_id".into(),
                FieldValue::Utf8(Some(owner.to_string())),
            ));
        }
        let filter = Filter::And(parts);

        let records = self
            .backend
            .query(THOUGHTS_TABLE, Some(&filter), Some(1))
            .await?;

        let thoughts = Self::records_to_thoughts(&records)?;

        Ok(thoughts.into_iter().next().map(|t| GetThoughtResponse {
            id: t.id,
            content: t.content,
            category: t.category.to_string(),
            tags: t.tags,
            source: t.source.to_string(),
            importance: t.importance,
            created_at: t.created_at,
            updated_at: t.updated_at,
        }))
    }

    // ── Search knowledge (PKS/BKS) ──────────────────────────────────────

    /// Search PKS and/or BKS knowledge stores.
    pub fn search_knowledge(&self, req: SearchKnowledgeRequest) -> Result<SearchKnowledgeResponse> {
        let search_pks = req
            .source
            .as_ref()
            .is_none_or(|s| s == "all" || s == "personal");
        let search_bks = req
            .source
            .as_ref()
            .is_none_or(|s| s == "all" || s == "behavioral");

        let mut results = Vec::new();

        if search_pks {
            let pks_results = self.pks_cache.search_facts(&req.query);
            for fact in pks_results {
                if fact.confidence >= req.min_confidence {
                    results.push(KnowledgeResult {
                        source: "personal".into(),
                        category: format!("{:?}", fact.category),
                        key: fact.key.clone(),
                        value: fact.value.clone(),
                        confidence: fact.confidence,
                        context: fact.context.clone(),
                    });
                }
            }
        }

        if search_bks {
            let bks_results = self
                .bks_cache
                .get_matching_truths_with_scores(&req.query, req.min_confidence, req.limit)
                .unwrap_or_default();
            for (truth, score) in bks_results {
                results.push(KnowledgeResult {
                    source: "behavioral".into(),
                    category: format!("{:?}", truth.category),
                    key: truth.context_pattern.clone(),
                    value: truth.rule.clone(),
                    confidence: score,
                    context: Some(truth.rationale.clone()),
                });
            }
        }

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(req.limit);

        let total = results.len();
        Ok(SearchKnowledgeResponse { results, total })
    }

    // ── Stats ────────────────────────────────────────────────────────────

    /// Get aggregate statistics across all memory stores.
    pub async fn memory_stats(&self) -> Result<MemoryStatsResponse> {
        let now = Utc::now().timestamp();
        let one_day = 86_400i64;

        // Thought stats: query all non-deleted
        let filter = Filter::Eq("deleted".into(), FieldValue::Boolean(Some(false)));
        let records = self
            .backend
            .query(THOUGHTS_TABLE, Some(&filter), None)
            .await?;
        let all_thoughts = Self::records_to_thoughts(&records)?;

        let total = all_thoughts.len();
        let mut by_category: HashMap<String, usize> = HashMap::new();
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        let mut recent_24h = 0usize;
        let mut recent_7d = 0usize;
        let mut recent_30d = 0usize;

        for t in &all_thoughts {
            *by_category.entry(t.category.to_string()).or_insert(0) += 1;
            for tag in &t.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
            let age = now - t.created_at;
            if age <= one_day {
                recent_24h += 1;
            }
            if age <= 7 * one_day {
                recent_7d += 1;
            }
            if age <= 30 * one_day {
                recent_30d += 1;
            }
        }

        let mut top_tags: Vec<(String, usize)> = tag_counts.into_iter().collect();
        top_tags.sort_by(|a, b| b.1.cmp(&a.1));
        top_tags.truncate(10);

        // PKS stats
        let pks_stats_raw = self.pks_cache.stats();
        let pks_by_cat: HashMap<String, u32> = pks_stats_raw
            .by_category
            .into_iter()
            .map(|(k, v)| (format!("{:?}", k), v))
            .collect();

        // BKS stats
        let bks_stats_raw = self.bks_cache.stats();
        let bks_by_cat: HashMap<String, u32> = bks_stats_raw
            .by_category
            .into_iter()
            .map(|(k, v)| (format!("{:?}", k), v))
            .collect();

        Ok(MemoryStatsResponse {
            thoughts: ThoughtStats {
                total,
                by_category,
                recent_24h,
                recent_7d,
                recent_30d,
                top_tags,
            },
            pks: PksStats {
                total_facts: pks_stats_raw.total_facts,
                by_category: pks_by_cat,
                avg_confidence: pks_stats_raw.avg_confidence,
            },
            bks: BksStats {
                total_truths: bks_stats_raw.total_truths,
                by_category: bks_by_cat,
            },
        })
    }

    // ── Delete ───────────────────────────────────────────────────────────

    /// Soft-delete a thought by ID, optionally scoped to an owner.
    ///
    /// When `owner_id` is `None`, any non-deleted thought with matching ID is
    /// removed (pre-tenant behavior). When `Some(x)`, only a thought owned by
    /// `x` is removed; attempting to delete another owner's thought is a
    /// no-op that returns `deleted: false`.
    pub async fn delete_thought(
        &self,
        id: &str,
        owner_id: Option<&str>,
    ) -> Result<DeleteThoughtResponse> {
        // Check existence (scope-respecting)
        let mut parts = vec![
            Filter::Eq("id".into(), FieldValue::Utf8(Some(id.to_string()))),
            Filter::Eq("deleted".into(), FieldValue::Boolean(Some(false))),
        ];
        if let Some(owner) = owner_id {
            parts.push(Filter::Eq(
                "owner_id".into(),
                FieldValue::Utf8(Some(owner.to_string())),
            ));
        }
        let filter = Filter::And(parts);

        let count = self.backend.count(THOUGHTS_TABLE, Some(&filter)).await?;
        if count == 0 {
            return Ok(DeleteThoughtResponse {
                deleted: false,
                id: id.to_string(),
            });
        }

        // Delete the row via backend (same scope-respecting filter, minus the
        // `deleted = false` clause which does not apply to hard deletion).
        let mut del_parts = vec![Filter::Eq(
            "id".into(),
            FieldValue::Utf8(Some(id.to_string())),
        )];
        if let Some(owner) = owner_id {
            del_parts.push(Filter::Eq(
                "owner_id".into(),
                FieldValue::Utf8(Some(owner.to_string())),
            ));
        }
        let delete_filter = if del_parts.len() == 1 {
            del_parts
                .into_iter()
                .next()
                .expect("len()==1 guarantees a single element")
        } else {
            Filter::And(del_parts)
        };
        self.backend.delete(THOUGHTS_TABLE, &delete_filter).await?;

        tracing::info!(id = id, "Deleted thought");
        Ok(DeleteThoughtResponse {
            deleted: true,
            id: id.to_string(),
        })
    }

    /// Update an existing thought's content in-place, preserving its ID.
    ///
    /// If `owner_id` is `Some`, the thought is only updated when its stored
    /// `owner_id` matches; otherwise this is a no-op (returns `Ok(false)`).
    /// If `owner_id` is `None`, the update applies regardless of owner,
    /// matching single-tenant semantics.
    ///
    /// Returns `Ok(true)` when the thought was updated, `Ok(false)` when it
    /// did not exist (or belonged to a different owner).
    ///
    /// Re-embeds the content so vector search stays consistent. Updates the
    /// `updated_at` timestamp; other fields (category, tags, importance,
    /// confidence, evidence_chain) are preserved.
    pub async fn update_thought(
        &self,
        id: &str,
        content: String,
        owner_id: Option<String>,
    ) -> Result<bool> {
        // Look up the existing thought with scoping.
        let mut parts = vec![
            Filter::Eq("id".into(), FieldValue::Utf8(Some(id.to_string()))),
            Filter::Eq("deleted".into(), FieldValue::Boolean(Some(false))),
        ];
        if let Some(ref owner) = owner_id {
            parts.push(Filter::Eq(
                "owner_id".into(),
                FieldValue::Utf8(Some(owner.clone())),
            ));
        }
        let filter = Filter::And(parts);

        let records = self
            .backend
            .query(THOUGHTS_TABLE, Some(&filter), Some(1))
            .await?;
        let mut thoughts = Self::records_to_thoughts(&records)?;
        let Some(mut thought) = thoughts.pop() else {
            return Ok(false);
        };

        thought.content = content;
        thought.updated_at = Utc::now().timestamp();

        self.replace_thought(&thought).await?;
        Ok(true)
    }

    /// Add a behavioral truth to the BKS.
    pub fn add_behavioral_truth(
        &mut self,
        truth: crate::knowledge::bks_pks::BehavioralTruth,
    ) -> Result<()> {
        self.bks_cache.add_truth(truth)?;
        Ok(())
    }

    /// Delete all thoughts matching a filter. Returns count deleted.
    pub async fn delete_by_filter(&self, filter: &Filter) -> Result<usize> {
        let count = self.backend.count(THOUGHTS_TABLE, Some(filter)).await?;
        if count > 0 {
            self.backend.delete(THOUGHTS_TABLE, filter).await?;
            tracing::info!("Deleted {} thoughts by filter", count);
        }
        Ok(count)
    }

    // ── Record conversion ────────────────────────────────────────────────

    fn thought_to_record(thought: &Thought, embedding: &[f32]) -> Record {
        let tags_json = serde_json::to_string(&thought.tags).unwrap_or_else(|_| "[]".into());
        let evidence_json =
            serde_json::to_string(&thought.evidence_chain).unwrap_or_else(|_| "[]".into());

        vec![
            ("vector".into(), FieldValue::Vector(embedding.to_vec())),
            ("id".into(), FieldValue::Utf8(Some(thought.id.clone()))),
            (
                "content".into(),
                FieldValue::Utf8(Some(thought.content.clone())),
            ),
            (
                "category".into(),
                FieldValue::Utf8(Some(thought.category.as_str().to_string())),
            ),
            ("tags".into(), FieldValue::Utf8(Some(tags_json))),
            (
                "source".into(),
                FieldValue::Utf8(Some(thought.source.as_str().to_string())),
            ),
            (
                "importance".into(),
                FieldValue::Float32(Some(thought.importance)),
            ),
            (
                "created_at".into(),
                FieldValue::Int64(Some(thought.created_at)),
            ),
            (
                "updated_at".into(),
                FieldValue::Int64(Some(thought.updated_at)),
            ),
            ("deleted".into(), FieldValue::Boolean(Some(thought.deleted))),
            (
                "confidence".into(),
                FieldValue::Float32(Some(thought.confidence)),
            ),
            (
                "evidence_chain".into(),
                FieldValue::Utf8(Some(evidence_json)),
            ),
            (
                "reinforcement_count".into(),
                FieldValue::Int64(Some(thought.reinforcement_count as i64)),
            ),
            (
                "contradiction_count".into(),
                FieldValue::Int64(Some(thought.contradiction_count as i64)),
            ),
            (
                "owner_id".into(),
                FieldValue::Utf8(thought.owner_id.clone()),
            ),
        ]
    }

    fn record_to_thought(record: &Record) -> Result<Thought> {
        let id = record_get(record, "id")
            .and_then(|v| v.as_str())
            .context("Missing id field")?
            .to_string();
        let content = record_get(record, "content")
            .and_then(|v| v.as_str())
            .context("Missing content field")?
            .to_string();
        let category = record_get(record, "category")
            .and_then(|v| v.as_str())
            .map(ThoughtCategory::parse)
            .context("Missing category field")?;
        let tags_str = record_get(record, "tags")
            .and_then(|v| v.as_str())
            .unwrap_or("[]");
        let tags: Vec<String> = serde_json::from_str(tags_str).unwrap_or_default();
        let source = record_get(record, "source")
            .and_then(|v| v.as_str())
            .map(ThoughtSource::parse)
            .context("Missing source field")?;
        let importance = record_get(record, "importance")
            .and_then(|v| v.as_f32())
            .context("Missing importance field")?;
        let created_at = record_get(record, "created_at")
            .and_then(|v| v.as_i64())
            .context("Missing created_at field")?;
        let updated_at = record_get(record, "updated_at")
            .and_then(|v| v.as_i64())
            .context("Missing updated_at field")?;
        let deleted = record_get(record, "deleted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let confidence = record_get(record, "confidence")
            .and_then(|v| v.as_f32())
            .unwrap_or(0.5);
        let evidence_str = record_get(record, "evidence_chain")
            .and_then(|v| v.as_str())
            .unwrap_or("[]");
        let evidence_chain: Vec<String> = serde_json::from_str(evidence_str).unwrap_or_default();
        let reinforcement_count = record_get(record, "reinforcement_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as u32;
        let contradiction_count = record_get(record, "contradiction_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as u32;
        let owner_id = record_get(record, "owner_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(Thought {
            id,
            content,
            category,
            tags,
            source,
            importance,
            created_at,
            updated_at,
            deleted,
            confidence,
            evidence_chain,
            reinforcement_count,
            contradiction_count,
            owner_id,
        })
    }

    fn records_to_thoughts(records: &[Record]) -> Result<Vec<Thought>> {
        records.iter().map(Self::record_to_thought).collect()
    }

    /// Persist an updated `Thought` by deleting the old record and reinserting.
    ///
    /// Required because `StorageBackend` has no `update` method.
    async fn replace_thought(&self, thought: &Thought) -> Result<()> {
        let delete_filter = Filter::Eq("id".into(), FieldValue::Utf8(Some(thought.id.clone())));
        self.backend.delete(THOUGHTS_TABLE, &delete_filter).await?;
        let embedding = self.embeddings.embed_cached(&thought.content)?;
        let record = Self::thought_to_record(thought, &embedding);
        self.backend
            .insert(THOUGHTS_TABLE, vec![record])
            .await
            .context("Failed to reinsert updated thought")?;
        Ok(())
    }

    /// Search for existing thoughts similar to `content`, classify them as
    /// corroborations or contradictions, update their confidence via EMA, and
    /// add bidirectional `evidence_chain` links.
    ///
    /// Returns an [`EvidenceCheckResult`] describing what was found.
    async fn apply_evidence_check(
        &self,
        new_thought_id: &str,
        content: &str,
    ) -> Result<crate::knowledge::types::EvidenceCheckResult> {
        use crate::knowledge::types::SearchMemoryRequest;

        // Search for semantically similar existing thoughts.
        let similar = self
            .search_memory(SearchMemoryRequest {
                query: content.to_string(),
                limit: 10,
                min_score: CONTRADICTION_THRESHOLD,
                category: None,
                sources: Some(vec!["thoughts".into()]),
                owner_id: None,
            })
            .await?;

        // Exclude the newly inserted thought itself.
        let similar_results: Vec<_> = similar
            .results
            .into_iter()
            .filter(|r| r.thought_id.as_deref() != Some(new_thought_id))
            .collect();

        let corroboration_result =
            fact_extractor::check_corroboration(&similar_results, CORROBORATION_THRESHOLD);
        let contradictions =
            fact_extractor::check_contradiction(content, &similar_results, CONTRADICTION_THRESHOLD);

        // Remove IDs that appear in both (corroboration wins).
        let contradiction_ids: Vec<String> = contradictions
            .into_iter()
            .filter(|id| !corroboration_result.corroborations.contains(id))
            .collect();

        let now = Utc::now().timestamp();

        // Update corroborated thoughts.
        for corr_id in &corroboration_result.corroborations {
            if let Some(mut t) = self.get_thought_internal(corr_id).await? {
                let old_conf = t.confidence;
                t.confidence = (EVIDENCE_EMA_ALPHA * (old_conf + 0.1)
                    + (1.0 - EVIDENCE_EMA_ALPHA) * old_conf)
                    .clamp(0.0, 1.0);
                t.reinforcement_count += 1;
                if !t.evidence_chain.contains(&new_thought_id.to_string()) {
                    t.evidence_chain.push(new_thought_id.to_string());
                }
                t.updated_at = now;
                if let Err(e) = self.replace_thought(&t).await {
                    tracing::warn!(id = %corr_id, "Failed to update corroborated thought: {}", e);
                }
            }
        }

        // Update contradicted thoughts.
        for contra_id in &contradiction_ids {
            if let Some(mut t) = self.get_thought_internal(contra_id).await? {
                let old_conf = t.confidence;
                t.confidence = (EVIDENCE_EMA_ALPHA * (old_conf - 0.1)
                    + (1.0 - EVIDENCE_EMA_ALPHA) * old_conf)
                    .clamp(0.0, 1.0);
                t.contradiction_count += 1;
                if !t.evidence_chain.contains(&new_thought_id.to_string()) {
                    t.evidence_chain.push(new_thought_id.to_string());
                }
                t.updated_at = now;
                if let Err(e) = self.replace_thought(&t).await {
                    tracing::warn!(id = %contra_id, "Failed to update contradicted thought: {}", e);
                }
            }
        }

        Ok(crate::knowledge::types::EvidenceCheckResult {
            corroborations: corroboration_result.corroborations,
            contradictions: contradiction_ids,
        })
    }

    /// Fetch a full `Thought` by ID (including soft-deleted records are excluded).
    async fn get_thought_internal(&self, id: &str) -> Result<Option<Thought>> {
        let filter = Filter::And(vec![
            Filter::Eq("id".into(), FieldValue::Utf8(Some(id.to_string()))),
            Filter::Eq("deleted".into(), FieldValue::Boolean(Some(false))),
        ]);
        let records = self
            .backend
            .query(THOUGHTS_TABLE, Some(&filter), Some(1))
            .await?;
        let mut thoughts = Self::records_to_thoughts(&records)?;
        Ok(thoughts.pop())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, BrainClient) {
        let temp = TempDir::new().unwrap();
        let lance_path = temp.path().join("brain.lance");
        let pks_path = temp.path().join("pks.db");
        let bks_path = temp.path().join("bks.db");

        let client = BrainClient::with_paths(
            lance_path.to_str().unwrap(),
            pks_path.to_str().unwrap(),
            bks_path.to_str().unwrap(),
        )
        .await
        .unwrap();

        (temp, client)
    }

    #[tokio::test]
    async fn test_capture_and_get() {
        let (_temp, mut client) = setup().await;

        let resp = client
            .capture_thought(CaptureThoughtRequest {
                content: "Decided to use PostgreSQL for auth service".into(),
                category: None,
                tags: Some(vec!["db".into()]),
                importance: Some(0.8),
                source: None,
                owner_id: None,
            })
            .await
            .unwrap();

        assert_eq!(resp.category, "decision");
        assert!(resp.tags.contains(&"db".to_string()));

        let thought = client.get_thought(&resp.id, None).await.unwrap();
        assert!(thought.is_some());
        let t = thought.unwrap();
        assert_eq!(t.category, "decision");
    }

    #[tokio::test]
    async fn test_search_memory() {
        let (_temp, mut client) = setup().await;

        client
            .capture_thought(CaptureThoughtRequest {
                content: "Rust is great for systems programming".into(),
                category: Some("insight".into()),
                tags: None,
                importance: None,
                source: None,
                owner_id: None,
            })
            .await
            .unwrap();

        let results = client
            .search_memory(SearchMemoryRequest {
                query: "programming languages".into(),
                limit: 10,
                min_score: 0.0,
                category: None,
                sources: None,
                owner_id: None,
            })
            .await
            .unwrap();

        assert!(!results.results.is_empty());
    }

    #[tokio::test]
    async fn test_delete_thought() {
        let (_temp, mut client) = setup().await;

        let resp = client
            .capture_thought(CaptureThoughtRequest {
                content: "Something to delete".into(),
                category: None,
                tags: None,
                importance: None,
                source: None,
                owner_id: None,
            })
            .await
            .unwrap();

        let del = client.delete_thought(&resp.id, None).await.unwrap();
        assert!(del.deleted);

        let thought = client.get_thought(&resp.id, None).await.unwrap();
        assert!(thought.is_none());
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let (_temp, client) = setup().await;
        let stats = client.memory_stats().await.unwrap();
        assert_eq!(stats.thoughts.total, 0);
    }
}
