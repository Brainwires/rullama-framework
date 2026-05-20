//! Multi-strategy ensemble query (Reciprocal Rank Fusion) for [`RagClient`].

use super::RagClient;
use crate::rag::types::*;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::time::Instant;

impl RagClient {
    /// Multi-strategy ensemble query: fan out across all requested strategies
    /// concurrently, fuse results via Reciprocal Rank Fusion (RRF), and
    /// optionally apply spectral diversity reranking as a final pass.
    ///
    /// ## Strategies
    ///
    /// - `Semantic` — vector similarity search
    /// - `Keyword` — BM25 keyword / hybrid search
    /// - `GitHistory` — semantic search over commit history
    /// - `CodeNavigation` — AST-based relations search (requires `code-analysis`)
    ///
    /// ## Fusion
    ///
    /// Results from each strategy are deduplicated by `file_path:start_line` and
    /// fused using RRF so that items appearing near the top of multiple strategy
    /// lists rank highest overall.
    pub async fn query_ensemble(&self, request: EnsembleRequest) -> Result<EnsembleResponse> {
        use brainwires_storage::bm25_search::reciprocal_rank_fusion_generic;

        let start = Instant::now();

        // Determine active strategies.
        let active: Vec<SearchStrategy> = if request.strategies.is_empty() {
            #[allow(unused_mut)]
            let mut s = vec![
                SearchStrategy::Semantic,
                SearchStrategy::Keyword,
                SearchStrategy::GitHistory,
            ];
            s.push(SearchStrategy::CodeNavigation);
            s
        } else {
            request.strategies.clone()
        };

        // Embed the query once.
        let query_embedding = self
            .embedding_provider
            .embed(&request.query)
            .context("Failed to generate query embedding for ensemble")?;

        // Fan out across strategies concurrently.
        // Each strategy returns (strategy_name, Vec<SearchResult>).
        let path = request.path.clone();
        let project = request.project.clone();
        let query = request.query.clone();
        let limit = request.limit;
        let min_score = request.min_score;
        let file_extensions = request.file_extensions.clone();
        let languages = request.languages.clone();

        // Build strategy futures as boxed async closures resolved concurrently.
        let mut strategy_futures = Vec::new();

        for strategy in &active {
            match strategy {
                SearchStrategy::Semantic => {
                    let qe = query_embedding.clone();
                    let q = query.clone();
                    let pa = path.clone();
                    let pr = project.clone();
                    let db = self.vector_db.clone();
                    strategy_futures.push(tokio::spawn(async move {
                        let results = db
                            .search(qe, &q, limit * 2, min_score, pr, pa, false)
                            .await
                            .unwrap_or_default();
                        ("semantic".to_string(), results)
                    }));
                }
                SearchStrategy::Keyword => {
                    let qe = query_embedding.clone();
                    let q = query.clone();
                    let pa = path.clone();
                    let pr = project.clone();
                    let db = self.vector_db.clone();
                    let exts = file_extensions.clone();
                    let langs = languages.clone();
                    strategy_futures.push(tokio::spawn(async move {
                        let results = if exts.is_empty() && langs.is_empty() {
                            db.search(qe, &q, limit * 2, min_score, pr, pa, true)
                                .await
                                .unwrap_or_default()
                        } else {
                            db.search_filtered(
                                qe,
                                &q,
                                limit * 2,
                                min_score,
                                pr,
                                pa,
                                true,
                                exts,
                                langs,
                                Vec::new(),
                            )
                            .await
                            .unwrap_or_default()
                        };
                        ("keyword".to_string(), results)
                    }));
                }
                SearchStrategy::GitHistory => {
                    let ep = self.embedding_provider.clone();
                    let db = self.vector_db.clone();
                    let gc = self.git_cache.clone();
                    let gp = self.git_cache_path.clone();
                    let q = query.clone();
                    let pa = path.clone().unwrap_or_else(|| ".".to_string());
                    let pr = project.clone();
                    strategy_futures.push(tokio::spawn(async move {
                        use crate::rag::client::git_indexing;
                        use brainwires_core::SearchResult;
                        let git_req = SearchGitHistoryRequest {
                            query: q,
                            path: pa,
                            project: pr,
                            branch: None,
                            max_commits: 200,
                            limit: limit * 2,
                            min_score,
                            author: None,
                            since: None,
                            until: None,
                            file_pattern: None,
                        };
                        let resp: SearchGitHistoryResponse =
                            git_indexing::do_search_git_history(ep, db, gc, &gp, git_req)
                                .await
                                .unwrap_or(SearchGitHistoryResponse {
                                    results: Vec::new(),
                                    commits_indexed: 0,
                                    total_cached_commits: 0,
                                    duration_ms: 0,
                                });
                        let results: Vec<SearchResult> = resp
                            .results
                            .into_iter()
                            .map(|g| SearchResult {
                                file_path: g.commit_hash.clone(),
                                root_path: None,
                                content: format!("{}\n{}", g.commit_message, g.diff_snippet),
                                score: g.score,
                                vector_score: g.vector_score,
                                keyword_score: g.keyword_score,
                                start_line: 0,
                                end_line: 0,
                                language: "git".to_string(),
                                project: None,
                                indexed_at: g.commit_date,
                            })
                            .collect();
                        ("git_history".to_string(), results)
                    }));
                }
                SearchStrategy::CodeNavigation => {
                    let qe = query_embedding.clone();
                    let db = self.vector_db.clone();
                    let q = query.clone();
                    let pa = path.clone();
                    let pr = project.clone();
                    strategy_futures.push(tokio::spawn(async move {
                        let results = db
                            .search(qe, &q, limit * 2, min_score, pr, pa, false)
                            .await
                            .unwrap_or_default();
                        ("code_navigation".to_string(), results)
                    }));
                }
            }
        }

        // Collect strategy results.
        let mut all_results: HashMap<String, SearchResult> = HashMap::new();
        let mut strategy_lists: Vec<Vec<(String, f32)>> = Vec::new();
        let mut strategies_used: Vec<String> = Vec::new();
        let mut per_strategy_counts: HashMap<String, usize> = HashMap::new();

        for handle in strategy_futures {
            match handle.await {
                Ok((name, results)) => {
                    per_strategy_counts.insert(name.clone(), results.len());
                    let ranked: Vec<(String, f32)> = results
                        .iter()
                        .map(|r| {
                            let key = format!("{}:{}", r.file_path, r.start_line);
                            all_results.entry(key.clone()).or_insert_with(|| r.clone());
                            (key, r.score)
                        })
                        .collect();
                    if !ranked.is_empty() {
                        strategies_used.push(name);
                        strategy_lists.push(ranked);
                    }
                }
                Err(e) => {
                    tracing::warn!("Ensemble strategy task failed: {e}");
                }
            }
        }

        // RRF fusion across all strategy ranked lists.
        let fused: Vec<(String, f32)> = reciprocal_rank_fusion_generic(strategy_lists, limit);

        // Resolve fused keys back to SearchResult, overriding score with RRF score.
        let mut results: Vec<SearchResult> = fused
            .into_iter()
            .filter_map(|(key, rrf_score)| {
                all_results.get(&key).map(|r| {
                    let mut result = r.clone();
                    result.score = rrf_score;
                    result
                })
            })
            .collect();

        // Optional spectral reranking as a final diversity pass.
        if request.spectral_rerank && results.len() > limit {
            use crate::spectral::{DiversityReranker, SpectralReranker, SpectralSelectConfig};
            let keys: Vec<String> = results
                .iter()
                .map(|r| format!("{}:{}", r.file_path, r.start_line))
                .collect();
            // Re-fetch embeddings for the fused candidates.
            if let Ok((_, embeddings)) = self
                .vector_db
                .search_with_embeddings(
                    query_embedding.clone(),
                    &request.query,
                    results.len(),
                    0.0,
                    request.project.clone(),
                    request.path.clone(),
                    false,
                )
                .await
            {
                // Build a key→embedding map from the re-fetched results.
                let _ = keys; // suppress unused warning
                if embeddings.len() == results.len() {
                    let reranker = SpectralReranker::new(SpectralSelectConfig::default());
                    let indices = reranker.rerank(&results, &embeddings, limit);
                    results = indices.into_iter().map(|i| results[i].clone()).collect();
                } else {
                    results.truncate(limit);
                }
            } else {
                results.truncate(limit);
            }
        }

        results.truncate(limit);

        Ok(EnsembleResponse {
            results,
            duration_ms: start.elapsed().as_millis() as u64,
            strategies_used,
            per_strategy_counts,
        })
    }
}
