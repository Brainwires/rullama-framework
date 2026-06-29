//! Diversity / relevance reranking via pluggable reranker strategies for [`RagClient`].
//!
//! Requires the `spectral` feature.

use super::RagClient;
use crate::rag::types::*;
use anyhow::{Context, Result};
use std::time::Instant;

impl RagClient {
    /// Query the indexed codebase with pluggable diversity/relevance reranking.
    ///
    /// This oversamples candidates (3× the limit), then applies the chosen
    /// reranker to select the final result set.  Pass `None` to use the default
    /// spectral reranker with its default configuration.
    ///
    /// ## Reranker options
    ///
    /// - [`RerankerKind::Spectral`](crate::spectral::RerankerKind::Spectral) — greedy log-det maximization (diversity-focused)
    /// - [`RerankerKind::CrossEncoder`](crate::spectral::RerankerKind::CrossEncoder) — query-document cosine blend (relevance-focused)
    /// - [`RerankerKind::Both`](crate::spectral::RerankerKind::Both) — spectral first, then cross-encoder on the selected subset
    ///
    /// Requires the `spectral` feature.
    pub async fn query_diverse(
        &self,
        request: QueryRequest,
        reranker: Option<crate::spectral::RerankerKind>,
    ) -> Result<QueryResponse> {
        use crate::spectral::{
            CrossEncoderReranker, DiversityReranker, RerankerKind, SpectralReranker,
        };

        request.validate().map_err(|e| anyhow::anyhow!(e))?;
        self.check_path_not_dirty(request.path.as_deref()).await?;

        let start = Instant::now();

        // Determine final_k from the reranker config or the request limit.
        let final_k = match &reranker {
            Some(RerankerKind::Spectral(cfg)) => cfg.k.unwrap_or(request.limit),
            Some(RerankerKind::Both { spectral, .. }) => spectral.k.unwrap_or(request.limit),
            _ => request.limit,
        };

        // Oversample: retrieve 3× candidates for the reranker to select from.
        let oversample_limit = final_k * 3;

        let query_embedding = self
            .embedding_provider
            .embed(&request.query)
            .context("Failed to generate query embedding")?;

        let original_threshold = request.min_score;
        let mut threshold_used = original_threshold;
        let mut threshold_lowered = false;

        // Search with embeddings so we can pass them to the reranker.
        let (mut candidates, mut embeddings) = self
            .vector_db
            .search_with_embeddings(
                query_embedding.clone(),
                &request.query,
                oversample_limit,
                threshold_used,
                request.project.clone(),
                request.path.clone(),
                request.hybrid,
            )
            .await
            .context("Failed to search with embeddings")?;

        // Adaptive threshold lowering if no results.
        if candidates.is_empty() && original_threshold > 0.3 {
            let fallback_thresholds = [0.6, 0.5, 0.4, 0.3];
            for &threshold in &fallback_thresholds {
                if threshold >= original_threshold {
                    continue;
                }
                let (c, e) = self
                    .vector_db
                    .search_with_embeddings(
                        query_embedding.clone(),
                        &request.query,
                        oversample_limit,
                        threshold,
                        request.project.clone(),
                        request.path.clone(),
                        request.hybrid,
                    )
                    .await
                    .context("Failed to search with embeddings")?;
                if !c.is_empty() {
                    candidates = c;
                    embeddings = e;
                    threshold_used = threshold;
                    threshold_lowered = true;
                    break;
                }
            }
        }

        let has_enough = candidates.len() > final_k && embeddings.iter().all(|e| !e.is_empty());

        let results = if has_enough {
            match reranker {
                None | Some(RerankerKind::Spectral(_)) => {
                    let spectral_cfg = match reranker {
                        Some(RerankerKind::Spectral(cfg)) => cfg,
                        _ => crate::spectral::SpectralSelectConfig::default(),
                    };
                    if candidates.len() >= spectral_cfg.min_candidates {
                        let r = SpectralReranker::new(spectral_cfg);
                        let indices = r.rerank(&candidates, &embeddings, final_k);
                        indices.into_iter().map(|i| candidates[i].clone()).collect()
                    } else {
                        candidates.truncate(final_k);
                        candidates
                    }
                }
                Some(RerankerKind::CrossEncoder(mut ce_cfg)) => {
                    // Inject query embedding if caller left it empty.
                    if ce_cfg.query_embedding.is_empty() {
                        ce_cfg.query_embedding = query_embedding.clone();
                    }
                    let r = CrossEncoderReranker::new(ce_cfg);
                    let indices = r.rerank(&candidates, &embeddings, final_k);
                    indices.into_iter().map(|i| candidates[i].clone()).collect()
                }
                Some(RerankerKind::Both {
                    spectral,
                    mut cross_encoder,
                }) => {
                    // Pass 1: spectral diversity selection.
                    let spectral_k = spectral.k.unwrap_or(final_k * 2).max(final_k);
                    let indices1 = if candidates.len() >= spectral.min_candidates {
                        let r = SpectralReranker::new(spectral);
                        r.rerank(&candidates, &embeddings, spectral_k)
                    } else {
                        (0..candidates.len().min(spectral_k)).collect()
                    };

                    // Build intermediate candidate/embedding slices.
                    let mid_candidates: Vec<_> =
                        indices1.iter().map(|&i| candidates[i].clone()).collect();
                    let mid_embeddings: Vec<_> =
                        indices1.iter().map(|&i| embeddings[i].clone()).collect();

                    // Pass 2: cross-encoder relevance ordering.
                    if cross_encoder.query_embedding.is_empty() {
                        cross_encoder.query_embedding = query_embedding.clone();
                    }
                    let r = CrossEncoderReranker::new(cross_encoder);
                    let indices2 = r.rerank(&mid_candidates, &mid_embeddings, final_k);
                    indices2
                        .into_iter()
                        .map(|i| mid_candidates[i].clone())
                        .collect()
                }
            }
        } else {
            candidates.truncate(final_k);
            candidates
        };

        Ok(QueryResponse {
            results,
            duration_ms: start.elapsed().as_millis() as u64,
            threshold_used,
            threshold_lowered,
        })
    }
}
