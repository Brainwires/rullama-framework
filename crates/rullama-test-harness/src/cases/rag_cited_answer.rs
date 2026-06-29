//! Tier-A `feature.rag.search_returns_traceable_citations`: build a
//! synthetic `Vec<SearchResult>`, run `CitedAnswer::from_search_results`,
//! and verify the citations point to the right source paths and line
//! ranges, in retrieval order, with the snippet body preserved. Also
//! verifies the markdown renderer emits one footnote per citation.

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::SearchResult;
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_rag::rag::types::CitedAnswer;

use crate::registry::TierACase;

pub struct RagSearchReturnsTraceableCitations;

fn result(path: &str, content: &str, start: usize, end: usize, score: f32) -> SearchResult {
    SearchResult {
        file_path: path.to_string(),
        root_path: None,
        content: content.to_string(),
        score,
        vector_score: score,
        keyword_score: None,
        start_line: start,
        end_line: end,
        language: "markdown".to_string(),
        project: None,
        indexed_at: 0,
    }
}

#[async_trait]
impl EvaluationCase for RagSearchReturnsTraceableCitations {
    fn name(&self) -> &str {
        "feature.rag.search_returns_traceable_citations"
    }
    fn category(&self) -> &str {
        "feature"
    }
    async fn run(&self, trial_id: usize) -> Result<TrialResult> {
        let started = std::time::Instant::now();
        let results = vec![
            result(
                "docs/penguins.md",
                "Penguins are flightless aquatic birds native to the Southern Hemisphere.",
                3,
                5,
                0.92,
            ),
            result(
                "docs/birds_overview.md",
                "Most birds can fly; notable exceptions include penguins, ostriches, and kiwis.",
                10,
                12,
                0.80,
            ),
        ];

        let cited = CitedAnswer::from_search_results(
            "Penguins are flightless birds that live in the Southern Hemisphere.",
            &results,
        );

        let elapsed = started.elapsed().as_millis() as u64;

        if cited.citations.len() != 2 {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("expected 2 citations, got {}", cited.citations.len()),
            ));
        }
        // Retrieval order preserved.
        if cited.citations[0].source_path != "docs/penguins.md" {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("first citation path: {}", cited.citations[0].source_path),
            ));
        }
        if cited.citations[0].span != (3..5) {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("first citation span: {:?}", cited.citations[0].span),
            ));
        }
        if !cited.citations[0].snippet.contains("Penguins are flightless") {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "first citation snippet missing penguin text",
            ));
        }
        if cited.citations[0].score != Some(0.92) {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                "first citation score not preserved",
            ));
        }

        let md = cited.render_markdown();
        if !md.contains("[^1]: `docs/penguins.md` (lines 3–5)") {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("markdown render missing penguin footnote: {md}"),
            ));
        }
        if !md.contains("[^2]: `docs/birds_overview.md` (lines 10–12)") {
            return Ok(TrialResult::failure(
                trial_id,
                elapsed,
                format!("markdown render missing birds_overview footnote: {md}"),
            ));
        }

        Ok(TrialResult::success(trial_id, elapsed)
            .with_meta("n_citations", cited.citations.len() as u64))
    }
}

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::rag_cited_answer::RagSearchReturnsTraceableCitations",
        crate_name: "rullama-rag",
        description: "CitedAnswer::from_search_results yields citations in retrieval order with paths, spans, snippets, and markdown footnotes",
        factory: || Box::new(RagSearchReturnsTraceableCitations),
    }
}
