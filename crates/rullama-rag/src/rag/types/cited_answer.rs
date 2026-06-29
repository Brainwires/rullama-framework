//! Traceable RAG answers: `CitedAnswer` pairs a final synthesised text
//! with the exact source spans the answer was drawn from.
//!
//! Consumers that need to surface citations (legal/compliance tools, RAG
//! UIs that link back to source code, evals that score grounding) build
//! a `CitedAnswer` from a `Vec<SearchResult>` via
//! [`CitedAnswer::from_search_results`] and render it with
//! [`CitedAnswer::render_markdown`].

use std::ops::Range;

use serde::{Deserialize, Serialize};

use rullama_core::SearchResult;

/// A single citation pointing back to the source span that backed part of
/// a [`CitedAnswer::text`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    /// Path the chunk was retrieved from (typically a file path relative
    /// to the indexed root).
    pub source_path: String,
    /// Byte range within the source, when known. RAG search results
    /// typically expose line ranges; the conversion in
    /// [`CitedAnswer::from_search_results`] uses
    /// `start_line..=end_line` as the span anchor (line numbers, not bytes)
    /// because byte offsets are not always available from the index.
    /// Consumers that need byte offsets resolve them by re-reading the
    /// source — the path + line range is enough to make that look-up.
    pub span: Range<usize>,
    /// Short excerpt from the source — the chunk content as the index
    /// stored it. Useful for UIs that want to show the cited passage
    /// inline without re-reading the file.
    pub snippet: String,
    /// Similarity score (0.0–1.0) from the index, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// A synthesised answer paired with the source spans that backed it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CitedAnswer {
    /// The final answer text — typically the LLM's synthesised response
    /// over the retrieved chunks, but `from_search_results` falls back
    /// to a stitched-together summary when no synthesis happened.
    pub text: String,
    /// Citations in retrieval order (highest-relevance first).
    pub citations: Vec<Citation>,
}

impl CitedAnswer {
    /// Build a `CitedAnswer` from a list of [`SearchResult`]s and a
    /// final synthesised `answer` text. The retrieval order is
    /// preserved as the citation order.
    pub fn from_search_results(answer: impl Into<String>, results: &[SearchResult]) -> Self {
        let citations = results
            .iter()
            .map(|r| Citation {
                source_path: r.file_path.clone(),
                span: r.start_line..r.end_line,
                snippet: r.content.clone(),
                score: Some(r.score),
            })
            .collect();
        Self {
            text: answer.into(),
            citations,
        }
    }

    /// Render the answer as a markdown string with footnote-style
    /// citations. Each citation gets a numbered footnote referencing the
    /// `source_path:start-end` location and the snippet body.
    pub fn render_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.text);
        if self.citations.is_empty() {
            return out;
        }
        out.push_str("\n\n");
        for (idx, c) in self.citations.iter().enumerate() {
            let n = idx + 1;
            out.push_str(&format!(
                "[^{n}]: `{}` (lines {}–{})\n",
                c.source_path, c.span.start, c.span.end
            ));
        }
        out
    }

    /// Whether the answer has at least one citation.
    pub fn has_citations(&self) -> bool {
        !self.citations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(path: &str, content: &str, start: usize, end: usize, score: f32) -> SearchResult {
        SearchResult {
            file_path: path.to_string(),
            root_path: None,
            content: content.to_string(),
            score,
            vector_score: score,
            keyword_score: None,
            start_line: start,
            end_line: end,
            language: "rust".to_string(),
            project: None,
            indexed_at: 0,
        }
    }

    #[test]
    fn from_search_results_preserves_order() {
        let results = vec![
            r("a.rs", "fn one() {}", 1, 1, 0.95),
            r("b.rs", "fn two() {}", 10, 12, 0.80),
        ];
        let cited = CitedAnswer::from_search_results("Two functions:", &results);
        assert_eq!(cited.text, "Two functions:");
        assert_eq!(cited.citations.len(), 2);
        assert_eq!(cited.citations[0].source_path, "a.rs");
        assert_eq!(cited.citations[0].span, 1..1);
        assert_eq!(cited.citations[0].score, Some(0.95));
        assert_eq!(cited.citations[1].source_path, "b.rs");
        assert_eq!(cited.citations[1].span, 10..12);
    }

    #[test]
    fn render_markdown_emits_footnotes() {
        let cited = CitedAnswer::from_search_results(
            "An answer.",
            &[r("docs/penguins.md", "penguins are flightless", 3, 5, 0.9)],
        );
        let md = cited.render_markdown();
        assert!(md.contains("An answer."));
        assert!(md.contains("[^1]: `docs/penguins.md` (lines 3–5)"));
    }

    #[test]
    fn empty_citations_skip_footnotes() {
        let cited = CitedAnswer {
            text: "Hello.".to_string(),
            citations: vec![],
        };
        assert_eq!(cited.render_markdown(), "Hello.");
        assert!(!cited.has_citations());
    }
}
