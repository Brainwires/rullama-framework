//! Shared BM25 helpers for vector database backends that use client-side
//! keyword scoring for hybrid search.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Document frequency statistics for IDF calculation.
#[derive(Debug, Clone, Default)]
pub struct IdfStats {
    /// Total number of documents in corpus.
    pub total_docs: usize,
    /// Term -> number of documents containing that term.
    pub doc_frequencies: HashMap<String, usize>,
}

/// Shared IDF statistics wrapped for concurrent access.
pub type SharedIdfStats = Arc<RwLock<IdfStats>>;

/// Create a new shared IDF stats instance.
pub fn new_shared_idf_stats() -> SharedIdfStats {
    Arc::new(RwLock::new(IdfStats::default()))
}

/// Tokenize text into lowercase whitespace-delimited terms.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(String::from)
        .collect()
}

/// Calculate BM25 score for a query against a document's content.
///
/// Uses the shared IDF statistics for term weighting. Returns a score
/// clamped to \[0, 1\].
pub async fn calculate_bm25_score(idf_stats: &SharedIdfStats, query: &str, content: &str) -> f32 {
    let query_terms = tokenize(query);
    if query_terms.is_empty() {
        return 0.0;
    }

    let content_terms = tokenize(content);
    let content_len = content_terms.len() as f32;

    let stats = idf_stats.read().await;
    let total_docs = stats.total_docs as f32;

    // BM25 parameters
    let k1 = 1.5;
    let b = 0.75;
    let avg_doc_len = 100.0;

    let mut score = 0.0;

    for term in &query_terms {
        let tf = content_terms.iter().filter(|t| t == &term).count() as f32;

        if tf > 0.0 {
            let doc_freq = stats.doc_frequencies.get(term).copied().unwrap_or(1) as f32;
            let idf = ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
            let norm = 1.0 - b + b * (content_len / avg_doc_len);
            let term_score = idf * (tf * (k1 + 1.0)) / (tf + k1 * norm);
            score += term_score;
        }
    }

    let normalized_score = score / query_terms.len() as f32;
    normalized_score.clamp(0.0, 1.0)
}

/// Update IDF statistics from a set of document contents.
///
/// Replaces the current stats with fresh calculations from the provided
/// corpus.
pub async fn update_idf_stats(idf_stats: &SharedIdfStats, documents: &[String]) {
    let mut doc_frequencies: HashMap<String, usize> = HashMap::new();
    let total_docs = documents.len();

    for content in documents {
        let terms = tokenize(content);
        let unique_terms: std::collections::HashSet<String> = terms.into_iter().collect();
        for term in unique_terms {
            *doc_frequencies.entry(term).or_insert(0) += 1;
        }
    }

    let mut stats = idf_stats.write().await;
    stats.total_docs = total_docs;
    stats.doc_frequencies = doc_frequencies;
}

/// Combine vector and keyword scores for hybrid search.
///
/// Default weighting: 70% vector + 30% keyword.
pub fn combine_scores(vector_score: f32, keyword_score: f32) -> f32 {
    (vector_score * 0.7) + (keyword_score * 0.3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello World Foo");
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_special_chars() {
        let tokens = tokenize("fn main() { println!(\"hello\"); }");
        // split_whitespace keeps punctuation attached to tokens
        assert_eq!(
            tokens,
            vec!["fn", "main()", "{", "println!(\"hello\");", "}"]
        );
    }

    #[tokio::test]
    async fn test_bm25_score_zero_for_no_match() {
        let stats = new_shared_idf_stats();
        update_idf_stats(&stats, &["some document content".to_string()]).await;

        let score = calculate_bm25_score(&stats, "zzzznonexistent", "some document content").await;
        assert_eq!(score, 0.0);
    }

    #[tokio::test]
    async fn test_bm25_score_positive_for_match() {
        let stats = new_shared_idf_stats();
        update_idf_stats(
            &stats,
            &[
                "hello world rust programming".to_string(),
                "goodbye world python scripting".to_string(),
            ],
        )
        .await;

        let score = calculate_bm25_score(&stats, "hello", "hello world rust programming").await;
        assert!(score > 0.0, "Expected positive score, got {}", score);
    }

    #[tokio::test]
    async fn test_bm25_score_empty_query() {
        let stats = new_shared_idf_stats();
        update_idf_stats(&stats, &["some content".to_string()]).await;

        let score = calculate_bm25_score(&stats, "", "some content").await;
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_combine_scores() {
        let combined = combine_scores(1.0, 1.0);
        assert!((combined - 1.0).abs() < f32::EPSILON);

        let combined = combine_scores(1.0, 0.0);
        assert!((combined - 0.7).abs() < f32::EPSILON);

        let combined = combine_scores(0.0, 1.0);
        assert!((combined - 0.3).abs() < f32::EPSILON);

        let combined = combine_scores(0.0, 0.0);
        assert_eq!(combined, 0.0);
    }

    #[tokio::test]
    async fn test_update_idf_stats() {
        let stats = new_shared_idf_stats();

        let documents = vec![
            "hello world".to_string(),
            "hello rust".to_string(),
            "goodbye world".to_string(),
        ];
        update_idf_stats(&stats, &documents).await;

        let s = stats.read().await;
        assert_eq!(s.total_docs, 3);
        assert_eq!(s.doc_frequencies.get("hello"), Some(&2));
        assert_eq!(s.doc_frequencies.get("world"), Some(&2));
        assert_eq!(s.doc_frequencies.get("rust"), Some(&1));
        assert_eq!(s.doc_frequencies.get("goodbye"), Some(&1));
    }

    #[test]
    fn test_new_shared_idf_stats() {
        let stats = new_shared_idf_stats();
        let s = stats.try_read().unwrap();
        assert_eq!(s.total_docs, 0);
        assert!(s.doc_frequencies.is_empty());
    }
}
