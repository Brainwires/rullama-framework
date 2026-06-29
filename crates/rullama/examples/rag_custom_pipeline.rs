//! Example: RAG pipeline with custom chunking and scoring
//!
//! Shows how to plug in a custom Chunker and SearchScorer into the RAG system.
//!
//! Run: cargo run -p rullama --example rag_custom_pipeline --features rag

use rullama::rag::indexer::{Chunker, CodeChunk, FileInfo};
use rullama::rag::types::ChunkMetadata;
use rullama::storage::bm25_search::{BM25Result, SearchScorer};
use std::time::{SystemTime, UNIX_EPOCH};

// --- Custom Chunker ---

/// A simple sentence-based chunker that splits on period boundaries.
/// Researchers can replace this with semantic chunking, ML-based segmentation, etc.
struct SentenceChunker {
    max_sentences: usize,
}

impl Chunker for SentenceChunker {
    fn chunk_file(&self, file_info: &FileInfo) -> Vec<CodeChunk> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Split content into sentence groups
        let sentences: Vec<&str> = file_info.content.split('.').collect();
        let mut chunks = Vec::new();

        for (i, group) in sentences.chunks(self.max_sentences).enumerate() {
            let content = group.join(".");
            if content.trim().is_empty() {
                continue;
            }

            chunks.push(CodeChunk {
                content,
                metadata: ChunkMetadata {
                    file_path: file_info.relative_path.clone(),
                    root_path: Some(file_info.root_path.clone()),
                    project: file_info.project.clone(),
                    start_line: i * self.max_sentences + 1,
                    end_line: (i + 1) * self.max_sentences,
                    language: file_info.language.clone(),
                    extension: file_info.extension.clone(),
                    file_hash: file_info.hash.clone(),
                    indexed_at: timestamp,
                },
            });
        }

        chunks
    }
}

// --- Custom Search Scorer ---

/// Weighted linear combination scorer instead of RRF.
/// Allows tuning the balance between vector and keyword results.
struct WeightedScorer {
    vector_weight: f32,
    keyword_weight: f32,
}

impl SearchScorer for WeightedScorer {
    fn fuse(
        &self,
        vector_results: Vec<(String, f32)>,
        bm25_results: Vec<BM25Result>,
        limit: usize,
    ) -> Vec<(String, f32)> {
        use std::collections::HashMap;

        let mut scores: HashMap<String, f32> = HashMap::new();

        // Normalize vector scores (already 0-1) and weight them
        for (id, score) in &vector_results {
            *scores.entry(id.clone()).or_default() += score * self.vector_weight;
        }

        // Normalize BM25 scores to 0-1 range, then weight
        let max_bm25 = bm25_results.iter().map(|r| r.score).fold(0.0f32, f32::max);

        if max_bm25 > 0.0 {
            for result in &bm25_results {
                let normalized = result.score / max_bm25;
                *scores.entry(result.string_id.clone()).or_default() +=
                    normalized * self.keyword_weight;
            }
        }

        let mut combined: Vec<(String, f32)> = scores.into_iter().collect();
        combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(limit);
        combined
    }
}

fn main() {
    // Demonstrate custom chunker
    let chunker = SentenceChunker { max_sentences: 3 };
    let file_info = FileInfo {
        path: "example.rs".into(),
        relative_path: "example.rs".to_string(),
        root_path: "/project".to_string(),
        project: Some("demo".to_string()),
        extension: Some("rs".to_string()),
        language: Some("Rust".to_string()),
        content:
            "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
                .to_string(),
        hash: "abc123".to_string(),
    };

    let chunks = chunker.chunk_file(&file_info);
    println!("Custom chunker produced {} chunks:", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {:?}", i, chunk.content.trim());
    }

    // Demonstrate custom scorer
    let scorer = WeightedScorer {
        vector_weight: 0.7,
        keyword_weight: 0.3,
    };

    let vector_results = vec![
        ("file.rs:1".to_string(), 0.95),
        ("file.rs:10".to_string(), 0.80),
        ("file.rs:20".to_string(), 0.60),
    ];
    let bm25_results = vec![
        BM25Result {
            id: 2,
            string_id: "file.rs:10".to_string(),
            score: 12.5,
        },
        BM25Result {
            id: 4,
            string_id: "other.rs:1".to_string(),
            score: 10.0,
        },
        BM25Result {
            id: 1,
            string_id: "file.rs:1".to_string(),
            score: 5.0,
        },
    ];

    let fused = scorer.fuse(vector_results, bm25_results, 5);
    println!("\nWeighted fusion results (0.7 vector + 0.3 keyword):");
    for (id, score) in &fused {
        println!("  {}: combined score {:.4}", id, score);
    }

    // To use with the actual RAG system:
    //
    //   use std::sync::Arc;
    //   use rullama::rag::indexer::{CodeChunker, ChunkStrategy};
    //
    //   // Custom chunker
    //   let strategy = ChunkStrategy::Custom(Arc::new(SentenceChunker { max_sentences: 5 }));
    //   let chunker = CodeChunker::new(strategy);
    //
    //   // Custom scorer on LanceDatabase
    //   let db = LanceDatabase::with_path("/path/to/db").await?
    //       .with_scorer(Arc::new(WeightedScorer { vector_weight: 0.7, keyword_weight: 0.3 }));
}
