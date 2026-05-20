//! RAG Search — Codebase Indexing and Hybrid Semantic+Keyword Search
//!
//! Demonstrates how to initialize a `RagClient`, index a directory of
//! source files, and perform hybrid (vector + BM25 keyword) queries
//! against the indexed codebase.
//!
//! **Note:** This example requires a local codebase directory to index.
//! By default it indexes the brainwires-knowledge crate itself.
//!
//! Run:
//! ```sh
//! cargo run -p brainwires-knowledge --example rag_search --features rag
//! ```

use brainwires_rag::rag::client::RagClient;
use brainwires_rag::rag::types::{IndexRequest, QueryRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for visibility into what the RAG system is doing
    tracing_subscriber::fmt()
        .with_max_level(tracing_subscriber::filter::LevelFilter::INFO)
        .init();

    println!("=== Brainwires RAG Search Example ===\n");

    // ── Step 1: Create the RagClient ─────────────────────────────────
    println!("--- Step 1: Initialize RagClient ---\n");

    let client = RagClient::new().await?;
    println!("RagClient initialized successfully.\n");

    // ── Step 2: Index a codebase directory ────────────────────────────
    println!("--- Step 2: Index a Codebase ---\n");

    // Index this crate's own source code as a demonstration
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let src_dir = std::path::Path::new(&crate_dir).join("src");
    let index_path = if src_dir.exists() {
        src_dir.display().to_string()
    } else {
        // Fallback: index current directory
        ".".to_string()
    };

    println!("Indexing directory: {}", index_path);

    let index_request = IndexRequest {
        path: index_path.clone(),
        project: Some("rag-example".to_string()),
        include_patterns: vec!["**/*.rs".to_string()],
        exclude_patterns: vec!["**/target/**".to_string(), "**/node_modules/**".to_string()],
        max_file_size: 1_048_576, // 1 MB
    };

    let index_response = client.index_codebase(index_request).await?;
    println!("Indexing complete:");
    println!("  Mode:       {:?}", index_response.mode);
    println!("  Files:      {}", index_response.files_indexed);
    println!("  Chunks:     {}", index_response.chunks_created);
    println!("  Embeddings: {}", index_response.embeddings_generated);
    println!("  Duration:   {} ms", index_response.duration_ms);
    if !index_response.errors.is_empty() {
        println!("  Errors:     {}", index_response.errors.len());
        for err in &index_response.errors {
            println!("    - {}", err);
        }
    }
    println!();

    // ── Step 3: Check statistics ─────────────────────────────────────
    println!("--- Step 3: Index Statistics ---\n");

    let stats = client.get_statistics().await?;
    println!("Total files:      {}", stats.total_files);
    println!("Total chunks:     {}", stats.total_chunks);
    println!("Total embeddings: {}", stats.total_embeddings);
    println!("Database size:    {} bytes", stats.database_size_bytes);
    println!("Language breakdown:");
    for lang in &stats.language_breakdown {
        println!(
            "  {}: {} files, {} chunks",
            lang.language, lang.file_count, lang.chunk_count
        );
    }
    println!();

    // ── Step 4: Perform hybrid semantic + keyword queries ─────────────
    println!("--- Step 4: Semantic Search Queries ---\n");

    let queries = [
        "entity relationship graph traversal",
        "embedding vector search",
        "error handling and result types",
    ];

    for query_text in &queries {
        println!("Query: \"{}\"", query_text);

        let query = QueryRequest {
            query: query_text.to_string(),
            path: Some(index_path.clone()),
            project: Some("rag-example".to_string()),
            limit: 3,
            min_score: 0.5,
            hybrid: true, // combine vector similarity with BM25 keyword scoring
        };

        let response = client.query_codebase(query).await?;
        println!(
            "  Found {} results in {} ms (threshold: {:.2}, lowered: {})",
            response.results.len(),
            response.duration_ms,
            response.threshold_used,
            response.threshold_lowered
        );

        for (i, result) in response.results.iter().enumerate() {
            let preview = result.content.lines().take(2).collect::<Vec<_>>().join(" ");
            let preview = if preview.len() > 100 {
                format!("{}...", &preview[..100])
            } else {
                preview
            };

            println!(
                "  [{}] score={:.3} (vec={:.3}, kw={}) | {}:{}-{} | {}",
                i + 1,
                result.score,
                result.vector_score,
                result
                    .keyword_score
                    .map(|s| format!("{:.3}", s))
                    .unwrap_or_else(|| "n/a".into()),
                result.file_path,
                result.start_line,
                result.end_line,
                preview
            );
        }
        println!();
    }

    // ── Step 5: Vector-only search (disable hybrid) ──────────────────
    println!("--- Step 5: Vector-Only Search ---\n");

    let query = QueryRequest {
        query: "how are thoughts stored and retrieved".to_string(),
        path: Some(index_path.clone()),
        project: Some("rag-example".to_string()),
        limit: 3,
        min_score: 0.5,
        hybrid: false, // pure vector similarity, no keyword scoring
    };

    let response = client.query_codebase(query).await?;
    println!("Vector-only results ({} found):", response.results.len());
    for (i, result) in response.results.iter().enumerate() {
        println!(
            "  [{}] score={:.3} | {} (lines {}-{})",
            i + 1,
            result.score,
            result.file_path,
            result.start_line,
            result.end_line
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
