//! Example: Storage + RAG end-to-end pipeline
//!
//! Creates a LanceDB database, stores conversation messages via `MessageStore`,
//! then uses `RagClient` to index source files and perform semantic search.
//!
//! Run: cargo run -p brainwires --example storage_with_rag --features memory,rag

use anyhow::Result;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// Storage primitives come from `brainwires::storage::*`; the per-domain
// `MessageStore` lives under `brainwires::memory::*` (extracted in the
// brainwires-memory split).
use brainwires::memory::{MessageMetadata, MessageStore};
use brainwires::storage::{CachedEmbeddingProvider, LanceDatabase};

// RAG types come from `brainwires::rag::*`
use brainwires::rag::{IndexRequest, QueryRequest, RagClient};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing so we can see what the subsystems are doing.
    tracing_subscriber::fmt::init();

    // ── 1. Create a LanceDB database ────────────────────────────────────
    //
    // LanceDatabase implements both `StorageBackend` (for domain stores)
    // and `VectorDatabase` (for RAG), using a single shared connection.
    let db_path = "/tmp/brainwires_storage_rag_example";
    let db = Arc::new(LanceDatabase::new(db_path).await?);
    println!("Created LanceDB at {db_path}");

    // ── 2. Set up the embedding provider ────────────────────────────────
    //
    // CachedEmbeddingProvider wraps FastEmbedManager with an LRU cache
    // for repeated queries.
    let embeddings: Arc<CachedEmbeddingProvider> = Arc::new(CachedEmbeddingProvider::new()?);
    println!("Embedding provider ready (dim={})", embeddings.dimension());

    // ── 3. Store conversation messages ──────────────────────────────────
    //
    // MessageStore builds on top of any `StorageBackend`. It generates
    // vector embeddings for each message so you can later search
    // semantically ("find messages about X").
    let message_store = MessageStore::new(db.clone(), embeddings.clone());
    message_store.ensure_table().await?;

    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;

    let messages = vec![
        MessageMetadata {
            message_id: "msg-001".into(),
            conversation_id: "conv-1".into(),
            role: "user".into(),
            content: "How do I set up a Rust project with Cargo?".into(),
            token_count: Some(12),
            model_id: None,
            images: None,
            created_at: now,
            expires_at: None,
        },
        MessageMetadata {
            message_id: "msg-002".into(),
            conversation_id: "conv-1".into(),
            role: "assistant".into(),
            content:
                "Run `cargo init` in a new directory. This creates Cargo.toml and src/main.rs. \
                      Then use `cargo build` to compile and `cargo run` to execute."
                    .into(),
            token_count: Some(35),
            model_id: Some("demo-model".into()),
            images: None,
            created_at: now + 1,
            expires_at: None,
        },
        MessageMetadata {
            message_id: "msg-003".into(),
            conversation_id: "conv-1".into(),
            role: "user".into(),
            content: "What about adding dependencies and feature flags?".into(),
            token_count: Some(10),
            model_id: None,
            images: None,
            created_at: now + 2,
            expires_at: None,
        },
    ];

    message_store.add_batch(messages).await?;
    println!("Stored 3 conversation messages");

    // ── 4. Search messages semantically ─────────────────────────────────
    //
    // search() returns Vec<(MessageMetadata, f32)> — each result is a
    // message paired with its similarity score.
    let results = message_store.search("cargo project setup", 2, 0.0).await?;
    println!("\nSemantic search for \"cargo project setup\":");
    for (i, (msg, score)) in results.iter().enumerate() {
        let preview = &msg.content[..msg.content.len().min(80)];
        println!("  {}: [{:.2}] [{}] {}", i + 1, score, msg.role, preview);
    }

    // ── 5. Index source code with RagClient ─────────────────────────────
    //
    // RagClient wraps an embedding model + vector database and provides a
    // high-level API for codebase indexing and semantic search.
    println!("\nInitializing RAG client...");
    let rag = RagClient::new().await?;

    // Index the current crate's examples directory as a small demo corpus.
    let index_req = IndexRequest {
        path: env!("CARGO_MANIFEST_DIR").to_string(),
        project: Some("brainwires-facade".into()),
        include_patterns: vec!["**/*.rs".into()],
        exclude_patterns: vec!["**/target/**".into()],
        max_file_size: 1_048_576,
    };

    let index_resp = rag.index_codebase(index_req).await?;
    println!(
        "Indexed {} files ({} chunks) in {}ms",
        index_resp.files_indexed, index_resp.chunks_created, index_resp.duration_ms,
    );

    // ── 6. Query the indexed codebase ───────────────────────────────────
    let query_req = QueryRequest {
        query: "provider configuration and factory".into(),
        path: None,
        project: Some("brainwires-facade".into()),
        limit: 3,
        min_score: 0.3,
        hybrid: true,
    };

    let query_resp = rag.query_codebase(query_req).await?;
    println!("\nRAG query for \"provider configuration and factory\":");
    for result in &query_resp.results {
        println!(
            "  [{:.2}] {}:{}-{}",
            result.score, result.file_path, result.start_line, result.end_line,
        );
    }

    // ── Cleanup ─────────────────────────────────────────────────────────
    println!("\nDone! Database stored at {db_path}");
    // In production you would keep the database around; here we clean up.
    let _ = std::fs::remove_dir_all(db_path);

    Ok(())
}
