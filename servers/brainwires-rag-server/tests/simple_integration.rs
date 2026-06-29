/// Simple integration tests for basic server functionality
use anyhow::Result;
use brainwires_rag::rag::config::Config;
use brainwires_rag::rag::types::{
    AdvancedSearchRequest, ClearRequest, QueryRequest, StatisticsRequest,
};
use brainwires_rag_server::mcp_server::RagMcpServer;
use std::path::Path;
use tempfile::TempDir;

/// Helper: build a server backed by isolated temp dirs for db + caches.
async fn test_server(db_dir: &Path, cache_dir: &Path) -> Result<RagMcpServer> {
    let mut config = Config::default();
    config.vector_db.lancedb_path = db_dir.to_path_buf();
    config.cache.hash_cache_path = cache_dir.join("hash_cache.json");
    config.cache.git_cache_path = cache_dir.join("git_cache.json");
    RagMcpServer::with_config(config).await
}

/// Helper: write a single-file Rust fixture that contains a unique magic symbol.
fn write_rust_fixture(root: &Path) -> Result<()> {
    let src_dir = root.join("src");
    std::fs::create_dir_all(&src_dir)?;
    std::fs::write(
        src_dir.join("lib.rs"),
        "/// Returns the answer to life, the universe, and everything.\n\
         pub fn brainwires_rag_magic_word_42() -> u32 { 42 }\n",
    )?;
    Ok(())
}

/// Helper: drive a full index on a fixture directory.
async fn index_fixture(server: &RagMcpServer, root: &Path, project: &str) -> Result<()> {
    let normalized = RagMcpServer::normalize_path(&root.to_string_lossy())?;
    let response = server
        .do_index(
            normalized,
            Some(project.to_string()),
            vec![],
            vec![],
            1_048_576,
            None,
            None,
            None,
        )
        .await?;
    assert!(
        response.files_indexed > 0,
        "fixture did not produce any indexed files: {:?}",
        response
    );
    assert!(
        response.chunks_created > 0,
        "fixture did not produce any chunks: {:?}",
        response
    );
    Ok(())
}

#[tokio::test]
async fn test_server_creation_with_config() -> Result<()> {
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    let mut config = Config::default();
    config.vector_db.lancedb_path = db_dir.path().to_path_buf();
    config.cache.hash_cache_path = cache_dir.path().join("hash_cache.json");
    config.cache.git_cache_path = cache_dir.path().join("git_cache.json");

    let server = RagMcpServer::with_config(config).await?;

    // Verify server was created successfully
    assert!(std::mem::size_of_val(&server) > 0);

    Ok(())
}

#[tokio::test]
async fn test_server_creation_with_defaults() -> Result<()> {
    // This should work with default configuration
    let server = RagMcpServer::new().await;

    // Server creation should succeed
    assert!(server.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_path_normalization() -> Result<()> {
    // Test path normalization with current directory
    let normalized = RagMcpServer::normalize_path(".")?;
    assert!(normalized.len() > 1);
    assert!(normalized.starts_with('/') || normalized.chars().nth(1) == Some(':'));

    Ok(())
}

#[tokio::test]
async fn test_config_with_custom_batch_size() -> Result<()> {
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    let mut config = Config::default();
    config.vector_db.lancedb_path = db_dir.path().to_path_buf();
    config.cache.hash_cache_path = cache_dir.path().join("hash_cache.json");
    config.cache.git_cache_path = cache_dir.path().join("git_cache.json");
    config.embedding.batch_size = 64;
    config.embedding.timeout_secs = 60;

    let server = RagMcpServer::with_config(config).await?;

    // Verify server was created with custom config
    assert!(std::mem::size_of_val(&server) > 0);

    Ok(())
}

#[tokio::test]
async fn test_full_indexing_workflow() -> Result<()> {
    let codebase_dir = TempDir::new()?;
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    // Create a simple test file
    let src_dir = codebase_dir.path().join("src");
    std::fs::create_dir_all(&src_dir)?;
    std::fs::write(src_dir.join("test.rs"), "fn main() { println!(\"test\"); }")?;

    // Create server
    let mut config = Config::default();
    config.vector_db.lancedb_path = db_dir.path().to_path_buf();
    config.cache.hash_cache_path = cache_dir.path().join("hash_cache.json");
    config.cache.git_cache_path = cache_dir.path().join("git_cache.json");

    let server = RagMcpServer::with_config(config).await?;

    // Test path normalization
    let normalized_path = RagMcpServer::normalize_path(&codebase_dir.path().to_string_lossy())?;
    assert!(!normalized_path.is_empty());

    // Test indexing (using the public do_index method)
    let index_response = server
        .do_index(
            normalized_path,
            Some("test_project".to_string()),
            vec![],
            vec![],
            1_048_576,
            None,
            None,
            None, // cancel_token
        )
        .await?;

    // Verify basic indexing worked
    assert!(index_response.files_indexed > 0);
    assert!(index_response.chunks_created > 0);

    Ok(())
}

// ---------------------------------------------------------------------------
// Extended lifecycle tests: index → query → stats → clear on isolated stores.
// ---------------------------------------------------------------------------

/// Indexing a tiny fixture then querying for a unique symbol should return at
/// least one hit whose content includes that symbol. We accept the adaptive
/// threshold lowering that the client performs and use a permissive
/// `min_score` so the test is robust across embedding-model variations.
#[tokio::test]
async fn test_index_then_query_returns_hit() -> Result<()> {
    let codebase_dir = TempDir::new()?;
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    write_rust_fixture(codebase_dir.path())?;

    let server = test_server(db_dir.path(), cache_dir.path()).await?;
    index_fixture(&server, codebase_dir.path(), "lifecycle_query").await?;

    // Query for the unique symbol. Use a low min_score to accept BM25 /
    // keyword hits even when the embedding similarity is modest.
    let response = server
        .client()
        .query_codebase(QueryRequest {
            query: "brainwires_rag_magic_word_42 function returning 42".to_string(),
            path: None,
            project: Some("lifecycle_query".to_string()),
            limit: 10,
            min_score: 0.0,
            hybrid: true,
        })
        .await?;

    assert!(
        !response.results.is_empty(),
        "query returned no results (duration_ms={}, threshold_used={})",
        response.duration_ms,
        response.threshold_used,
    );

    let any_match = response
        .results
        .iter()
        .any(|r| r.content.contains("brainwires_rag_magic_word_42"));
    assert!(
        any_match,
        "no result contained the unique symbol; got: {:#?}",
        response
            .results
            .iter()
            .map(|r| &r.file_path)
            .collect::<Vec<_>>()
    );

    Ok(())
}

/// After indexing, statistics should report at least one chunk; after
/// `clear_index`, the store should report zero chunks and stay usable.
#[tokio::test]
async fn test_clear_index_empties_store() -> Result<()> {
    let codebase_dir = TempDir::new()?;
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    write_rust_fixture(codebase_dir.path())?;

    let server = test_server(db_dir.path(), cache_dir.path()).await?;
    index_fixture(&server, codebase_dir.path(), "lifecycle_clear").await?;

    // Precondition: stats should show > 0 chunks.
    let before = server.client().get_statistics().await?;
    assert!(
        before.total_chunks > 0,
        "expected > 0 chunks after indexing, got {:?}",
        before
    );

    // Clear and verify success.
    let clear = server.client().clear_index().await?;
    assert!(clear.success, "clear_index failed: {}", clear.message);

    // Postcondition: stats must report an empty index.
    let after = server.client().get_statistics().await?;
    assert_eq!(
        after.total_chunks, 0,
        "expected 0 chunks after clear, got {:?}",
        after
    );
    assert_eq!(
        after.total_files, 0,
        "expected 0 files after clear, got {:?}",
        after
    );
    assert!(
        after.language_breakdown.is_empty(),
        "expected empty language breakdown after clear, got {:?}",
        after.language_breakdown
    );

    Ok(())
}

/// Querying an empty (never-indexed) store must return an empty result set
/// without panicking or producing an error.
#[tokio::test]
async fn test_query_on_empty_index_returns_empty() -> Result<()> {
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    let server = test_server(db_dir.path(), cache_dir.path()).await?;

    let response = server
        .client()
        .query_codebase(QueryRequest {
            query: "anything at all".to_string(),
            path: None,
            project: None,
            limit: 10,
            min_score: 0.0,
            hybrid: true,
        })
        .await?;

    assert!(
        response.results.is_empty(),
        "expected empty results on empty index, got {} hits",
        response.results.len()
    );

    // Statistics should also show an empty store.
    let stats = server.client().get_statistics().await?;
    assert_eq!(stats.total_chunks, 0);
    assert_eq!(stats.total_files, 0);

    // The MCP-facing statistics tool wrapper should also work against an
    // empty store (exercises the JSON serialization path).
    // We call the client directly; the MCP handler merely wraps it.
    let _ = StatisticsRequest {};
    let _ = ClearRequest {};

    Ok(())
}

/// `search_by_filters` should restrict results by file extension. We index a
/// fixture containing both a Rust file and a Python file with similar
/// semantics, then assert that filtering by extension narrows the hit set.
#[tokio::test]
async fn test_search_by_filters_restricts_to_extension() -> Result<()> {
    let codebase_dir = TempDir::new()?;
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    // Two files, each with a unique per-language marker so we can tell them
    // apart in the returned content.
    let src_dir = codebase_dir.path().join("src");
    std::fs::create_dir_all(&src_dir)?;
    std::fs::write(
        src_dir.join("foo.rs"),
        "/// Rust-side marker.\n\
         pub fn brainwires_filter_marker_rust() -> u32 { 7 }\n",
    )?;
    std::fs::write(
        src_dir.join("foo.py"),
        "# Python-side marker.\n\
         def brainwires_filter_marker_python():\n    return 7\n",
    )?;

    let server = test_server(db_dir.path(), cache_dir.path()).await?;
    index_fixture(&server, codebase_dir.path(), "lifecycle_filter").await?;

    // Restrict to Rust.
    let rust_only = server
        .client()
        .search_with_filters(AdvancedSearchRequest {
            query: "brainwires_filter_marker function returning 7".to_string(),
            path: None,
            project: Some("lifecycle_filter".to_string()),
            limit: 10,
            min_score: 0.0,
            file_extensions: vec!["rs".to_string()],
            languages: vec![],
            path_patterns: vec![],
        })
        .await?;
    assert!(
        !rust_only.results.is_empty(),
        "rust-filtered search returned no results"
    );
    for r in &rust_only.results {
        assert!(
            r.file_path.ends_with(".rs"),
            "rust filter leaked a non-rust result: {}",
            r.file_path
        );
    }

    // Restrict to Python.
    let py_only = server
        .client()
        .search_with_filters(AdvancedSearchRequest {
            query: "brainwires_filter_marker function returning 7".to_string(),
            path: None,
            project: Some("lifecycle_filter".to_string()),
            limit: 10,
            min_score: 0.0,
            file_extensions: vec!["py".to_string()],
            languages: vec![],
            path_patterns: vec![],
        })
        .await?;
    assert!(
        !py_only.results.is_empty(),
        "python-filtered search returned no results"
    );
    for r in &py_only.results {
        assert!(
            r.file_path.ends_with(".py"),
            "python filter leaked a non-python result: {}",
            r.file_path
        );
    }

    Ok(())
}

/// Statistics should include Rust in the language breakdown after indexing
/// a Rust-only fixture.
#[tokio::test]
async fn test_get_statistics_reports_language() -> Result<()> {
    let codebase_dir = TempDir::new()?;
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    write_rust_fixture(codebase_dir.path())?;

    let server = test_server(db_dir.path(), cache_dir.path()).await?;
    index_fixture(&server, codebase_dir.path(), "lifecycle_stats").await?;

    let stats = server.client().get_statistics().await?;
    assert!(
        !stats.language_breakdown.is_empty(),
        "expected at least one language entry, got {:?}",
        stats
    );

    let has_rust = stats
        .language_breakdown
        .iter()
        .any(|ls| ls.language.eq_ignore_ascii_case("rust"));
    assert!(
        has_rust,
        "expected Rust in language breakdown, got {:?}",
        stats.language_breakdown
    );

    Ok(())
}
