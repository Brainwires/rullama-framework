//! Integration tests for the SurrealDB backend.
//!
//! These tests require a running SurrealDB v2.x+ server. They are gated
//! behind the `surrealdb-backend` feature and marked `#[ignore]` so they
//! only run when explicitly requested (e.g. `cargo test --features
//! surrealdb-backend -- --ignored`).

#[cfg(feature = "surrealdb-backend")]
mod surrealdb_integration {
    use std::sync::Arc;

    use rullama_storage::databases::surrealdb::SurrealDatabase;
    use rullama_storage::databases::traits::{StorageBackend, VectorDatabase};
    use rullama_storage::databases::types::{FieldDef, FieldType, FieldValue, Filter};

    /// Helper: try to connect to SurrealDB, returning None if the server is
    /// not reachable.
    async fn try_connect() -> Option<SurrealDatabase> {
        let url = std::env::var("SURREALDB_URL").unwrap_or_else(|_| SurrealDatabase::default_url());
        let ns = std::env::var("SURREALDB_NS").unwrap_or_else(|_| "test".to_string());
        let db = std::env::var("SURREALDB_DB").unwrap_or_else(|_| "integration_test".to_string());
        SurrealDatabase::new(&url, &ns, &db).await.ok()
    }

    // ── StorageBackend CRUD tests ────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires a running SurrealDB server"]
    async fn test_storage_backend_crud() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: SurrealDB not reachable");
                return;
            }
        };

        let backend: Arc<dyn StorageBackend> = Arc::new(db);
        let table = "test_surreal_crud";

        // ensure_table
        let schema = vec![
            FieldDef::required("name", FieldType::Utf8),
            FieldDef::required("value", FieldType::Int64),
        ];
        backend.ensure_table(table, &schema).await.unwrap();

        // Clean up from prior runs.
        let _ = backend
            .delete(table, &Filter::Raw("true".to_string()))
            .await;

        // insert
        let records = vec![
            vec![
                ("name".to_string(), FieldValue::Utf8(Some("x".into()))),
                ("value".to_string(), FieldValue::Int64(Some(10))),
            ],
            vec![
                ("name".to_string(), FieldValue::Utf8(Some("y".into()))),
                ("value".to_string(), FieldValue::Int64(Some(20))),
            ],
        ];
        backend.insert(table, records).await.unwrap();

        // count (all)
        let total = backend.count(table, None).await.unwrap();
        assert!(total >= 2, "expected at least 2 rows, got {}", total);

        // query with filter
        let filter = Filter::Eq("name".into(), FieldValue::Utf8(Some("x".into())));
        let rows = backend.query(table, Some(&filter), None).await.unwrap();
        assert!(!rows.is_empty(), "expected rows matching name=x");

        // delete
        backend.delete(table, &filter).await.unwrap();
        let after = backend
            .count(
                table,
                Some(&Filter::Eq(
                    "name".into(),
                    FieldValue::Utf8(Some("x".into())),
                )),
            )
            .await
            .unwrap();
        assert_eq!(after, 0);
    }

    #[tokio::test]
    #[ignore = "requires a running SurrealDB server"]
    async fn test_storage_backend_vector_search() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: SurrealDB not reachable");
                return;
            }
        };

        let backend: Arc<dyn StorageBackend> = Arc::new(db);
        let table = "test_surreal_vec";

        let dim = 4;
        let schema = vec![
            FieldDef::required("label", FieldType::Utf8),
            FieldDef::required("embedding", FieldType::Vector(dim)),
        ];
        backend.ensure_table(table, &schema).await.unwrap();

        // Clean up.
        let _ = backend
            .delete(table, &Filter::Raw("true".to_string()))
            .await;

        let records = vec![
            vec![
                ("label".to_string(), FieldValue::Utf8(Some("near".into()))),
                (
                    "embedding".to_string(),
                    FieldValue::Vector(vec![1.0, 0.0, 0.0, 0.0]),
                ),
            ],
            vec![
                ("label".to_string(), FieldValue::Utf8(Some("far".into()))),
                (
                    "embedding".to_string(),
                    FieldValue::Vector(vec![0.0, 0.0, 0.0, 1.0]),
                ),
            ],
        ];
        backend.insert(table, records).await.unwrap();

        let results = backend
            .vector_search(table, "embedding", vec![1.0, 0.0, 0.0, 0.0], 2, None)
            .await
            .unwrap();

        assert!(!results.is_empty(), "expected vector search results");
        let top_label = results[0]
            .record
            .iter()
            .find(|(n, _)| n == "label")
            .and_then(|(_, v)| v.as_str())
            .unwrap();
        assert_eq!(top_label, "near");
    }

    #[tokio::test]
    #[ignore = "requires a running SurrealDB server"]
    async fn test_capabilities() {
        let caps = SurrealDatabase::capabilities();
        assert!(
            caps.vector_search,
            "SurrealDB backend should advertise vector search"
        );
    }

    // ── VectorDatabase trait tests ───────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires a running SurrealDB server"]
    async fn test_vector_database_lifecycle() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: SurrealDB not reachable");
                return;
            }
        };

        // Initialize with small dimension.
        db.initialize(4).await.unwrap();

        // Clear any prior data.
        db.clear().await.unwrap();

        let meta = rullama_storage::rullama_core::ChunkMetadata {
            file_path: "src/lib.rs".to_string(),
            root_path: Some("/project".to_string()),
            project: Some("test-project".to_string()),
            start_line: 1,
            end_line: 5,
            language: Some("Rust".to_string()),
            extension: Some("rs".to_string()),
            file_hash: "def456".to_string(),
            indexed_at: chrono::Utc::now().timestamp(),
        };

        let stored = db
            .store_embeddings(
                vec![vec![0.5, 0.5, 0.0, 0.0]],
                vec![meta],
                vec!["pub fn hello() {}".to_string()],
                "/project",
            )
            .await
            .unwrap();
        assert_eq!(stored, 1);

        // Search
        let results = db
            .search(
                vec![0.5, 0.5, 0.0, 0.0],
                "hello function",
                5,
                0.0,
                None,
                None,
                false,
            )
            .await
            .unwrap();
        assert!(!results.is_empty(), "expected search results");

        // Statistics
        let stats = db.get_statistics().await.unwrap();
        assert!(stats.total_points >= 1);

        // Count by root path
        let count = db.count_by_root_path("/project").await.unwrap();
        assert!(count >= 1);

        // Get indexed files
        let files = db.get_indexed_files("/project").await.unwrap();
        assert!(files.contains(&"src/lib.rs".to_string()));

        // Delete by file
        let deleted = db.delete_by_file("src/lib.rs").await.unwrap();
        assert!(deleted >= 1);

        // Flush (no-op but should not error)
        db.flush().await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires a running SurrealDB server"]
    async fn test_vector_database_filtered_search() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: SurrealDB not reachable");
                return;
            }
        };

        db.initialize(4).await.unwrap();
        db.clear().await.unwrap();

        let ts = chrono::Utc::now().timestamp();

        let embeddings = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let metadata = vec![
            rullama_storage::rullama_core::ChunkMetadata {
                file_path: "src/main.rs".to_string(),
                root_path: Some("/project".to_string()),
                project: Some("test".to_string()),
                start_line: 1,
                end_line: 10,
                language: Some("Rust".to_string()),
                extension: Some("rs".to_string()),
                file_hash: "h1".to_string(),
                indexed_at: ts,
            },
            rullama_storage::rullama_core::ChunkMetadata {
                file_path: "src/app.py".to_string(),
                root_path: Some("/project".to_string()),
                project: Some("test".to_string()),
                start_line: 1,
                end_line: 5,
                language: Some("Python".to_string()),
                extension: Some("py".to_string()),
                file_hash: "h2".to_string(),
                indexed_at: ts,
            },
        ];
        let contents = vec!["fn main()".to_string(), "def app()".to_string()];

        db.store_embeddings(embeddings, metadata, contents, "/project")
            .await
            .unwrap();

        // Filter by language
        let results = db
            .search_filtered(
                vec![1.0, 0.0, 0.0, 0.0],
                "main",
                5,
                0.0,
                None,
                None,
                false,
                vec![],
                vec!["Rust".to_string()],
                vec![],
            )
            .await
            .unwrap();

        // All results should be Rust files.
        for r in &results {
            assert_eq!(r.language, "Rust");
        }
    }
}
