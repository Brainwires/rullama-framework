//! Integration tests for the PostgreSQL + pgvector backend.
//!
//! These tests require a running PostgreSQL server with the `vector` extension
//! installed. They are gated behind the `postgres-backend` feature and marked
//! `#[ignore]` so they only run when explicitly requested (e.g. `cargo test
//! --features postgres-backend -- --ignored`).

#[cfg(feature = "postgres-backend")]
mod postgres_integration {
    use std::sync::Arc;

    use rullama_storage::databases::postgres::PostgresDatabase;
    use rullama_storage::databases::traits::{StorageBackend, VectorDatabase};
    use rullama_storage::databases::types::{FieldDef, FieldType, FieldValue, Filter};

    /// Helper: try to connect to PostgreSQL, returning None if the server is
    /// not reachable.
    async fn try_connect() -> Option<PostgresDatabase> {
        // Use the POSTGRES_URL env var if set, otherwise fall back to default.
        let url = std::env::var("POSTGRES_URL").unwrap_or_else(|_| PostgresDatabase::default_url());
        PostgresDatabase::with_url(&url).await.ok()
    }

    // ── StorageBackend CRUD tests ────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires a running PostgreSQL server"]
    async fn test_storage_backend_crud() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: PostgreSQL not reachable");
                return;
            }
        };

        let backend: Arc<dyn StorageBackend> = Arc::new(db);
        let table = "test_pg_crud";

        // Clean up from prior runs.
        let _ = backend.delete(table, &Filter::Raw("1=1".to_string())).await;

        // ensure_table
        let schema = vec![
            FieldDef::required("key", FieldType::Utf8),
            FieldDef::required("val", FieldType::Int64),
        ];
        backend.ensure_table(table, &schema).await.unwrap();

        // insert
        let records = vec![
            vec![
                ("key".to_string(), FieldValue::Utf8(Some("alpha".into()))),
                ("val".to_string(), FieldValue::Int64(Some(100))),
            ],
            vec![
                ("key".to_string(), FieldValue::Utf8(Some("beta".into()))),
                ("val".to_string(), FieldValue::Int64(Some(200))),
            ],
        ];
        backend.insert(table, records).await.unwrap();

        // count (all)
        let total = backend.count(table, None).await.unwrap();
        assert!(total >= 2, "expected at least 2 rows, got {}", total);

        // query with filter
        let filter = Filter::Eq("key".into(), FieldValue::Utf8(Some("alpha".into())));
        let rows = backend.query(table, Some(&filter), None).await.unwrap();
        assert!(!rows.is_empty(), "expected rows matching key=alpha");

        let val = rows[0]
            .iter()
            .find(|(n, _)| n == "val")
            .and_then(|(_, v)| v.as_i64())
            .unwrap();
        assert_eq!(val, 100);

        // delete
        backend.delete(table, &filter).await.unwrap();
        let after_delete = backend
            .count(
                table,
                Some(&Filter::Eq(
                    "key".into(),
                    FieldValue::Utf8(Some("alpha".into())),
                )),
            )
            .await
            .unwrap();
        assert_eq!(after_delete, 0);
    }

    #[tokio::test]
    #[ignore = "requires a running PostgreSQL server"]
    async fn test_storage_backend_vector_search() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: PostgreSQL not reachable");
                return;
            }
        };

        let backend: Arc<dyn StorageBackend> = Arc::new(db);
        let table = "test_pg_vec";

        let _ = backend.delete(table, &Filter::Raw("1=1".to_string())).await;

        let dim = 4;
        let schema = vec![
            FieldDef::required("label", FieldType::Utf8),
            FieldDef::required("embedding", FieldType::Vector(dim)),
        ];
        backend.ensure_table(table, &schema).await.unwrap();

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
        // The closest vector should be "near".
        let top_label = results[0]
            .record
            .iter()
            .find(|(n, _)| n == "label")
            .and_then(|(_, v)| v.as_str())
            .unwrap();
        assert_eq!(top_label, "near");
    }

    // ── VectorDatabase trait tests ───────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires a running PostgreSQL server with pgvector"]
    async fn test_vector_database_lifecycle() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: PostgreSQL not reachable");
                return;
            }
        };

        // Initialize with a small dimension for testing.
        db.initialize(4).await.unwrap();

        // Clear any prior data.
        db.clear().await.unwrap();

        let meta = rullama_storage::rullama_core::ChunkMetadata {
            file_path: "src/main.rs".to_string(),
            root_path: Some("/project".to_string()),
            project: Some("test-project".to_string()),
            start_line: 1,
            end_line: 10,
            language: Some("Rust".to_string()),
            extension: Some("rs".to_string()),
            file_hash: "abc123".to_string(),
            indexed_at: chrono::Utc::now().timestamp(),
        };

        let stored = db
            .store_embeddings(
                vec![vec![1.0, 0.0, 0.0, 0.0]],
                vec![meta],
                vec!["fn main() {}".to_string()],
                "/project",
            )
            .await
            .unwrap();
        assert_eq!(stored, 1);

        // Search
        let results = db
            .search(
                vec![1.0, 0.0, 0.0, 0.0],
                "main function",
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

        // Delete by file
        let deleted = db.delete_by_file("src/main.rs").await.unwrap();
        assert!(deleted >= 1);

        // Flush (no-op for PostgreSQL but should not error)
        db.flush().await.unwrap();
    }
}
