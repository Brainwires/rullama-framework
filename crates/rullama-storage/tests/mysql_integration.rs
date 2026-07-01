//! Integration tests for the MySQL / MariaDB backend.
//!
//! These tests require a running MySQL server. They are gated behind the
//! `mysql-backend` feature and marked `#[ignore]` so they only run when
//! explicitly requested (e.g. `cargo test --features mysql-backend -- --ignored`).

#[cfg(feature = "mysql-backend")]
mod mysql_integration {
    use std::sync::Arc;

    use rullama_storage::databases::mysql::MySqlDatabase;
    use rullama_storage::databases::traits::StorageBackend;
    use rullama_storage::databases::types::{FieldDef, FieldType, FieldValue, Filter};

    /// Helper: try to connect to MySQL, returning None if the server is
    /// not reachable.
    async fn try_connect() -> Option<MySqlDatabase> {
        let url = std::env::var("MYSQL_URL").unwrap_or_else(|_| MySqlDatabase::default_url());
        MySqlDatabase::new(&url).await.ok()
    }

    #[tokio::test]
    #[ignore = "requires a running MySQL server"]
    async fn test_storage_backend_crud() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: MySQL not reachable");
                return;
            }
        };

        let backend: Arc<dyn StorageBackend> = Arc::new(db);
        let table = "test_mysql_crud";

        // Clean up from prior runs.
        let _ = backend.delete(table, &Filter::Raw("1=1".to_string())).await;

        // ensure_table
        let schema = vec![
            FieldDef::required("name", FieldType::Utf8),
            FieldDef::required("score", FieldType::Int64),
        ];
        backend.ensure_table(table, &schema).await.unwrap();

        // insert
        let records = vec![
            vec![
                ("name".to_string(), FieldValue::Utf8(Some("alice".into()))),
                ("score".to_string(), FieldValue::Int64(Some(95))),
            ],
            vec![
                ("name".to_string(), FieldValue::Utf8(Some("bob".into()))),
                ("score".to_string(), FieldValue::Int64(Some(87))),
            ],
        ];
        backend.insert(table, records).await.unwrap();

        // count
        let total = backend.count(table, None).await.unwrap();
        assert!(total >= 2, "expected at least 2 rows, got {}", total);

        // query with filter
        let filter = Filter::Eq("name".into(), FieldValue::Utf8(Some("alice".into())));
        let rows = backend.query(table, Some(&filter), None).await.unwrap();
        assert!(!rows.is_empty(), "expected rows matching name=alice");

        // delete
        backend.delete(table, &filter).await.unwrap();
        let after = backend
            .count(
                table,
                Some(&Filter::Eq(
                    "name".into(),
                    FieldValue::Utf8(Some("alice".into())),
                )),
            )
            .await
            .unwrap();
        assert_eq!(after, 0);
    }

    #[tokio::test]
    #[ignore = "requires a running MySQL server"]
    async fn test_capabilities_no_vector_search() {
        let caps = MySqlDatabase::capabilities();
        assert!(
            !caps.vector_search,
            "MySQL backend should not advertise native vector search"
        );
    }

    #[tokio::test]
    #[ignore = "requires a running MySQL server"]
    async fn test_client_side_vector_search() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: MySQL not reachable");
                return;
            }
        };

        let backend: Arc<dyn StorageBackend> = Arc::new(db);
        let table = "test_mysql_vec";

        let _ = backend.delete(table, &Filter::Raw("1=1".to_string())).await;

        // MySQL stores vectors as JSON, so use Utf8 type for the vector column
        // (the backend converts Vector values to JSON strings on insert).
        let schema = vec![
            FieldDef::required("label", FieldType::Utf8),
            FieldDef::required("embedding", FieldType::Utf8),
        ];
        backend.ensure_table(table, &schema).await.unwrap();

        let records = vec![
            vec![
                ("label".to_string(), FieldValue::Utf8(Some("close".into()))),
                (
                    "embedding".to_string(),
                    FieldValue::Vector(vec![1.0, 0.0, 0.0]),
                ),
            ],
            vec![
                (
                    "label".to_string(),
                    FieldValue::Utf8(Some("distant".into())),
                ),
                (
                    "embedding".to_string(),
                    FieldValue::Vector(vec![0.0, 0.0, 1.0]),
                ),
            ],
        ];
        backend.insert(table, records).await.unwrap();

        // Client-side cosine similarity search.
        let results = backend
            .vector_search(table, "embedding", vec![1.0, 0.0, 0.0], 2, None)
            .await
            .unwrap();

        assert!(!results.is_empty(), "expected vector search results");
    }

    #[tokio::test]
    #[ignore = "requires a running MySQL server"]
    async fn test_query_with_limit() {
        let db = match try_connect().await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: MySQL not reachable");
                return;
            }
        };

        let backend: Arc<dyn StorageBackend> = Arc::new(db);
        let table = "test_mysql_limit";

        let _ = backend.delete(table, &Filter::Raw("1=1".to_string())).await;

        let schema = vec![FieldDef::required("idx", FieldType::Int64)];
        backend.ensure_table(table, &schema).await.unwrap();

        let records: Vec<_> = (0..10)
            .map(|i| vec![("idx".to_string(), FieldValue::Int64(Some(i)))])
            .collect();
        backend.insert(table, records).await.unwrap();

        let limited = backend.query(table, None, Some(3)).await.unwrap();
        assert_eq!(limited.len(), 3, "LIMIT should restrict result count");
    }
}
