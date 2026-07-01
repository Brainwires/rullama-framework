//! Unit tests for [`LanceDatabase`].

use super::database::LanceDatabase;
use crate::databases::traits::{StorageBackend, VectorDatabase};
use crate::databases::types::{FieldDef, FieldValue, Filter};
use tempfile::TempDir;

#[tokio::test]
async fn test_lance_database_new() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("test.lance");
    let db = LanceDatabase::new(db_path.to_str().unwrap()).await.unwrap();
    assert_eq!(db.db_path(), db_path.to_str().unwrap());
}

#[tokio::test]
async fn test_lance_storage_backend_crud() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("test.lance");
    let db = LanceDatabase::new(db_path.to_str().unwrap()).await.unwrap();

    let schema = vec![
        FieldDef::required("id", crate::databases::types::FieldType::Utf8),
        FieldDef::required("value", crate::databases::types::FieldType::Int64),
    ];
    db.ensure_table("test_table", &schema).await.unwrap();

    let records = vec![vec![
        ("id".to_string(), FieldValue::Utf8(Some("row1".to_string()))),
        ("value".to_string(), FieldValue::Int64(Some(42))),
    ]];
    db.insert("test_table", records).await.unwrap();

    let results = db.query("test_table", None, None).await.unwrap();
    assert_eq!(results.len(), 1);

    let count = db.count("test_table", None).await.unwrap();
    assert_eq!(count, 1);

    db.delete(
        "test_table",
        &Filter::Eq("id".into(), FieldValue::Utf8(Some("row1".into()))),
    )
    .await
    .unwrap();

    let count = db.count("test_table", None).await.unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_lance_vector_search() {
    use crate::databases::types::FieldType;

    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("vec_search.lance");
    let db = LanceDatabase::new(db_path.to_str().unwrap()).await.unwrap();

    let dim = 4;
    let schema = vec![
        FieldDef::required("id", FieldType::Utf8),
        FieldDef::required("embedding", FieldType::Vector(dim)),
    ];
    db.ensure_table("vectors", &schema).await.unwrap();

    // Insert three records with different vectors.
    let records = vec![
        vec![
            ("id".to_string(), FieldValue::Utf8(Some("a".to_string()))),
            (
                "embedding".to_string(),
                FieldValue::Vector(vec![1.0, 0.0, 0.0, 0.0]),
            ),
        ],
        vec![
            ("id".to_string(), FieldValue::Utf8(Some("b".to_string()))),
            (
                "embedding".to_string(),
                FieldValue::Vector(vec![0.0, 1.0, 0.0, 0.0]),
            ),
        ],
        vec![
            ("id".to_string(), FieldValue::Utf8(Some("c".to_string()))),
            (
                "embedding".to_string(),
                FieldValue::Vector(vec![0.9, 0.1, 0.0, 0.0]),
            ),
        ],
    ];
    db.insert("vectors", records).await.unwrap();

    // Search for a vector closest to [1, 0, 0, 0] — should rank "a" first.
    let results = db
        .vector_search("vectors", "embedding", vec![1.0, 0.0, 0.0, 0.0], 3, None)
        .await
        .unwrap();

    assert!(!results.is_empty(), "vector_search should return results");
    // The first result should be "a" (exact match → distance 0 → highest score).
    let first_id = results[0]
        .record
        .iter()
        .find(|(n, _)| n == "id")
        .and_then(|(_, v)| v.as_str())
        .unwrap();
    assert_eq!(first_id, "a");

    // Scores should be in descending order.
    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "scores should be descending: {} >= {}",
            w[0].score,
            w[1].score
        );
    }
}

#[tokio::test]
async fn test_lance_capabilities() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("caps.lance");
    let db = LanceDatabase::new(db_path.to_str().unwrap()).await.unwrap();

    let caps = db.capabilities();
    assert!(
        caps.vector_search,
        "LanceDatabase should support vector search"
    );
}

#[tokio::test]
async fn test_lance_shared_connection() {
    use crate::databases::types::FieldType;

    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("shared.lance");
    let db = LanceDatabase::new(db_path.to_str().unwrap()).await.unwrap();

    // Use StorageBackend trait
    let schema = vec![FieldDef::required("name", FieldType::Utf8)];
    db.ensure_table("store_table", &schema).await.unwrap();
    let records = vec![vec![(
        "name".to_string(),
        FieldValue::Utf8(Some("test".to_string())),
    )]];
    db.insert("store_table", records).await.unwrap();

    // Use VectorDatabase trait on same instance
    db.initialize(4).await.unwrap();

    // Both should work on the same connection
    let store_count = db.count("store_table", None).await.unwrap();
    assert_eq!(store_count, 1);

    let stats = db.get_statistics().await.unwrap();
    assert_eq!(stats.total_vectors, 0);
}
