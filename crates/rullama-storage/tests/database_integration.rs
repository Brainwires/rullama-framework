//! Integration tests for the unified database layer.
//!
//! These tests exercise `LanceDatabase` through trait objects to verify
//! that the `StorageBackend` and `VectorDatabase` dyn-safety works as
//! expected in real consumer code.

#[cfg(feature = "lance-backend")]
mod lance_integration {
    use std::sync::Arc;

    use rullama_storage::databases::capabilities::BackendCapabilities;
    use rullama_storage::databases::lance::LanceDatabase;
    use rullama_storage::databases::traits::StorageBackend;
    use rullama_storage::databases::types::{FieldDef, FieldType, FieldValue, Filter};

    /// Verify that `LanceDatabase` can be used as `Arc<dyn StorageBackend>`
    /// and that basic CRUD operations work through the trait object.
    #[tokio::test]
    async fn test_trait_object_crud() {
        let temp = tempfile::TempDir::new().unwrap();
        let db_path = temp.path().join("integration.lance");
        let db: Arc<dyn StorageBackend> =
            Arc::new(LanceDatabase::new(db_path.to_str().unwrap()).await.unwrap());

        // ensure_table
        let schema = vec![
            FieldDef::required("key", FieldType::Utf8),
            FieldDef::required("val", FieldType::Int64),
        ];
        db.ensure_table("items", &schema).await.unwrap();

        // insert
        let records = vec![
            vec![
                ("key".to_string(), FieldValue::Utf8(Some("x".to_string()))),
                ("val".to_string(), FieldValue::Int64(Some(10))),
            ],
            vec![
                ("key".to_string(), FieldValue::Utf8(Some("y".to_string()))),
                ("val".to_string(), FieldValue::Int64(Some(20))),
            ],
        ];
        db.insert("items", records).await.unwrap();

        // count (all)
        let total = db.count("items", None).await.unwrap();
        assert_eq!(total, 2);

        // query with filter
        let filter = Filter::Eq("key".into(), FieldValue::Utf8(Some("x".into())));
        let rows = db.query("items", Some(&filter), None).await.unwrap();
        assert_eq!(rows.len(), 1);
        let val = rows[0]
            .iter()
            .find(|(n, _)| n == "val")
            .and_then(|(_, v)| v.as_i64())
            .unwrap();
        assert_eq!(val, 10);

        // delete
        db.delete("items", &filter).await.unwrap();
        let remaining = db.count("items", None).await.unwrap();
        assert_eq!(remaining, 1);
    }

    /// Verify that `BackendCapabilities` reports correctly for LanceDatabase.
    #[tokio::test]
    async fn test_backend_capabilities_lance() {
        let temp = tempfile::TempDir::new().unwrap();
        let db_path = temp.path().join("caps_int.lance");
        let db = LanceDatabase::new(db_path.to_str().unwrap()).await.unwrap();

        let caps = db.capabilities();
        assert_eq!(
            caps,
            BackendCapabilities {
                vector_search: true,
            }
        );

        // Also verify the Default impl (which should match LanceDB's advertised caps).
        assert_eq!(caps, BackendCapabilities::default());
    }
}
