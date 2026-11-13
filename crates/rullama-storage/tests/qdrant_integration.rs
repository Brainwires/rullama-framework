//! Integration tests for the Qdrant vector database backend.
//!
//! These tests require a running Qdrant server. They are gated behind the
//! `qdrant-backend` feature and marked `#[ignore]` so they only run when
//! explicitly requested (e.g. `cargo test --features qdrant-backend -- --ignored`).

#[cfg(feature = "qdrant-backend")]
mod qdrant_integration {
    use rullama_storage::databases::qdrant::QdrantDatabase;
    use rullama_storage::databases::traits::VectorDatabase;

    /// A [`QdrantDatabase`] bound to a collection unique to one test run.
    ///
    /// The tests run in parallel against a single Qdrant server, so each one
    /// gets its own collection; the guard deletes it on drop, including when
    /// the test panics.
    struct TestDb {
        db: QdrantDatabase,
        url: String,
        collection: String,
    }

    impl std::ops::Deref for TestDb {
        type Target = QdrantDatabase;
        fn deref(&self) -> &QdrantDatabase {
            &self.db
        }
    }

    impl Drop for TestDb {
        fn drop(&mut self) {
            // Drop is sync and may run inside a current-thread runtime, so the
            // cleanup runs on its own thread with its own runtime.
            let url = self.url.clone();
            let collection = self.collection.clone();
            let _ = std::thread::spawn(move || {
                if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    rt.block_on(async {
                        if let Ok(db) =
                            QdrantDatabase::with_url_and_collection(&url, &collection).await
                        {
                            let _ = db.clear().await;
                        }
                    });
                }
            })
            .join();
        }
    }

    /// Helper: try to connect to Qdrant with a collection unique to `test`,
    /// returning None if the server is not reachable.
    async fn try_connect(test: &str) -> Option<TestDb> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let url = std::env::var("QDRANT_URL").unwrap_or_else(|_| QdrantDatabase::default_url());
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let collection = format!(
            "{}_{}_{}_{}",
            test,
            std::process::id(),
            nanos,
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let db = QdrantDatabase::with_url_and_collection(&url, &collection)
            .await
            .ok()?;
        Some(TestDb {
            db,
            url,
            collection,
        })
    }

    #[tokio::test]
    #[ignore = "requires a running Qdrant server"]
    async fn test_vector_database_initialize_and_store() {
        let db = match try_connect("test_vector_database_initialize_and_store").await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: Qdrant not reachable");
                return;
            }
        };

        // Initialize with small dimension.
        db.initialize(4).await.unwrap();

        let meta = rullama_storage::rullama_core::ChunkMetadata {
            file_path: "src/lib.rs".to_string(),
            root_path: Some("/project".to_string()),
            project: Some("qdrant-test".to_string()),
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
                vec!["pub fn hello() {}".to_string()],
                "/project",
            )
            .await
            .unwrap();
        assert_eq!(stored, 1);
    }

    #[tokio::test]
    #[ignore = "requires a running Qdrant server"]
    async fn test_vector_database_search() {
        let db = match try_connect("test_vector_database_search").await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: Qdrant not reachable");
                return;
            }
        };

        db.initialize(4).await.unwrap();

        let ts = chrono::Utc::now().timestamp();
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let metadata: Vec<_> = (0..3)
            .map(|i| rullama_storage::rullama_core::ChunkMetadata {
                file_path: format!("src/file{}.rs", i),
                root_path: Some("/project".to_string()),
                project: Some("qdrant-test".to_string()),
                start_line: 1,
                end_line: 10,
                language: Some("Rust".to_string()),
                extension: Some("rs".to_string()),
                file_hash: format!("hash{}", i),
                indexed_at: ts,
            })
            .collect();
        let contents: Vec<_> = (0..3).map(|i| format!("fn func{}() {{}}", i)).collect();

        db.store_embeddings(embeddings, metadata, contents, "/project")
            .await
            .unwrap();

        // Search for the first vector.
        let results = db
            .search(vec![1.0, 0.0, 0.0, 0.0], "func0", 3, 0.0, None, None, false)
            .await
            .unwrap();
        assert!(!results.is_empty(), "expected search results from Qdrant");
        // The top result should be the most similar vector.
        assert!(results[0].score > 0.5, "top result should have high score");
    }

    #[tokio::test]
    #[ignore = "requires a running Qdrant server"]
    async fn test_vector_database_statistics() {
        let db = match try_connect("test_vector_database_statistics").await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: Qdrant not reachable");
                return;
            }
        };

        db.initialize(4).await.unwrap();

        let meta = rullama_storage::rullama_core::ChunkMetadata {
            file_path: "test.rs".to_string(),
            root_path: Some("/p".to_string()),
            project: Some("stats-test".to_string()),
            start_line: 0,
            end_line: 1,
            language: Some("Rust".to_string()),
            extension: Some("rs".to_string()),
            file_hash: "h".to_string(),
            indexed_at: 0,
        };

        db.store_embeddings(
            vec![vec![0.5, 0.5, 0.0, 0.0]],
            vec![meta],
            vec!["test content".to_string()],
            "/p",
        )
        .await
        .unwrap();

        let stats = db.get_statistics().await.unwrap();
        assert!(stats.total_points >= 1, "should have at least 1 point");
    }

    #[tokio::test]
    #[ignore = "requires a running Qdrant server"]
    async fn test_vector_database_delete_and_clear() {
        let db = match try_connect("test_vector_database_delete_and_clear").await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: Qdrant not reachable");
                return;
            }
        };

        db.initialize(4).await.unwrap();

        let meta = rullama_storage::rullama_core::ChunkMetadata {
            file_path: "delete_me.rs".to_string(),
            root_path: Some("/p".to_string()),
            project: None,
            start_line: 0,
            end_line: 1,
            language: None,
            extension: Some("rs".to_string()),
            file_hash: "d".to_string(),
            indexed_at: 0,
        };

        db.store_embeddings(
            vec![vec![1.0, 0.0, 0.0, 0.0]],
            vec![meta],
            vec!["to delete".to_string()],
            "/p",
        )
        .await
        .unwrap();

        // delete_by_file
        let _ = db.delete_by_file("delete_me.rs").await.unwrap();

        // flush (no-op but should not error)
        db.flush().await.unwrap();

        // clear
        db.clear().await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires a running Qdrant server"]
    async fn test_vector_database_filtered_search() {
        let db = match try_connect("test_vector_database_filtered_search").await {
            Some(db) => db,
            None => {
                eprintln!("Skipping: Qdrant not reachable");
                return;
            }
        };

        db.initialize(4).await.unwrap();

        let ts = chrono::Utc::now().timestamp();
        let embeddings = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let metadata = vec![
            rullama_storage::rullama_core::ChunkMetadata {
                file_path: "main.rs".to_string(),
                root_path: Some("/project".to_string()),
                project: Some("filtered-test".to_string()),
                start_line: 1,
                end_line: 10,
                language: Some("Rust".to_string()),
                extension: Some("rs".to_string()),
                file_hash: "f1".to_string(),
                indexed_at: ts,
            },
            rullama_storage::rullama_core::ChunkMetadata {
                file_path: "app.py".to_string(),
                root_path: Some("/project".to_string()),
                project: Some("filtered-test".to_string()),
                start_line: 1,
                end_line: 5,
                language: Some("Python".to_string()),
                extension: Some("py".to_string()),
                file_hash: "f2".to_string(),
                indexed_at: ts,
            },
        ];
        let contents = vec!["fn main()".to_string(), "def app()".to_string()];

        db.store_embeddings(embeddings, metadata, contents, "/project")
            .await
            .unwrap();

        // Filter by extension
        let results = db
            .search_filtered(
                vec![1.0, 0.0, 0.0, 0.0],
                "main",
                5,
                0.0,
                None,
                None,
                false,
                vec!["rs".to_string()],
                vec![],
                vec![],
            )
            .await
            .unwrap();

        // All results should be .rs files.
        for r in &results {
            assert!(
                r.file_path.ends_with(".rs"),
                "expected .rs file, got {}",
                r.file_path
            );
        }
    }
}
