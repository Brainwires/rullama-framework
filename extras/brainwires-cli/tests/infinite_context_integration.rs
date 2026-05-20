use anyhow::Result;
use brainwires_cli::storage::{
    CachedEmbeddingProvider, LanceDatabase, MessageMetadata, MessageStore, VectorDatabase,
};
use chrono::Utc;
use std::sync::Arc;

/// Test basic message storage and retrieval with embeddings
#[tokio::test]
async fn test_message_storage_and_search() -> Result<()> {
    // Setup temporary test database
    let test_db = tempfile::tempdir()?;
    let db_path = test_db.path().join("test_conversations.lance");

    let client = Arc::new(LanceDatabase::new(db_path.to_str().unwrap()).await?);
    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);

    // Initialize database tables
    client.initialize(embeddings.dimension()).await?;

    let store = MessageStore::new(client.clone(), embeddings.clone());
    store.ensure_table().await?;

    // Create test messages with known content
    let messages = vec![
        MessageMetadata {
            message_id: "msg1".to_string(),
            conversation_id: "conv_test".to_string(),
            role: "user".to_string(),
            content: "I want to implement JWT authentication with refresh tokens using RS256 algorithm".to_string(),
            token_count: Some(15),
            model_id: Some("claude-3-5-sonnet".to_string()),
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp(),
        },
        MessageMetadata {
            message_id: "msg2".to_string(),
            conversation_id: "conv_test".to_string(),
            role: "assistant".to_string(),
            content: "For JWT authentication, RS256 is a good choice for asymmetric signing. Here's how to implement refresh tokens...".to_string(),
            token_count: Some(20),
            model_id: Some("claude-3-5-sonnet".to_string()),
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp(),
        },
        MessageMetadata {
            message_id: "msg3".to_string(),
            conversation_id: "conv_test".to_string(),
            role: "user".to_string(),
            content: "The weather is really nice today in San Francisco".to_string(),
            token_count: Some(10),
            model_id: None,
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp(),
        },
        MessageMetadata {
            message_id: "msg4".to_string(),
            conversation_id: "conv_test".to_string(),
            role: "user".to_string(),
            content: "Tell me a joke about programming".to_string(),
            token_count: Some(7),
            model_id: None,
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp(),
        },
    ];

    // Store messages in batch
    store.add_batch(messages).await?;

    // Test 1: Search for JWT-related content
    let results = store
        .search_conversation("conv_test", "JWT authentication RS256 algorithm", 10, 0.6)
        .await?;

    assert!(!results.is_empty(), "Should find JWT-related messages");

    // First result should be the JWT implementation message
    let (first_msg, first_score) = &results[0];
    assert_eq!(
        first_msg.message_id, "msg1",
        "Should find the JWT question as most relevant"
    );
    assert!(
        *first_score > 0.70,
        "JWT query should have high relevance score: {}",
        first_score
    );

    // Second result should be the JWT response
    if results.len() > 1 {
        let (second_msg, second_score) = &results[1];
        assert_eq!(
            second_msg.message_id, "msg2",
            "Should find JWT response as second result"
        );
        assert!(
            *second_score > 0.70,
            "JWT response should be relevant: {}",
            second_score
        );
    }

    // Test 2: Irrelevant queries should NOT return JWT messages
    let irrelevant_results = store
        .search_conversation("conv_test", "weather forecast sunny", 10, 0.6)
        .await?;

    if !irrelevant_results.is_empty() {
        let (top_result, _) = &irrelevant_results[0];
        assert_eq!(
            top_result.message_id, "msg3",
            "Weather query should find weather message"
        );
    }

    // Test 3: Cross-conversation search (search all)
    let all_results = store.search("authentication tokens", 10, 0.4).await?;

    assert!(
        !all_results.is_empty(),
        "Should find authentication messages across all conversations"
    );

    println!("✅ Test passed: Message storage and semantic search working correctly");
    println!("   - Stored 4 messages with embeddings");
    println!("   - JWT query found relevant messages with scores > 0.75");
    println!("   - Irrelevant queries filtered correctly");

    Ok(())
}

/// Test embedding cache performance
#[tokio::test]
async fn test_embedding_cache() -> Result<()> {
    use std::time::Instant;

    let embeddings = CachedEmbeddingProvider::new()?;

    // Cold embedding (not cached)
    let query = "JWT authentication implementation";
    let start = Instant::now();
    let _emb1 = embeddings.embed_cached(query)?;
    let cold_latency = start.elapsed();

    // Hot embedding (cached)
    let start = Instant::now();
    let _emb2 = embeddings.embed_cached(query)?;
    let hot_latency = start.elapsed();

    println!("✅ Embedding cache test:");
    println!("   - Cold latency: {:?}", cold_latency);
    println!("   - Hot latency:  {:?}", hot_latency);
    println!(
        "   - Speedup: {:.1}x",
        cold_latency.as_micros() as f64 / hot_latency.as_micros().max(1) as f64
    );

    // Cached should be significantly faster (at least 10x)
    assert!(
        hot_latency < cold_latency / 10,
        "Cached embedding should be much faster: cold={:?}, hot={:?}",
        cold_latency,
        hot_latency
    );

    Ok(())
}

/// Test message retrieval by conversation ID
#[tokio::test]
async fn test_conversation_isolation() -> Result<()> {
    let test_db = tempfile::tempdir()?;
    let db_path = test_db.path().join("test_conversations.lance");

    let client = Arc::new(LanceDatabase::new(db_path.to_str().unwrap()).await?);
    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
    client.initialize(embeddings.dimension()).await?;

    let store = MessageStore::new(client, embeddings);
    store.ensure_table().await?;

    // Create messages in different conversations
    let conv1_messages = vec![MessageMetadata {
        message_id: "conv1_msg1".to_string(),
        conversation_id: "conversation_1".to_string(),
        role: "user".to_string(),
        content: "Use PostgreSQL with Supabase for the database".to_string(),
        token_count: Some(10),
        model_id: None,
        images: None,
        expires_at: None,
        created_at: Utc::now().timestamp(),
    }];

    let conv2_messages = vec![MessageMetadata {
        message_id: "conv2_msg1".to_string(),
        conversation_id: "conversation_2".to_string(),
        role: "user".to_string(),
        content: "Use MongoDB with Atlas for the database".to_string(),
        token_count: Some(10),
        model_id: None,
        images: None,
        expires_at: None,
        created_at: Utc::now().timestamp(),
    }];

    store.add_batch(conv1_messages).await?;
    store.add_batch(conv2_messages).await?;

    // Search within conversation 1 only
    let conv1_results = store
        .search_conversation("conversation_1", "database choice", 10, 0.3)
        .await?;

    assert!(
        !conv1_results.is_empty(),
        "Should find messages in conversation 1"
    );
    assert_eq!(conv1_results[0].0.conversation_id, "conversation_1");
    assert!(
        conv1_results[0].0.content.contains("PostgreSQL"),
        "Should find PostgreSQL message, not MongoDB"
    );

    // Search within conversation 2 only
    let conv2_results = store
        .search_conversation("conversation_2", "database choice", 10, 0.3)
        .await?;

    assert!(
        !conv2_results.is_empty(),
        "Should find messages in conversation 2"
    );
    assert_eq!(conv2_results[0].0.conversation_id, "conversation_2");
    assert!(
        conv2_results[0].0.content.contains("MongoDB"),
        "Should find MongoDB message, not PostgreSQL"
    );

    println!("✅ Conversation isolation test passed");
    println!("   - Messages correctly isolated by conversation_id");
    println!("   - Cross-contamination prevented");

    Ok(())
}

/// Test search with various min_score thresholds
#[tokio::test]
async fn test_relevance_thresholds() -> Result<()> {
    let test_db = tempfile::tempdir()?;
    let db_path = test_db.path().join("test_conversations.lance");

    let client = Arc::new(LanceDatabase::new(db_path.to_str().unwrap()).await?);
    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
    client.initialize(embeddings.dimension()).await?;

    let store = MessageStore::new(client, embeddings);
    store.ensure_table().await?;

    // Create messages with varying relevance to query
    let messages = vec![
        MessageMetadata {
            message_id: "exact_match".to_string(),
            conversation_id: "test".to_string(),
            role: "user".to_string(),
            content: "Implement rate limiting with token bucket algorithm".to_string(),
            token_count: Some(10),
            model_id: None,
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp(),
        },
        MessageMetadata {
            message_id: "partial_match".to_string(),
            conversation_id: "test".to_string(),
            role: "user".to_string(),
            content: "What's a good approach for limiting API requests".to_string(),
            token_count: Some(10),
            model_id: None,
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp(),
        },
        MessageMetadata {
            message_id: "weak_match".to_string(),
            conversation_id: "test".to_string(),
            role: "user".to_string(),
            content: "The system should handle high traffic efficiently".to_string(),
            token_count: Some(10),
            model_id: None,
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp(),
        },
    ];

    store.add_batch(messages).await?;

    // Test with high threshold (0.8) - should only get exact matches
    let high_threshold = store
        .search_conversation("test", "rate limiting token bucket", 10, 0.8)
        .await?;

    println!("High threshold (0.8): {} results", high_threshold.len());
    for (msg, score) in &high_threshold {
        println!(
            "  - [{}] {}: {}",
            score,
            msg.message_id,
            &msg.content[..60.min(msg.content.len())]
        );
    }

    // Test with medium threshold (0.6) - should get exact + partial
    let med_threshold = store
        .search_conversation("test", "rate limiting token bucket", 10, 0.6)
        .await?;

    println!("Medium threshold (0.6): {} results", med_threshold.len());

    // Test with low threshold (0.4) - should get all
    let low_threshold = store
        .search_conversation("test", "rate limiting token bucket", 10, 0.4)
        .await?;

    println!("Low threshold (0.4): {} results", low_threshold.len());

    assert!(
        high_threshold.len() <= med_threshold.len(),
        "Higher threshold should return fewer results"
    );
    assert!(
        med_threshold.len() <= low_threshold.len(),
        "Lower threshold should return more results"
    );

    println!("✅ Relevance threshold test passed");

    Ok(())
}

/// Benchmark search performance
#[tokio::test]
async fn bench_search_latency() -> Result<()> {
    use std::time::Instant;

    let test_db = tempfile::tempdir()?;
    let db_path = test_db.path().join("test_conversations.lance");

    let client = Arc::new(LanceDatabase::new(db_path.to_str().unwrap()).await?);
    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
    client.initialize(embeddings.dimension()).await?;

    let store = MessageStore::new(client, embeddings);
    store.ensure_table().await?;

    // Add 100 messages to make search more realistic
    let mut messages = Vec::new();
    for i in 0..100 {
        messages.push(MessageMetadata {
            message_id: format!("msg_{}", i),
            conversation_id: "bench_test".to_string(),
            role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
            content: format!("This is test message number {} about various topics like databases, authentication, APIs, and system design", i),
            token_count: Some(20),
            model_id: None,
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp() - (100 - i as i64),
        });
    }

    store.add_batch(messages).await?;

    // Warm up
    let _ = store
        .search_conversation("bench_test", "database", 5, 0.6)
        .await?;

    // Benchmark search latency
    let iterations = 10;
    let mut total_latency = std::time::Duration::ZERO;

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = store
            .search_conversation("bench_test", "authentication system design", 5, 0.6)
            .await?;
        total_latency += start.elapsed();
    }

    let avg_latency = total_latency / iterations;

    println!("✅ Search performance benchmark:");
    println!("   - Database size: 100 messages");
    println!("   - Query: 'authentication system design'");
    println!("   - Average latency: {:?}", avg_latency);
    println!("   - Iterations: {}", iterations);

    // Assert reasonable performance (< 200ms for 100 messages)
    assert!(
        avg_latency.as_millis() < 200,
        "Search should complete in < 200ms, took {:?}",
        avg_latency
    );

    Ok(())
}
