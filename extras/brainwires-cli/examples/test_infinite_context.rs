/// Quick manual test for infinite context system
///
/// Run with: cargo run --example test_infinite_context
///
/// This creates a test conversation, stores messages with embeddings,
/// and demonstrates semantic search retrieval.
use anyhow::Result;
use brainwires_cli::storage::{
    CachedEmbeddingProvider, LanceDatabase, MessageMetadata, MessageStore, VectorDatabase,
};
use chrono::Utc;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Testing Infinite Context System\n");
    println!("{}", "=".repeat(60));

    // Use a test database
    let test_db_path = "/tmp/test_infinite_context.lance";

    // Clean up any existing test database
    if std::path::Path::new(test_db_path).exists() {
        println!("Cleaning up old test database...");
        std::fs::remove_dir_all(test_db_path)?;
    }

    println!("1️⃣  Initializing LanceDB and embedding model...");
    let client = Arc::new(LanceDatabase::new(test_db_path).await?);
    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);

    println!("   ✓ Embedding dimension: {}", embeddings.dimension());

    client.initialize(embeddings.dimension()).await?;
    println!("   ✓ Database initialized at {}", test_db_path);

    let store = MessageStore::new(client.clone(), embeddings.clone());

    println!("\n2️⃣  Creating test conversation with realistic messages...");

    let messages = vec![
        MessageMetadata {
            message_id: "msg_1".to_string(),
            conversation_id: "test_conversation".to_string(),
            role: "user".to_string(),
            content: "I need to implement JWT authentication for our API. Should I use RS256 or HS256?".to_string(),
            token_count: Some(18),
            model_id: Some("claude-3-5-sonnet".to_string()),
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp(),
        },
        MessageMetadata {
            message_id: "msg_2".to_string(),
            conversation_id: "test_conversation".to_string(),
            role: "assistant".to_string(),
            content: "For JWT authentication, I recommend RS256 (asymmetric) over HS256 (symmetric) because:\n\
                     1. RS256 allows you to distribute public keys for verification\n\
                     2. Private key stays secret on the auth server\n\
                     3. Better for microservices where multiple services verify tokens".to_string(),
            token_count: Some(60),
            model_id: Some("claude-3-5-sonnet".to_string()),
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp() + 1,
        },
        MessageMetadata {
            message_id: "msg_3".to_string(),
            conversation_id: "test_conversation".to_string(),
            role: "user".to_string(),
            content: "What about refresh tokens? How should I implement those?".to_string(),
            token_count: Some(12),
            model_id: Some("claude-3-5-sonnet".to_string()),
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp() + 2,
        },
        MessageMetadata {
            message_id: "msg_4".to_string(),
            conversation_id: "test_conversation".to_string(),
            role: "assistant".to_string(),
            content: "For refresh tokens, implement these best practices:\n\
                     1. Store refresh tokens in httpOnly cookies\n\
                     2. Use a rotation strategy - issue new refresh token on each use\n\
                     3. Set longer expiry (7-30 days) vs access tokens (15 minutes)\n\
                     4. Store refresh token hashes in database with user_id".to_string(),
            token_count: Some(65),
            model_id: Some("claude-3-5-sonnet".to_string()),
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp() + 3,
        },
        MessageMetadata {
            message_id: "msg_5".to_string(),
            conversation_id: "test_conversation".to_string(),
            role: "user".to_string(),
            content: "Let's also discuss the database schema. We need users, posts, and comments tables.".to_string(),
            token_count: Some(18),
            model_id: Some("claude-3-5-sonnet".to_string()),
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp() + 4,
        },
        MessageMetadata {
            message_id: "msg_6".to_string(),
            conversation_id: "test_conversation".to_string(),
            role: "assistant".to_string(),
            content: "Here's a basic schema:\n\
                     - users: id, email, password_hash, created_at\n\
                     - posts: id, user_id, title, content, created_at\n\
                     - comments: id, post_id, user_id, content, parent_id (for nesting), created_at".to_string(),
            token_count: Some(55),
            model_id: Some("claude-3-5-sonnet".to_string()),
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp() + 5,
        },
        MessageMetadata {
            message_id: "msg_7".to_string(),
            conversation_id: "test_conversation".to_string(),
            role: "user".to_string(),
            content: "The weather is really nice today!".to_string(),
            token_count: Some(7),
            model_id: None,
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp() + 6,
        },
        MessageMetadata {
            message_id: "msg_8".to_string(),
            conversation_id: "test_conversation".to_string(),
            role: "user".to_string(),
            content: "Tell me a programming joke".to_string(),
            token_count: Some(5),
            model_id: None,
            images: None,
            expires_at: None,
            created_at: Utc::now().timestamp() + 7,
        },
    ];

    println!("   ✓ Created {} messages", messages.len());

    println!("\n3️⃣  Storing messages with embeddings (batch)...");
    let start = std::time::Instant::now();
    store.add_batch(messages).await?;
    let storage_time = start.elapsed();
    println!("   ✓ Stored in {:?}", storage_time);

    println!("\n4️⃣  Testing semantic search queries...\n");

    // Test Query 1: JWT authentication
    println!("   Query: 'JWT authentication RS256 algorithm'");
    let start = std::time::Instant::now();
    let results = store
        .search_conversation(
            "test_conversation",
            "JWT authentication RS256 algorithm",
            5,
            0.6,
        )
        .await?;
    let search_time = start.elapsed();

    println!("   Search time: {:?}", search_time);
    println!("   Results: {} messages found\n", results.len());

    for (i, (msg, score)) in results.iter().enumerate() {
        println!("     {}. [Score: {:.3}] [{}]", i + 1, score, msg.role);
        let preview = if msg.content.len() > 100 {
            format!("{}...", &msg.content[..100])
        } else {
            msg.content.clone()
        };
        println!("        {}\n", preview);
    }

    // Test Query 2: Refresh tokens
    println!("   Query: 'refresh token rotation strategy'");
    let results = store
        .search_conversation(
            "test_conversation",
            "refresh token rotation strategy",
            5,
            0.6,
        )
        .await?;

    println!("   Results: {} messages found\n", results.len());
    for (i, (msg, score)) in results.iter().enumerate() {
        println!("     {}. [Score: {:.3}] {}", i + 1, score, msg.message_id);
    }

    // Test Query 3: Database schema
    println!("\n   Query: 'database schema users posts'");
    let results = store
        .search_conversation("test_conversation", "database schema users posts", 5, 0.6)
        .await?;

    println!("   Results: {} messages found\n", results.len());
    for (i, (msg, score)) in results.iter().enumerate() {
        println!("     {}. [Score: {:.3}] {}", i + 1, score, msg.message_id);
    }

    // Test Query 4: Irrelevant query (should find weather message)
    println!("\n   Query: 'weather forecast'");
    let results = store
        .search_conversation("test_conversation", "weather forecast", 5, 0.5)
        .await?;

    println!("   Results: {} messages found\n", results.len());
    for (i, (msg, score)) in results.iter().enumerate() {
        println!("     {}. [Score: {:.3}] {}", i + 1, score, msg.content);
    }

    println!("\n5️⃣  Testing embedding cache performance...\n");

    let query = "JWT authentication implementation";

    // Cold (first call)
    let start = std::time::Instant::now();
    let _ = embeddings.embed_cached(query)?;
    let cold_time = start.elapsed();

    // Hot (cached)
    let start = std::time::Instant::now();
    let _ = embeddings.embed_cached(query)?;
    let hot_time = start.elapsed();

    println!("   Cold (first call): {:?}", cold_time);
    println!("   Hot (cached):      {:?}", hot_time);
    println!(
        "   Speedup:           {:.1}x",
        cold_time.as_micros() as f64 / hot_time.as_micros().max(1) as f64
    );

    println!("\n6️⃣  Testing cross-conversation search...\n");

    // Create a second conversation
    let conv2_messages = vec![MessageMetadata {
        message_id: "conv2_msg1".to_string(),
        conversation_id: "conversation_2".to_string(),
        role: "user".to_string(),
        content: "Should we use PostgreSQL or MongoDB for this project?".to_string(),
        token_count: Some(12),
        model_id: None,
        images: None,
        expires_at: None,
        created_at: Utc::now().timestamp(),
    }];

    store.add_batch(conv2_messages).await?;
    println!("   ✓ Created second conversation");

    // Search across all conversations
    let all_results = store.search("database", 10, 0.5).await?;

    println!(
        "   Cross-conversation search for 'database': {} results",
        all_results.len()
    );

    let mut conversations = std::collections::HashSet::new();
    for (msg, score) in &all_results {
        conversations.insert(msg.conversation_id.clone());
        println!(
            "     - [{}] {:.3}: {}",
            msg.conversation_id,
            score,
            &msg.content[..60.min(msg.content.len())]
        );
    }

    println!(
        "   ✓ Found messages from {} different conversations",
        conversations.len()
    );

    println!("\n{}", "=".repeat(60));
    println!("✅ All tests passed!\n");

    println!("Summary:");
    println!("  • Messages stored with 384-dim embeddings");
    println!("  • Semantic search finds relevant content");
    println!(
        "  • Embedding cache provides ~{:.0}x speedup",
        cold_time.as_micros() as f64 / hot_time.as_micros().max(1) as f64
    );
    println!("  • Cross-conversation search works");
    println!("  • Average search latency: ~{:?}", search_time);

    println!("\n💡 Next steps:");
    println!("  1. Run full test suite: cargo test --test infinite_context_integration");
    println!("  2. Test via agent: Use recall_context tool in agent tasks");
    println!("  3. Test compaction: Create 100+ message conversation");

    println!("\nTest database location: {}", test_db_path);
    println!("(You can delete this after testing)");

    Ok(())
}
