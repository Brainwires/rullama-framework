//! Example: MessageStore — conversation messages with vector search
//!
//! Demonstrates creating a LanceDB-backed `MessageStore`, adding messages,
//! searching by semantic similarity, and listing messages by conversation.
//!
//! Run: cargo run -p brainwires-cli --example message_store

use std::sync::Arc;

use anyhow::Result;
use brainwires_storage::{CachedEmbeddingProvider, LanceDatabase};
use brainwires_stores::{MessageMetadata, MessageStore};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create a temporary LanceDB database
    let tmp_dir = tempfile::tempdir()?;
    let db_path = tmp_dir.path().join("example_messages.lance");
    let db = Arc::new(LanceDatabase::new(db_path.to_string_lossy()).await?);

    // 2. Initialise the embedding provider (all-MiniLM-L6-v2, 384 dims)
    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
    println!(
        "Embedding model: {} ({} dimensions)",
        embeddings.model_name(),
        embeddings.dimension(),
    );

    // 3. Create the message store and ensure the backing table exists
    let store = MessageStore::new(Arc::clone(&db), Arc::clone(&embeddings));
    store.ensure_table().await?;
    println!("MessageStore initialised.\n");

    // 4. Insert some demo messages
    let conversation_id = "conv-demo-1";
    let messages = vec![
        MessageMetadata {
            message_id: "msg-1".into(),
            conversation_id: conversation_id.into(),
            role: "user".into(),
            content: "How do I implement a binary search tree in Rust?".into(),
            token_count: Some(12),
            model_id: None,
            images: None,
            created_at: chrono::Utc::now().timestamp(),
            expires_at: None,
        },
        MessageMetadata {
            message_id: "msg-2".into(),
            conversation_id: conversation_id.into(),
            role: "assistant".into(),
            content: "You can implement a BST using an enum with Box pointers for left and right children. Each node stores a key and optional value.".into(),
            token_count: Some(28),
            model_id: Some("gpt-4".into()),
            images: None,
            created_at: chrono::Utc::now().timestamp(),
            expires_at: None,
        },
        MessageMetadata {
            message_id: "msg-3".into(),
            conversation_id: conversation_id.into(),
            role: "user".into(),
            content: "What about balancing? Should I use a red-black tree?".into(),
            token_count: Some(11),
            model_id: None,
            images: None,
            created_at: chrono::Utc::now().timestamp(),
            expires_at: None,
        },
        MessageMetadata {
            message_id: "msg-4".into(),
            conversation_id: "conv-demo-2".into(),
            role: "user".into(),
            content: "Explain the difference between TCP and UDP protocols.".into(),
            token_count: Some(10),
            model_id: None,
            images: None,
            created_at: chrono::Utc::now().timestamp(),
            expires_at: None,
        },
    ];

    store.add_batch(messages).await?;
    println!("Inserted 4 messages across 2 conversations.\n");

    // 5. Search by semantic similarity (across all conversations)
    let query = "tree data structures and balancing";
    let results = store.search(query, 3, 0.0).await?;
    println!("Search: \"{}\"", query);
    for (msg, score) in &results {
        println!(
            "  [{:.3}] ({}) {}: {}",
            score, msg.message_id, msg.role, msg.content,
        );
    }
    println!();

    // 6. Search within a single conversation
    let results = store
        .search_conversation(conversation_id, "balancing algorithms", 2, 0.0)
        .await?;
    println!("Search within conversation \"{}\":", conversation_id);
    for (msg, score) in &results {
        println!("  [{:.3}] {}", score, msg.content);
    }
    println!();

    // 7. List all messages for a conversation
    let conv_messages = store.get_by_conversation(conversation_id).await?;
    println!(
        "Conversation \"{}\" has {} messages:",
        conversation_id,
        conv_messages.len(),
    );
    for msg in &conv_messages {
        println!("  {} ({}): {}", msg.message_id, msg.role, msg.content);
    }

    println!("\nDone.");
    Ok(())
}
