//! Example: TieredMemory — hot/warm/cold memory hierarchy
//!
//! Demonstrates configuring `TieredMemory` with custom tier thresholds,
//! adding messages with importance scores, and performing multi-factor
//! adaptive search that blends similarity, recency, and importance.
//!
//! Run: cargo run -p brainwires-memory --example tiered_memory

use std::sync::Arc;

use anyhow::Result;
use brainwires_memory::{
    MemoryTier, MessageMetadata, MessageStore, TieredMemory, TieredMemoryConfig,
};
use brainwires_storage::{CachedEmbeddingProvider, LanceDatabase};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create a temporary LanceDB database
    let tmp_dir = tempfile::tempdir()?;
    let db_path = tmp_dir.path().join("tiered_memory.lance");
    let db = Arc::new(LanceDatabase::new(db_path.to_string_lossy()).await?);

    // 2. Initialise embeddings
    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);

    // 3. Create the hot-tier message store
    let hot_store = Arc::new(MessageStore::new(Arc::clone(&db), Arc::clone(&embeddings)));
    hot_store.ensure_table().await?;

    // 4. Configure tiered memory with custom thresholds
    let config = TieredMemoryConfig {
        hot_retention_hours: 12,       // demote from hot after 12 hours
        warm_retention_hours: 72,      // demote from warm after 3 days
        hot_importance_threshold: 0.4, // need 0.4+ importance to stay hot
        warm_importance_threshold: 0.2,
        max_hot_messages: 500,
        max_warm_summaries: 2000,
        session_ttl_secs: None, // no automatic expiry
        ..TieredMemoryConfig::default()
    };

    println!("TieredMemory config:");
    println!("  hot retention:  {} hours", config.hot_retention_hours);
    println!("  warm retention: {} hours", config.warm_retention_hours);
    println!("  max hot msgs:   {}", config.max_hot_messages);
    println!();

    let mut tiered = TieredMemory::new(
        Arc::clone(&hot_store),
        Arc::clone(&db),
        Arc::clone(&embeddings),
        config,
    )
    .await;

    // 5. Add messages with varying importance scores
    let conversation_id = "conv-tiered-1";
    let entries = [
        (
            "Architecture decision: we will use an event-driven design with CQRS.",
            0.95,
        ),
        ("Let me check the test output... all 42 tests pass.", 0.2),
        (
            "The database schema has three main tables: users, projects, and events.",
            0.85,
        ),
        ("Can you add a newline at the end of that file?", 0.05),
        (
            "We decided to use PostgreSQL with pgvector for production storage.",
            0.9,
        ),
    ];

    for (i, (content, importance)) in entries.iter().enumerate() {
        let msg = MessageMetadata {
            message_id: format!("tmsg-{}", i + 1),
            conversation_id: conversation_id.into(),
            role: if i % 2 == 0 { "user" } else { "assistant" }.into(),
            content: content.to_string(),
            token_count: Some(content.split_whitespace().count() as i32),
            model_id: None,
            images: None,
            created_at: chrono::Utc::now().timestamp(),
            expires_at: None,
        };
        tiered.add_message(msg, *importance).await?;
        println!("  Added (importance={:.2}): {}", importance, content);
    }
    println!();

    // 6. Adaptive search — searches hot tier first, falls through to warm/cold
    let query = "database storage decisions";
    let results = tiered.search_adaptive(query, Some(conversation_id)).await?;
    println!("Adaptive search: \"{}\"", query);
    for r in &results {
        println!("  [{:.3}] [tier={:?}] {}", r.score, r.tier, r.content);
    }
    println!();

    // 7. Multi-factor search — blends similarity, recency, and importance
    let results = tiered
        .search_adaptive_multi_factor(query, Some(conversation_id))
        .await?;
    println!("Multi-factor search: \"{}\"", query);
    for r in &results {
        if let Some(ref mf) = r.multi_factor_score {
            println!(
                "  [combined={:.3}] sim={:.3} rec={:.3} imp={:.3} | {:?}: {}",
                mf.combined, mf.similarity, mf.recency, mf.importance, r.tier, r.content,
            );
        }
    }
    println!();

    // 8. Check tier statistics
    let stats = tiered.get_stats().await?;
    println!("Tier statistics:");
    println!("  Hot:          {} entries", stats.hot_count);
    println!("  Warm:         {} entries", stats.warm_count);
    println!("  Cold:         {} entries", stats.cold_count);
    println!("  Mental Model: {} entries", stats.mental_model_count);
    println!("  Total:        {} tracked", stats.total_tracked);

    // 9. Identify demotion candidates (lowest retention score)
    let candidates = tiered.get_demotion_candidates(MemoryTier::Hot, 2).await?;
    println!("\nDemotion candidates (lowest retention score):");
    for id in &candidates {
        println!("  {}", id);
    }

    println!("\nDone.");
    Ok(())
}
