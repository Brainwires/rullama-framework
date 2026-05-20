use anyhow::{Context, Result};
use chrono::{DateTime, Local};
use clap::Subcommand;
use console::style;

use crate::config::PlatformPaths;
use crate::storage::{
    CachedEmbeddingProvider, ConversationStore, LanceDatabase, MessageStore, VectorDatabase,
};
use std::sync::Arc;

#[derive(Subcommand)]
pub enum HistoryCommands {
    /// List all saved conversations
    List {
        #[arg(short, long)]
        limit: Option<usize>,
    },

    /// Search conversations by semantic similarity
    Search {
        query: String,

        #[arg(short, long, default_value = "10")]
        limit: usize,

        #[arg(short, long, default_value = "0.5")]
        min_score: f32,
    },

    /// Show conversation details
    Show {
        conversation_id: String,

        #[arg(short, long)]
        messages: bool,
    },

    /// Delete a conversation
    Delete {
        conversation_id: String,

        #[arg(short, long)]
        confirm: bool,
    },

    /// Open and resume a conversation
    Open { conversation_id: String },
}

pub async fn handle_history(cmd: HistoryCommands) -> Result<()> {
    match cmd {
        HistoryCommands::List { limit } => handle_list(limit).await,
        HistoryCommands::Search {
            query,
            limit,
            min_score,
        } => handle_search(&query, limit, min_score).await,
        HistoryCommands::Show {
            conversation_id,
            messages,
        } => handle_show(&conversation_id, messages).await,
        HistoryCommands::Delete {
            conversation_id,
            confirm,
        } => handle_delete(&conversation_id, confirm).await,
        HistoryCommands::Open { conversation_id } => handle_open(&conversation_id).await,
    }
}

/// List all saved conversations
pub async fn handle_list(limit: Option<usize>) -> Result<()> {
    let (client, _conversation_store, _message_store) = initialize_storage().await?;

    let conversation_store = ConversationStore::new(client.clone());
    let conversations = conversation_store.list(limit).await?;

    if conversations.is_empty() {
        println!("{}", style("No conversations found").yellow());
        return Ok(());
    }

    println!("\n{}\n", style("Saved Conversations:").cyan().bold());

    for conv in conversations {
        let created_at = DateTime::from_timestamp(conv.created_at, 0)
            .map(|dt| {
                dt.with_timezone(&Local)
                    .format("%Y-%m-%d %H:%M:%S")
                    .to_string()
            })
            .unwrap_or_else(|| "Unknown".to_string());

        let updated_at = DateTime::from_timestamp(conv.updated_at, 0)
            .map(|dt| {
                dt.with_timezone(&Local)
                    .format("%Y-%m-%d %H:%M:%S")
                    .to_string()
            })
            .unwrap_or_else(|| "Unknown".to_string());

        let title = conv.title.unwrap_or_else(|| "Untitled".to_string());
        let model = conv.model_id.unwrap_or_else(|| "unknown".to_string());

        println!("{}", style(&conv.conversation_id).cyan().bold());
        println!("  Title:     {}", title);
        println!("  Model:     {}", style(&model).green());
        println!("  Messages:  {}", conv.message_count);
        println!("  Created:   {}", created_at);
        println!("  Updated:   {}", updated_at);
        println!();
    }

    Ok(())
}

/// Search conversations by semantic similarity
pub async fn handle_search(query: &str, limit: usize, min_score: f32) -> Result<()> {
    let (client, _conversation_store, message_store) = initialize_storage().await?;

    println!("\n{} '{}'\n", style("Searching for:").cyan().bold(), query);

    let results = message_store.search(query, limit, min_score).await?;

    if results.is_empty() {
        println!("{}", style("No matching conversations found").yellow());
        return Ok(());
    }

    println!("{} result(s):\n", results.len());

    for (message, score) in results {
        let conversation_store = ConversationStore::new(client.clone());
        let conv = conversation_store
            .get(&message.conversation_id)
            .await?
            .context("Conversation not found")?;

        let title = conv.title.unwrap_or_else(|| "Untitled".to_string());

        println!(
            "{} (score: {:.2})",
            style(&conv.conversation_id).cyan().bold(),
            score
        );
        println!("  Title:   {}", title);
        println!("  Role:    {}", message.role);
        println!("  Content: {}", truncate_text(&message.content, 100));
        println!();
    }

    Ok(())
}

/// Delete a conversation by ID
pub async fn handle_delete(conversation_id: &str, confirm: bool) -> Result<()> {
    let (client, _conversation_store, message_store) = initialize_storage().await?;

    let conversation_store = ConversationStore::new(client.clone());

    // Check if conversation exists
    let conv = match conversation_store.get(conversation_id).await? {
        Some(c) => c,
        None => {
            println!(
                "{}",
                style(format!("Conversation '{}' not found", conversation_id)).red()
            );
            return Ok(());
        }
    };
    let title = conv.title.unwrap_or_else(|| "Untitled".to_string());

    if !confirm {
        println!(
            "{}",
            style("Are you sure you want to delete this conversation?").yellow()
        );
        println!("  ID:       {}", conversation_id);
        println!("  Title:    {}", title);
        println!("  Messages: {}", conv.message_count);
        println!("\n{}", style("Use --confirm to delete").dim());
        return Ok(());
    }

    // Delete messages first
    message_store
        .delete_by_conversation(conversation_id)
        .await?;

    // Delete conversation
    conversation_store.delete(conversation_id).await?;

    println!(
        "{}",
        style(format!("Deleted conversation '{}'", title)).green()
    );

    Ok(())
}

/// Open and resume a conversation
pub async fn handle_open(conversation_id: &str) -> Result<()> {
    let (client, _conversation_store, _message_store) = initialize_storage().await?;

    let conversation_store = ConversationStore::new(client.clone());

    // Check if conversation exists
    let conv = conversation_store
        .get(conversation_id)
        .await?
        .context(format!("Conversation '{}' not found", conversation_id))?;

    let title = conv.title.unwrap_or_else(|| "Untitled".to_string());
    let model = conv.model_id.unwrap_or_else(|| "default".to_string());

    println!(
        "{}",
        style(format!("Opening conversation: {}", title))
            .cyan()
            .bold()
    );
    println!("  ID:       {}", conversation_id);
    println!("  Model:    {}", style(&model).green());
    println!("  Messages: {}\n", conv.message_count);

    // Call chat handler with the conversation_id to resume
    super::chat::handle_chat_with_conversation(
        Some(model),
        None,
        None,
        Some(conversation_id.to_string()),
        false,  // Don't output JSON when resuming from history
        false,  // Not quiet
        "full", // Full format
        None,   // No MDAP config when resuming from history
        None,   // No backend URL override - use default
    )
    .await
}

/// Show details of a specific conversation
pub async fn handle_show(conversation_id: &str, show_messages: bool) -> Result<()> {
    let (client, _conversation_store, message_store) = initialize_storage().await?;

    let conversation_store = ConversationStore::new(client.clone());

    // Get conversation metadata
    let conv = conversation_store
        .get(conversation_id)
        .await?
        .context(format!("Conversation '{}' not found", conversation_id))?;

    let created_at = DateTime::from_timestamp(conv.created_at, 0)
        .map(|dt| {
            dt.with_timezone(&Local)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        })
        .unwrap_or_else(|| "Unknown".to_string());

    let updated_at = DateTime::from_timestamp(conv.updated_at, 0)
        .map(|dt| {
            dt.with_timezone(&Local)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        })
        .unwrap_or_else(|| "Unknown".to_string());

    let title = conv.title.unwrap_or_else(|| "Untitled".to_string());
    let model = conv.model_id.unwrap_or_else(|| "unknown".to_string());

    println!("\n{}\n", style("Conversation Details:").cyan().bold());
    println!("  ID:        {}", style(&conv.conversation_id).cyan());
    println!("  Title:     {}", title);
    println!("  Model:     {}", style(&model).green());
    println!("  Messages:  {}", conv.message_count);
    println!("  Created:   {}", created_at);
    println!("  Updated:   {}", updated_at);

    if show_messages {
        println!("\n{}\n", style("Messages:").cyan().bold());

        let messages = message_store.get_by_conversation(conversation_id).await?;

        for (i, msg) in messages.iter().enumerate() {
            let msg_time = DateTime::from_timestamp(msg.created_at, 0)
                .map(|dt| dt.with_timezone(&Local).format("%H:%M:%S").to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            println!(
                "{} [{}] {}",
                style(format!("#{}", i + 1)).dim(),
                msg_time,
                style(&msg.role).bold()
            );
            println!("{}", msg.content);
            println!();
        }
    } else {
        println!("\n{}", style("Use --messages to show all messages").dim());
    }

    Ok(())
}

/// Initialize storage clients
async fn initialize_storage() -> Result<(
    Arc<LanceDatabase>,
    Arc<ConversationStore>,
    Arc<MessageStore>,
)> {
    let db_path = PlatformPaths::conversations_db_path()?;
    let client = Arc::new(
        LanceDatabase::new(db_path.to_str().context("Invalid DB path")?)
            .await
            .context("Failed to create LanceDB client")?,
    );

    let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
    client
        .initialize(embeddings.dimension())
        .await
        .context("Failed to initialize LanceDB tables")?;

    let conversation_store = Arc::new(ConversationStore::new(Arc::clone(&client)));
    let message_store = Arc::new(MessageStore::new(Arc::clone(&client), embeddings));

    conversation_store
        .ensure_table()
        .await
        .context("Failed to ensure conversations table")?;
    message_store
        .ensure_table()
        .await
        .context("Failed to ensure messages table")?;

    Ok((client, conversation_store, message_store))
}

/// Truncate text to a maximum length
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}
