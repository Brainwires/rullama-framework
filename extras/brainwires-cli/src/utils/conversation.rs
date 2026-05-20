// Conversation management with local LanceDB storage
use crate::config::PlatformPaths;
#[allow(deprecated)]
use crate::config::constants::COMPACTION_THRESHOLD_TOKENS;
use crate::storage::{
    CachedEmbeddingProvider, ConversationStore, DocumentMetadata, DocumentScope,
    DocumentSearchRequest, DocumentSearchResult, DocumentStore, FileContent, FileContextManager,
    ImageFormat, ImageMetadata, ImageSearchRequest, ImageSearchResult, ImageStorage, ImageStore,
    LanceDatabase, MessageMetadata, MessageStore, VectorDatabase,
};
use crate::types::message::{Message, MessageContent, Role};
use crate::utils::context_builder::{ContextBuilder, ContextBuilderConfig};
use crate::utils::entity_extraction::{EntityExtractor, EntityStore};
use anyhow::{Context, Result};
use brainwires::knowledge::RelationshipGraph;
use chrono::Utc;
use std::path::Path;
use std::sync::Arc;

/// Estimate token count for a string (roughly 4 chars per token)
pub fn estimate_tokens(text: &str) -> usize {
    // Simple heuristic: ~4 characters per token for English text
    // This is a rough approximation - actual tokenization varies by model
    text.len().div_ceil(4)
}

/// Estimate token count for a message
pub fn estimate_message_tokens(message: &Message) -> usize {
    let content_tokens = match &message.content {
        MessageContent::Text(text) => estimate_tokens(text),
        MessageContent::Blocks(blocks) => {
            blocks
                .iter()
                .map(|block| {
                    match block {
                        crate::types::message::ContentBlock::Text { text } => estimate_tokens(text),
                        crate::types::message::ContentBlock::Image { .. } => 85, // Base image tokens
                        crate::types::message::ContentBlock::ToolUse { input, .. } => {
                            estimate_tokens(&serde_json::to_string(input).unwrap_or_default())
                        }
                        crate::types::message::ContentBlock::ToolResult { content, .. } => {
                            estimate_tokens(content)
                        }
                    }
                })
                .sum()
        }
    };
    // Add overhead for role and message structure
    content_tokens + 4
}

pub struct ConversationManager {
    messages: Vec<Message>,
    max_tokens: usize,
    conversation_id: String,
    // Optional storage clients (lazy initialized)
    lance_client: Option<Arc<LanceDatabase>>,
    conversation_store: Option<Arc<ConversationStore>>,
    message_store: Option<Arc<MessageStore>>,
    document_store: Option<Arc<DocumentStore>>,
    image_store: Option<Arc<ImageStore>>,
    current_model: Option<String>,
    // Context enhancement
    context_builder: ContextBuilder,
    // Entity extraction and relationship graph
    entity_extractor: EntityExtractor,
    entity_store: EntityStore,
    relationship_graph: RelationshipGraph,
    // Smart file context management (chunking large files)
    file_context: FileContextManager,
}

impl ConversationManager {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_tokens,
            conversation_id: uuid::Uuid::new_v4().to_string(),
            lance_client: None,
            conversation_store: None,
            message_store: None,
            document_store: None,
            image_store: None,
            current_model: None,
            context_builder: ContextBuilder::new(),
            entity_extractor: EntityExtractor::new(),
            entity_store: EntityStore::new(),
            relationship_graph: RelationshipGraph::new(),
            file_context: FileContextManager::new(),
        }
    }

    /// Create with custom context builder configuration
    pub fn with_context_config(max_tokens: usize, config: ContextBuilderConfig) -> Self {
        Self {
            messages: Vec::new(),
            max_tokens,
            conversation_id: uuid::Uuid::new_v4().to_string(),
            lance_client: None,
            conversation_store: None,
            message_store: None,
            document_store: None,
            image_store: None,
            current_model: None,
            context_builder: ContextBuilder::with_config(config),
            entity_extractor: EntityExtractor::new(),
            entity_store: EntityStore::new(),
            relationship_graph: RelationshipGraph::new(),
            file_context: FileContextManager::new(),
        }
    }

    pub fn add_message(&mut self, message: Message) {
        // Extract entities from message content
        let content = match &message.content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    crate::types::message::ContentBlock::Text { text } => Some(text.clone()),
                    crate::types::message::ContentBlock::ToolResult { content, .. } => {
                        Some(content.clone())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        };

        // Generate a message ID for tracking
        let message_id = format!("msg_{}", self.messages.len());
        let timestamp = Utc::now().timestamp();

        // Extract and store entities
        let extraction = self.entity_extractor.extract(&content, &message_id);

        // Incrementally add to relationship graph
        for (name, entity_type) in &extraction.entities {
            use brainwires::knowledge::relationship_graph::GraphNode;
            self.relationship_graph.add_node(GraphNode {
                entity_name: name.clone(),
                entity_type: entity_type.clone(),
                message_ids: vec![message_id.clone()],
                mention_count: 1,
                importance: 0.5, // Base importance, will be recalculated
            });
        }

        // Add relationships to graph
        for rel in &extraction.relationships {
            self.relationship_graph.add_relationship(rel);
        }

        // Store entities
        self.entity_store
            .add_extraction(extraction, &message_id, timestamp);

        self.messages.push(message);
    }

    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn clear(&mut self) {
        self.messages.clear();
        self.entity_store = EntityStore::new();
        self.relationship_graph = RelationshipGraph::new();
    }

    /// Set the current model being used
    pub fn set_model(&mut self, model: String) {
        self.current_model = Some(model);
    }

    /// Create a snapshot of the conversation state
    /// Used for backup before clearing - entities will be re-extracted on restore
    pub fn snapshot(&self) -> Self {
        let mut snapshot = Self::new(self.max_tokens);
        snapshot.conversation_id = self.conversation_id.clone();
        snapshot.current_model = self.current_model.clone();
        // Re-add messages (this will re-extract entities)
        for msg in &self.messages {
            snapshot.add_message(msg.clone());
        }
        snapshot
    }

    /// Restore from a snapshot
    pub fn restore_from(&mut self, snapshot: Self) {
        self.messages = snapshot.messages;
        self.conversation_id = snapshot.conversation_id;
        self.current_model = snapshot.current_model;
        self.entity_store = snapshot.entity_store;
        self.relationship_graph = snapshot.relationship_graph;
    }

    /// Initialize storage clients (lazy initialization)
    async fn ensure_storage(&mut self) -> Result<()> {
        if self.lance_client.is_some() {
            return Ok(());
        }

        let db_path = PlatformPaths::conversations_db_path()?;
        let client = Arc::new(
            LanceDatabase::new(db_path.to_str().context("Invalid DB path")?)
                .await
                .context("Failed to create LanceDatabase")?,
        );

        let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
        client
            .initialize(embeddings.dimension())
            .await
            .context("Failed to initialize LanceDB tables")?;

        let conversation_store = Arc::new(ConversationStore::new(Arc::clone(&client)));
        let message_store = Arc::new(MessageStore::new(
            Arc::clone(&client),
            Arc::clone(&embeddings),
        ));
        // Create BM25 index path alongside the LanceDB path
        let bm25_path = db_path.parent().unwrap_or(&db_path).join("bm25_indices");
        let document_store = Arc::new(DocumentStore::new(
            Arc::new(client.connection().clone()),
            Arc::clone(&embeddings) as Arc<dyn crate::storage::EmbeddingProvider>,
            bm25_path,
        ));
        let image_store = Arc::new(ImageStore::new(
            Arc::clone(&client),
            Arc::clone(&embeddings),
        ));

        self.lance_client = Some(client);
        self.conversation_store = Some(conversation_store);
        self.message_store = Some(message_store);
        self.document_store = Some(document_store);
        self.image_store = Some(image_store);

        Ok(())
    }

    /// Estimate total tokens in current conversation
    pub fn estimate_total_tokens(&self) -> usize {
        self.messages.iter().map(estimate_message_tokens).sum()
    }

    // =========================================================================
    // DEPRECATED COMPACTION FUNCTIONS
    //
    // Manual compaction is deprecated. The system now uses automatic context
    // management via:
    // - TieredMemory: Auto-demotes hot→warm→cold tiers
    // - MessageStore: Persists all messages with embeddings for retrieval
    // - ContextBuilder: Dynamically composes context for each request
    // - SEAL: Resolves coreferences and tracks entities automatically
    //
    // These functions are kept for backward compatibility with existing
    // compacted conversations but should not be used for new development.
    // =========================================================================

    /// Check if compaction is needed based on token threshold
    ///
    /// **DEPRECATED**: Manual compaction is no longer needed. The system
    /// automatically manages context via TieredMemory and ContextBuilder.
    #[deprecated(
        note = "Manual compaction is deprecated. Use automatic context management instead."
    )]
    #[allow(deprecated)]
    pub fn needs_compaction(&self) -> bool {
        self.estimate_total_tokens() > COMPACTION_THRESHOLD_TOKENS
    }

    /// Compact conversation if needed (token management)
    ///
    /// **DEPRECATED**: Manual compaction is no longer needed. The system
    /// automatically manages context via TieredMemory and ContextBuilder.
    #[deprecated(
        note = "Manual compaction is deprecated. Use automatic context management instead."
    )]
    pub async fn compact_if_needed(&mut self) -> Result<bool> {
        // Save current state before potentially compacting
        self.save_to_db().await?;

        // Check if we need compaction
        #[allow(deprecated)]
        Ok(self.needs_compaction())
    }

    /// Generate a compaction prompt for the LLM to summarize the conversation
    ///
    /// **DEPRECATED**: Manual compaction is no longer needed. The system
    /// automatically manages context via TieredMemory and ContextBuilder.
    #[deprecated(
        note = "Manual compaction is deprecated. Use automatic context management instead."
    )]
    pub fn generate_compaction_prompt(
        &self,
        instructions: Option<&str>,
    ) -> Option<(String, usize)> {
        if self.messages.len() < 4 {
            return None; // Not enough messages to compact
        }

        // Keep the most recent messages (approximately 20% of context or at least 4 messages)
        let keep_recent = std::cmp::max(4, self.messages.len() / 5);
        let messages_to_summarize = self.messages.len() - keep_recent;

        if messages_to_summarize < 2 {
            return None; // Not enough messages to summarize
        }

        // Build the summary of messages to compact
        let mut conversation_text = String::new();
        for (i, msg) in self.messages.iter().take(messages_to_summarize).enumerate() {
            let role = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
                Role::Tool => "Tool",
            };
            let content = match &msg.content {
                MessageContent::Text(text) => text.clone(),
                MessageContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| match b {
                        crate::types::message::ContentBlock::Text { text } => Some(text.clone()),
                        crate::types::message::ContentBlock::ToolUse { name, .. } => {
                            Some(format!("[Tool call: {}]", name))
                        }
                        crate::types::message::ContentBlock::ToolResult { content, .. } => {
                            Some(format!(
                                "[Tool result: {}]",
                                &content[..std::cmp::min(100, content.len())]
                            ))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            conversation_text.push_str(&format!("[{}] {}: {}\n\n", i + 1, role, content));
        }

        let base_instructions = instructions.unwrap_or(
            "Summarize the key points, decisions made, code changes discussed, and any important context that should be preserved for continuing the conversation."
        );

        let prompt = format!(
            "Please summarize the following conversation history concisely while preserving important context:\n\n\
            {}\n\n\
            Instructions: {}\n\n\
            Provide a summary that captures:\n\
            - Key topics discussed\n\
            - Important decisions or conclusions\n\
            - Any code or file changes mentioned\n\
            - Context needed to continue the conversation\n\n\
            Keep the summary focused and under 500 words.",
            conversation_text, base_instructions
        );

        Some((prompt, messages_to_summarize))
    }

    /// Apply compaction by replacing old messages with a summary
    ///
    /// **DEPRECATED**: Manual compaction is no longer needed. The system
    /// automatically manages context via TieredMemory and ContextBuilder.
    #[deprecated(
        note = "Manual compaction is deprecated. Use automatic context management instead."
    )]
    pub fn apply_compaction(&mut self, summary: &str, messages_compacted: usize) {
        if messages_compacted >= self.messages.len() {
            return; // Safety check
        }

        // Remove the compacted messages
        let remaining_messages: Vec<Message> = self.messages.drain(messages_compacted..).collect();

        // Create a system message with the summary
        let summary_message = Message {
            role: Role::System,
            content: MessageContent::Text(format!(
                "[Conversation Summary - {} messages compacted]\n\n{}",
                messages_compacted, summary
            )),
            name: None,
            metadata: None,
        };

        // Replace messages with summary + remaining
        self.messages.clear();
        self.messages.push(summary_message);
        self.messages.extend(remaining_messages);
    }

    // =========================================================================
    // END DEPRECATED COMPACTION FUNCTIONS
    // =========================================================================

    /// Get conversation ID
    pub fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    /// Set conversation ID (for loading existing conversations)
    pub fn set_conversation_id(&mut self, id: String) {
        self.conversation_id = id;
    }

    /// Save messages to local LanceDB
    /// Messages are persisted locally, NOT sent to backend
    pub async fn save_to_db(&mut self) -> Result<()> {
        if self.messages.is_empty() {
            return Ok(());
        }

        self.ensure_storage().await?;

        let conversation_store = self
            .conversation_store
            .as_ref()
            .context("Conversation store not initialized")?;
        let message_store = self
            .message_store
            .as_ref()
            .context("Message store not initialized")?;

        // Check if conversation metadata exists, create if not
        let existing = conversation_store.get(&self.conversation_id).await?;
        if existing.is_none() {
            // Create conversation metadata with message count
            let title = self.generate_title();
            let message_count = self.messages.len() as i32;
            conversation_store
                .create(
                    self.conversation_id.clone(),
                    Some(title),
                    self.current_model.clone(),
                    Some(message_count),
                )
                .await
                .context("Failed to create conversation metadata")?;
        }

        // Convert and save messages
        let message_metas: Vec<MessageMetadata> = self
            .messages
            .iter()
            .map(|msg| self.message_to_metadata(msg))
            .collect::<Result<Vec<_>>>()?;

        // Save messages in batch
        message_store
            .add_batch(message_metas)
            .await
            .context("Failed to save messages")?;

        // Update conversation metadata with message count
        conversation_store
            .update(
                &self.conversation_id,
                None,
                Some(self.messages.len() as i32),
            )
            .await
            .context("Failed to update conversation metadata")?;

        Ok(())
    }

    /// Load messages from local LanceDB
    pub async fn load_from_db(&mut self, conversation_id: &str) -> Result<()> {
        self.ensure_storage().await?;

        let message_store = self
            .message_store
            .as_ref()
            .context("Message store not initialized")?;

        // Load messages for the conversation
        let message_metas = message_store
            .get_by_conversation(conversation_id)
            .await
            .context("Failed to load messages from DB")?;

        // Convert metadata to messages
        self.messages = message_metas
            .into_iter()
            .map(|meta| self.metadata_to_message(&meta))
            .collect::<Result<Vec<_>>>()?;

        self.conversation_id = conversation_id.to_string();

        Ok(())
    }

    /// Convert Message to MessageMetadata for storage
    fn message_to_metadata(&self, message: &Message) -> Result<MessageMetadata> {
        let content = match &message.content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Blocks(blocks) => {
                serde_json::to_string(blocks).context("Failed to serialize content blocks")?
            }
        };

        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "tool",
        }
        .to_string();

        // Extract images if any
        let images = if let MessageContent::Blocks(blocks) = &message.content {
            let has_images = blocks
                .iter()
                .any(|b| matches!(b, crate::types::message::ContentBlock::Image { .. }));
            if has_images {
                Some(serde_json::to_string(blocks).context("Failed to serialize images")?)
            } else {
                None
            }
        } else {
            None
        };

        // Estimate token count for the message
        let token_count = estimate_message_tokens(message) as i32;

        Ok(MessageMetadata {
            message_id: uuid::Uuid::new_v4().to_string(),
            conversation_id: self.conversation_id.clone(),
            role,
            content,
            token_count: Some(token_count),
            model_id: self.current_model.clone(),
            images,
            created_at: Utc::now().timestamp(),
            expires_at: None,
        })
    }

    /// Convert MessageMetadata back to Message
    fn metadata_to_message(&self, meta: &MessageMetadata) -> Result<Message> {
        let role = match meta.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            "tool" => Role::Tool,
            _ => Role::User, // Default to User for unknown roles
        };

        // Try to parse as blocks first, fall back to text
        let content = if meta.content.trim_start().starts_with('[') {
            match serde_json::from_str(&meta.content) {
                Ok(blocks) => MessageContent::Blocks(blocks),
                Err(_) => MessageContent::Text(meta.content.clone()),
            }
        } else {
            MessageContent::Text(meta.content.clone())
        };

        Ok(Message {
            role,
            content,
            name: None,
            metadata: None,
        })
    }

    /// Search through conversation history using semantic search
    ///
    /// This allows searching for specific content that may have been compacted
    /// out of the current context window. Messages remain stored with embeddings
    /// even after compaction.
    pub async fn search_history(
        &mut self,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<(MessageMetadata, f32)>> {
        self.ensure_storage().await?;

        let message_store = self
            .message_store
            .as_ref()
            .context("Message store not initialized")?;

        // Search within this conversation's history
        message_store
            .search_conversation(&self.conversation_id, query, limit, min_score)
            .await
            .context("Failed to search conversation history")
    }

    /// Get enhanced context with auto-injected relevant history
    ///
    /// This method uses retrieval gating to determine if RAG lookup is needed,
    /// then injects relevant historical context into the messages.
    ///
    /// Use this instead of `get_messages()` when sending to the API to
    /// automatically benefit from infinite context recall.
    pub async fn get_enhanced_context(&mut self, user_query: &str) -> Result<Vec<Message>> {
        // If no storage, return raw messages
        if self.message_store.is_none()
            && let Err(e) = self.ensure_storage().await
        {
            // If storage init fails, fall back to raw messages
            tracing::debug!("Storage init failed, using raw context: {}", e);
            return Ok(self.messages.clone());
        }

        let message_store = match self.message_store.as_ref() {
            Some(store) => store,
            None => return Ok(self.messages.clone()),
        };

        // Use context builder to inject personal knowledge and relevant history
        self.context_builder
            .build_full_context(
                &self.messages,
                user_query,
                message_store,
                &self.conversation_id,
            )
            .await
    }

    /// Check if the conversation has been compacted
    pub fn has_compaction(&self) -> bool {
        ContextBuilder::has_compaction_summary(&self.messages)
    }

    /// Get entity store for querying extracted entities
    pub fn entity_store(&self) -> &EntityStore {
        &self.entity_store
    }

    /// Get relationship graph
    pub fn relationship_graph(&self) -> &RelationshipGraph {
        &self.relationship_graph
    }

    /// Rebuild relationship graph from entity store
    pub fn rebuild_graph(&mut self) {
        self.relationship_graph = RelationshipGraph::from_entity_store(&self.entity_store);
    }

    /// Get entities related to a query (searches entity names)
    pub fn find_related_entities(&self, query: &str, limit: usize) -> Vec<String> {
        self.relationship_graph
            .search(query, limit)
            .into_iter()
            .map(|n| n.entity_name.clone())
            .collect()
    }

    /// Get message IDs where an entity was mentioned
    pub fn get_entity_messages(&self, entity_name: &str) -> Vec<String> {
        self.entity_store.get_message_ids(entity_name)
    }

    // =========================================================================
    // DOCUMENT MANAGEMENT
    //
    // Methods for attaching and searching documents within a conversation.
    // Documents are indexed with embeddings and BM25 for hybrid search.
    // =========================================================================

    /// Attach a document to this conversation
    ///
    /// Reads the file, chunks it, generates embeddings, and stores in LanceDB.
    /// Supports PDF, Markdown, Plain Text, and DOCX formats.
    ///
    /// # Arguments
    /// * `file_path` - Path to the document file
    ///
    /// # Returns
    /// Document metadata including document_id and chunk count
    pub async fn attach_document(&mut self, file_path: &Path) -> Result<DocumentMetadata> {
        self.ensure_storage().await?;

        let document_store = self
            .document_store
            .as_ref()
            .context("Document store not initialized")?;

        let scope = DocumentScope::Conversation(self.conversation_id.clone());

        document_store
            .index_file(file_path, scope)
            .await
            .with_context(|| format!("Failed to attach document: {}", file_path.display()))
    }

    /// Attach a document from bytes (for uploads without a file path)
    ///
    /// # Arguments
    /// * `bytes` - Document content as bytes
    /// * `file_name` - Original file name (used for type detection)
    ///
    /// # Returns
    /// Document metadata including document_id and chunk count
    pub async fn attach_document_bytes(
        &mut self,
        bytes: &[u8],
        file_name: &str,
    ) -> Result<DocumentMetadata> {
        use crate::storage::DocumentType;

        self.ensure_storage().await?;

        let document_store = self
            .document_store
            .as_ref()
            .context("Document store not initialized")?;

        let scope = DocumentScope::Conversation(self.conversation_id.clone());
        let file_type = DocumentType::from_path(Path::new(file_name));

        document_store
            .index_bytes(bytes, file_name, file_type, scope)
            .await
            .with_context(|| format!("Failed to attach document: {}", file_name))
    }

    /// Search documents attached to this conversation
    ///
    /// Uses hybrid search (vector + BM25) by default for best results.
    ///
    /// # Arguments
    /// * `query` - Search query
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    /// Search results with chunk content and scores
    pub async fn search_documents(
        &mut self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<DocumentSearchResult>> {
        self.ensure_storage().await?;

        let document_store = self
            .document_store
            .as_ref()
            .context("Document store not initialized")?;

        let request = DocumentSearchRequest {
            query: query.to_string(),
            conversation_id: Some(self.conversation_id.clone()),
            project_id: None,
            limit,
            min_score: 0.5,
            hybrid: true,
            file_type: None,
        };

        document_store
            .search(request)
            .await
            .context("Failed to search documents")
    }

    /// Search documents with custom options
    ///
    /// # Arguments
    /// * `request` - Search request with all options
    ///
    /// # Returns
    /// Search results with chunk content and scores
    pub async fn search_documents_advanced(
        &mut self,
        request: DocumentSearchRequest,
    ) -> Result<Vec<DocumentSearchResult>> {
        self.ensure_storage().await?;

        let document_store = self
            .document_store
            .as_ref()
            .context("Document store not initialized")?;

        document_store
            .search(request)
            .await
            .context("Failed to search documents")
    }

    /// List all documents attached to this conversation
    ///
    /// # Returns
    /// List of document metadata for all attached documents
    pub async fn list_documents(&mut self) -> Result<Vec<DocumentMetadata>> {
        self.ensure_storage().await?;

        let document_store = self
            .document_store
            .as_ref()
            .context("Document store not initialized")?;

        document_store
            .list_by_conversation(&self.conversation_id)
            .await
            .context("Failed to list documents")
    }

    /// Remove a document from this conversation
    ///
    /// # Arguments
    /// * `document_id` - ID of the document to remove
    ///
    /// # Returns
    /// `true` if document was deleted, `false` if not found
    pub async fn remove_document(&mut self, document_id: &str) -> Result<bool> {
        self.ensure_storage().await?;

        let document_store = self
            .document_store
            .as_ref()
            .context("Document store not initialized")?;

        document_store
            .delete_document(document_id)
            .await
            .with_context(|| format!("Failed to remove document: {}", document_id))
    }

    /// Get document store for direct access
    pub fn document_store(&self) -> Option<&Arc<DocumentStore>> {
        self.document_store.as_ref()
    }

    // =========================================================================
    // SMART FILE READING
    // =========================================================================

    /// Read file with smart chunking for large files
    ///
    /// For files smaller than MAX_DIRECT_FILE_CHARS (8000 chars), returns full content.
    /// For larger files, returns only relevant chunks based on the query context.
    /// Tracks files in context to avoid re-injection.
    ///
    /// # Arguments
    /// * `path` - Path to the file
    /// * `query` - Optional query to find relevant chunks in large files
    ///
    /// # Returns
    /// Formatted file content suitable for context injection
    pub async fn read_file_smart(&mut self, path: &str, query: Option<&str>) -> Result<String> {
        let content = self
            .file_context
            .get_file_content(path, query)
            .await
            .with_context(|| format!("Failed to read file: {}", path))?;

        Ok(FileContextManager::format_content(&content))
    }

    /// Read specific lines from a file
    ///
    /// # Arguments
    /// * `path` - Path to the file
    /// * `start_line` - Starting line number (1-indexed)
    /// * `end_line` - Ending line number (1-indexed)
    ///
    /// # Returns
    /// Formatted file content with the specified lines
    pub async fn read_file_lines(
        &mut self,
        path: &str,
        start_line: usize,
        end_line: usize,
    ) -> Result<String> {
        let content = self
            .file_context
            .get_file_lines(path, start_line, end_line)
            .await
            .with_context(|| format!("Failed to read lines from file: {}", path))?;

        Ok(FileContextManager::format_content(&content))
    }

    /// Check if a file is already in the current context
    ///
    /// Use this to avoid re-injecting files that have already been shown.
    pub fn is_file_in_context(&self, path: &str) -> bool {
        self.file_context.is_in_context(path)
    }

    /// Get the number of files currently tracked in context
    pub fn context_file_count(&self) -> usize {
        self.file_context.context_file_count()
    }

    /// Clear file context tracking
    ///
    /// Call this when starting a new conversation turn if you want to
    /// allow re-injection of previously seen files.
    pub fn clear_file_context(&mut self) {
        self.file_context.clear_context();
    }

    /// Get raw file content enum for custom handling
    ///
    /// Use this when you need to handle the different file content types
    /// (Full, Chunked, AlreadyInContext) separately.
    pub async fn get_file_content_raw(
        &mut self,
        path: &str,
        query: Option<&str>,
    ) -> Result<FileContent> {
        self.file_context
            .get_file_content(path, query)
            .await
            .with_context(|| format!("Failed to get file content: {}", path))
    }

    // =========================================================================
    // END DOCUMENT MANAGEMENT
    // =========================================================================

    // =========================================================================
    // IMAGE ANALYSIS MANAGEMENT
    //
    // Methods for storing and searching analyzed images within a conversation.
    // Images are indexed with embeddings generated from LLM analysis text.
    // =========================================================================

    /// Store an analyzed image in this conversation
    ///
    /// Stores the image with its LLM-generated analysis for semantic search.
    ///
    /// # Arguments
    /// * `image_bytes` - Raw image bytes
    /// * `analysis` - LLM-generated description/analysis of the image
    /// * `format` - Image format (PNG, JPEG, etc.)
    ///
    /// # Returns
    /// Image metadata including image_id
    pub async fn store_image_analysis(
        &mut self,
        image_bytes: &[u8],
        analysis: String,
        format: ImageFormat,
    ) -> Result<ImageMetadata> {
        self.ensure_storage().await?;

        let image_store = self
            .image_store
            .as_ref()
            .context("Image store not initialized")?;

        image_store
            .store_from_bytes(image_bytes, analysis, self.conversation_id.clone(), format)
            .await
            .context("Failed to store image analysis")
    }

    /// Store an analyzed image with full metadata
    ///
    /// # Arguments
    /// * `metadata` - Complete image metadata including analysis
    /// * `storage` - How to store the image (base64, file path, or URL)
    ///
    /// # Returns
    /// Stored image metadata
    pub async fn store_image_with_metadata(
        &mut self,
        metadata: ImageMetadata,
        storage: ImageStorage,
    ) -> Result<ImageMetadata> {
        self.ensure_storage().await?;

        let image_store = self
            .image_store
            .as_ref()
            .context("Image store not initialized")?;

        image_store
            .store(metadata, storage)
            .await
            .context("Failed to store image")
    }

    /// Search images in this conversation by description/analysis
    ///
    /// Uses semantic search on the LLM-generated analysis text.
    ///
    /// # Arguments
    /// * `query` - Search query (e.g., "architecture diagram", "screenshot of error")
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    /// Search results with image metadata and scores
    pub async fn search_images(
        &mut self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ImageSearchResult>> {
        self.ensure_storage().await?;

        let image_store = self
            .image_store
            .as_ref()
            .context("Image store not initialized")?;

        let request = ImageSearchRequest::new(query)
            .with_conversation(self.conversation_id.clone())
            .with_limit(limit)
            .with_min_score(0.5);

        image_store
            .search(request)
            .await
            .context("Failed to search images")
    }

    /// Search images with custom options
    ///
    /// # Arguments
    /// * `request` - Search request with all options
    ///
    /// # Returns
    /// Search results with image metadata and scores
    pub async fn search_images_advanced(
        &mut self,
        request: ImageSearchRequest,
    ) -> Result<Vec<ImageSearchResult>> {
        self.ensure_storage().await?;

        let image_store = self
            .image_store
            .as_ref()
            .context("Image store not initialized")?;

        image_store
            .search(request)
            .await
            .context("Failed to search images")
    }

    /// List all images in this conversation
    ///
    /// # Returns
    /// List of image metadata for all stored images
    pub async fn list_images(&mut self) -> Result<Vec<ImageMetadata>> {
        self.ensure_storage().await?;

        let image_store = self
            .image_store
            .as_ref()
            .context("Image store not initialized")?;

        image_store
            .list_by_conversation(&self.conversation_id)
            .await
            .context("Failed to list images")
    }

    /// Get a specific image by ID
    ///
    /// # Arguments
    /// * `image_id` - ID of the image to retrieve
    ///
    /// # Returns
    /// Image metadata if found
    pub async fn get_image(&mut self, image_id: &str) -> Result<Option<ImageMetadata>> {
        self.ensure_storage().await?;

        let image_store = self
            .image_store
            .as_ref()
            .context("Image store not initialized")?;

        image_store
            .get(image_id)
            .await
            .context("Failed to get image")
    }

    /// Get image data (base64 or file path)
    ///
    /// # Arguments
    /// * `image_id` - ID of the image
    ///
    /// # Returns
    /// Image storage containing the data
    pub async fn get_image_data(&mut self, image_id: &str) -> Result<Option<ImageStorage>> {
        self.ensure_storage().await?;

        let image_store = self
            .image_store
            .as_ref()
            .context("Image store not initialized")?;

        image_store
            .get_image_data(image_id)
            .await
            .context("Failed to get image data")
    }

    /// Remove an image from this conversation
    ///
    /// # Arguments
    /// * `image_id` - ID of the image to remove
    ///
    /// # Returns
    /// `true` if image was deleted
    pub async fn remove_image(&mut self, image_id: &str) -> Result<bool> {
        self.ensure_storage().await?;

        let image_store = self
            .image_store
            .as_ref()
            .context("Image store not initialized")?;

        image_store
            .delete(image_id)
            .await
            .with_context(|| format!("Failed to remove image: {}", image_id))
    }

    /// Get image store for direct access
    pub fn image_store(&self) -> Option<&Arc<ImageStore>> {
        self.image_store.as_ref()
    }

    // =========================================================================
    // END IMAGE ANALYSIS MANAGEMENT
    // =========================================================================

    /// Generate a title for the conversation from the first user message
    fn generate_title(&self) -> String {
        for message in &self.messages {
            if matches!(message.role, Role::User) {
                let text = match &message.content {
                    MessageContent::Text(t) => t.clone(),
                    MessageContent::Blocks(blocks) => {
                        // Extract text from first text block
                        blocks
                            .iter()
                            .find_map(|b| {
                                if let crate::types::message::ContentBlock::Text { text } = b {
                                    Some(text.clone())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_else(|| "New Conversation".to_string())
                    }
                };

                // Take first 50 characters
                return if text.len() > 50 {
                    format!("{}...", &text[..50])
                } else {
                    text
                };
            }
        }

        "New Conversation".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::MessageContent;

    #[test]
    fn test_conversation_manager_new() {
        let manager = ConversationManager::new(4096);
        assert_eq!(manager.max_tokens, 4096);
        assert!(manager.get_messages().is_empty());
    }

    #[test]
    fn test_add_message() {
        let mut manager = ConversationManager::new(4096);
        let msg = Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        };

        manager.add_message(msg);
        assert_eq!(manager.get_messages().len(), 1);
    }

    #[test]
    fn test_get_messages() {
        let mut manager = ConversationManager::new(4096);
        assert!(manager.get_messages().is_empty());

        manager.add_message(Message {
            role: Role::User,
            content: MessageContent::Text("Test".to_string()),
            name: None,
            metadata: None,
        });

        assert_eq!(manager.get_messages().len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut manager = ConversationManager::new(4096);
        manager.add_message(Message {
            role: Role::User,
            content: MessageContent::Text("Test".to_string()),
            name: None,
            metadata: None,
        });

        assert_eq!(manager.get_messages().len(), 1);
        manager.clear();
        assert!(manager.get_messages().is_empty());
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_compact_if_needed() {
        let mut manager = ConversationManager::new(4096);
        let result = manager.compact_if_needed().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_conversation_id() {
        let manager = ConversationManager::new(4096);
        let id = manager.conversation_id();
        assert!(!id.is_empty());
    }

    #[test]
    fn test_set_conversation_id() {
        let mut manager = ConversationManager::new(4096);
        let new_id = "test-conversation-id".to_string();
        manager.set_conversation_id(new_id.clone());
        assert_eq!(manager.conversation_id(), new_id);
    }

    #[test]
    fn test_multiple_messages() {
        let mut manager = ConversationManager::new(4096);

        manager.add_message(Message {
            role: Role::User,
            content: MessageContent::Text("Message 1".to_string()),
            name: None,
            metadata: None,
        });

        manager.add_message(Message {
            role: Role::Assistant,
            content: MessageContent::Text("Response 1".to_string()),
            name: None,
            metadata: None,
        });

        assert_eq!(manager.get_messages().len(), 2);
    }

    #[test]
    fn test_set_model() {
        let mut manager = ConversationManager::new(4096);
        manager.set_model("gpt-4".to_string());
        assert_eq!(manager.current_model, Some("gpt-4".to_string()));
    }

    #[test]
    fn test_generate_title() {
        let mut manager = ConversationManager::new(4096);
        manager.add_message(Message {
            role: Role::User,
            content: MessageContent::Text("Hello, how are you?".to_string()),
            name: None,
            metadata: None,
        });

        let title = manager.generate_title();
        assert_eq!(title, "Hello, how are you?");
    }

    #[test]
    fn test_generate_title_long() {
        let mut manager = ConversationManager::new(4096);
        let long_text =
            "This is a very long message that exceeds fifty characters and should be truncated";
        manager.add_message(Message {
            role: Role::User,
            content: MessageContent::Text(long_text.to_string()),
            name: None,
            metadata: None,
        });

        let title = manager.generate_title();
        assert_eq!(title.len(), 53); // 50 chars + "..."
        assert!(title.ends_with("..."));
    }
}
