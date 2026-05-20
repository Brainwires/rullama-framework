//! Context Builder with Smart Injection
//!
//! Automatically enhances conversation context with relevant historical
//! information before sending to the API. Uses retrieval gating to avoid
//! unnecessary lookups and injects only high-relevance content.
//!
//! ## Local LLM Integration
//!
//! When the `llama-cpp-2` feature is enabled, the context builder can use
//! a `RelevanceScorer` to re-rank retrieved items by semantic relevance,
//! replacing the fixed 0.75 threshold with intelligent scoring.

use anyhow::Result;

use crate::storage::{MessageMetadata, MessageStore, TieredMemory, TieredSearchResult};
use crate::types::message::{Message, MessageContent, Role};
use crate::utils::retrieval_gate::{RetrievalNeed, classify_retrieval_need, needs_retrieval};
use brainwires_seal::{ResolvedReference, SealProcessingResult};

use brainwires::reasoning::RelevanceScorer;

/// Configuration for context building
#[derive(Debug, Clone)]
pub struct ContextBuilderConfig {
    /// Minimum score to inject retrieved content (0.0-1.0)
    pub injection_threshold: f32,
    /// Maximum tokens to inject from retrieved content
    pub max_inject_tokens: usize,
    /// Maximum number of retrieved items to inject
    pub max_inject_items: usize,
    /// Whether to use retrieval gating
    pub use_gating: bool,
    /// Minimum retrieval need level to trigger lookup
    pub min_retrieval_need: RetrievalNeed,

    // SEAL integration options
    /// Enable SEAL-enhanced context building
    pub enable_seal_enhancement: bool,
    /// Use SEAL's resolved query (with coreferences resolved) for retrieval
    pub use_resolved_query: bool,
    /// Inject entity context from coreference resolutions
    pub inject_entity_context: bool,
    /// Adjust threshold based on SEAL's quality score
    pub quality_aware_threshold: bool,

    // Personal Knowledge System (PKS) options
    /// Enable personal knowledge injection
    pub enable_personal_knowledge: bool,
    /// Minimum confidence for personal facts to inject (0.0-1.0)
    pub personal_fact_min_confidence: f32,
    /// Maximum number of personal facts to inject
    pub personal_fact_max_items: usize,
    /// Include transient context facts (current project, etc.)
    pub include_context_facts: bool,
}

impl Default for ContextBuilderConfig {
    fn default() -> Self {
        Self {
            injection_threshold: 0.75,
            max_inject_tokens: 2000,
            max_inject_items: 3,
            use_gating: true,
            min_retrieval_need: RetrievalNeed::Medium,
            // SEAL defaults
            enable_seal_enhancement: true,
            use_resolved_query: true,
            inject_entity_context: true,
            quality_aware_threshold: true,
            // PKS defaults
            enable_personal_knowledge: true,
            personal_fact_min_confidence: 0.5,
            personal_fact_max_items: 15,
            include_context_facts: true,
        }
    }
}

/// Builds enhanced context for API calls
#[derive(Clone)]
pub struct ContextBuilder {
    config: ContextBuilderConfig,
}

impl ContextBuilder {
    /// Create a new context builder with default config
    pub fn new() -> Self {
        Self {
            config: ContextBuilderConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ContextBuilderConfig) -> Self {
        Self { config }
    }

    /// Check if context has been compacted (has a summary message)
    pub fn has_compaction_summary(messages: &[Message]) -> bool {
        messages.iter().any(|m| {
            if m.role == Role::System
                && let MessageContent::Text(text) = &m.content
            {
                return text.contains("[Compacted Context]")
                    || text.contains("Summary of earlier conversation");
            }
            false
        })
    }

    /// Determine if retrieval should be attempted
    pub fn should_retrieve(&self, user_query: &str, messages: &[Message]) -> bool {
        // No retrieval if no compaction has occurred
        if !Self::has_compaction_summary(messages) {
            return false;
        }

        if !self.config.use_gating {
            return true; // Always retrieve if gating disabled
        }

        let recent_count = messages.len();
        let has_compaction = Self::has_compaction_summary(messages);

        // Use simple gating first
        if needs_retrieval(user_query, recent_count, has_compaction) {
            return true;
        }

        // Use detailed classification for edge cases
        let (need, _confidence) = classify_retrieval_need(user_query, recent_count, has_compaction);

        match need {
            RetrievalNeed::High => true,
            RetrievalNeed::Medium => self.config.min_retrieval_need != RetrievalNeed::High,
            RetrievalNeed::Low => self.config.min_retrieval_need == RetrievalNeed::Low,
            RetrievalNeed::None => false,
        }
    }

    /// Build enhanced context with retrieved information
    ///
    /// Returns the original messages plus any injected context
    pub async fn build_context(
        &self,
        messages: &[Message],
        user_query: &str,
        message_store: &MessageStore,
        conversation_id: &str,
    ) -> Result<Vec<Message>> {
        // Check if retrieval is needed
        if !self.should_retrieve(user_query, messages) {
            return Ok(messages.to_vec());
        }

        // Search for relevant historical content
        let results = message_store
            .search_conversation(
                conversation_id,
                user_query,
                self.config.max_inject_items + 2,
                0.5,
            )
            .await?;

        // Filter by threshold
        let relevant: Vec<_> = results
            .into_iter()
            .filter(|(_, score)| *score >= self.config.injection_threshold)
            .take(self.config.max_inject_items)
            .collect();

        if relevant.is_empty() {
            return Ok(messages.to_vec());
        }

        // Build injection content
        let injection = self.format_injection(&relevant);

        // Insert into message list
        let mut result = messages.to_vec();
        let insert_pos = self.find_injection_point(&result);

        let injection_message = Message {
            role: Role::System,
            content: MessageContent::Text(injection),
            name: None,
            metadata: None,
        };

        result.insert(insert_pos, injection_message);

        Ok(result)
    }

    /// Build context using tiered memory for adaptive search
    pub async fn build_context_tiered(
        &self,
        messages: &[Message],
        user_query: &str,
        tiered_memory: &mut TieredMemory,
        conversation_id: &str,
    ) -> Result<Vec<Message>> {
        // Check if retrieval is needed
        if !self.should_retrieve(user_query, messages) {
            return Ok(messages.to_vec());
        }

        // Search using adaptive tiered search
        let results = tiered_memory
            .search_adaptive(user_query, Some(conversation_id))
            .await?;

        // Filter by threshold
        let relevant: Vec<_> = results
            .into_iter()
            .filter(|r| r.score >= self.config.injection_threshold)
            .take(self.config.max_inject_items)
            .collect();

        if relevant.is_empty() {
            return Ok(messages.to_vec());
        }

        // Build injection content
        let injection = self.format_tiered_injection(&relevant);

        // Insert into message list
        let mut result = messages.to_vec();
        let insert_pos = self.find_injection_point(&result);

        let injection_message = Message {
            role: Role::System,
            content: MessageContent::Text(injection),
            name: None,
            metadata: None,
        };

        result.insert(insert_pos, injection_message);

        Ok(result)
    }

    /// Build context using SEAL's resolved query and entity information
    ///
    /// This method uses SEAL's coreference-resolved query for better semantic
    /// matching and optionally injects entity context from resolutions.
    ///
    /// # Arguments
    /// * `messages` - Current conversation messages
    /// * `user_query` - Original user query (used as fallback)
    /// * `seal_result` - SEAL processing result with resolved query and resolutions
    /// * `message_store` - Message store for retrieval
    /// * `conversation_id` - Current conversation ID
    ///
    /// # Returns
    /// Enhanced message list with injected context
    pub async fn build_context_with_seal(
        &self,
        messages: &[Message],
        user_query: &str,
        seal_result: &SealProcessingResult,
        message_store: &MessageStore,
        conversation_id: &str,
    ) -> Result<Vec<Message>> {
        // Skip if SEAL enhancement is disabled
        if !self.config.enable_seal_enhancement {
            return self
                .build_context(messages, user_query, message_store, conversation_id)
                .await;
        }

        // Determine which query to use for search
        let search_query = if self.config.use_resolved_query {
            &seal_result.resolved_query
        } else {
            user_query
        };

        // Check if retrieval is needed (use resolved query for better gating)
        if !self.should_retrieve(search_query, messages) {
            return Ok(messages.to_vec());
        }

        // Adjust threshold based on quality score if enabled
        let threshold = if self.config.quality_aware_threshold && seal_result.quality_score > 0.0 {
            // Higher quality queries can use stricter threshold
            // Lower quality queries relax threshold to get more context
            let adjustment = 1.0 - (1.0 - seal_result.quality_score) * 0.3;
            self.config.injection_threshold * adjustment
        } else {
            self.config.injection_threshold
        };

        // Search for relevant historical content using resolved query
        let results = message_store
            .search_conversation(
                conversation_id,
                search_query,
                self.config.max_inject_items + 2,
                0.5,
            )
            .await?;

        // Filter by adjusted threshold
        let relevant: Vec<_> = results
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .take(self.config.max_inject_items)
            .collect();

        if relevant.is_empty() && seal_result.resolutions.is_empty() {
            return Ok(messages.to_vec());
        }

        // Build injection content with SEAL enhancements
        let injection = self.format_seal_injection(&relevant, &seal_result.resolutions);

        // Insert into message list
        let mut result = messages.to_vec();
        let insert_pos = self.find_injection_point(&result);

        let injection_message = Message {
            role: Role::System,
            content: MessageContent::Text(injection),
            name: None,
            metadata: None,
        };

        result.insert(insert_pos, injection_message);

        Ok(result)
    }

    /// Build context with SEAL using tiered memory
    ///
    /// Combines SEAL's resolved query with tiered memory for adaptive search
    /// across hot/warm/cold memory tiers.
    pub async fn build_context_with_seal_tiered(
        &self,
        messages: &[Message],
        user_query: &str,
        seal_result: &SealProcessingResult,
        tiered_memory: &mut TieredMemory,
        conversation_id: &str,
    ) -> Result<Vec<Message>> {
        // Skip if SEAL enhancement is disabled
        if !self.config.enable_seal_enhancement {
            return self
                .build_context_tiered(messages, user_query, tiered_memory, conversation_id)
                .await;
        }

        // Determine which query to use for search
        let search_query = if self.config.use_resolved_query {
            &seal_result.resolved_query
        } else {
            user_query
        };

        // Check if retrieval is needed
        if !self.should_retrieve(search_query, messages) {
            return Ok(messages.to_vec());
        }

        // Adjust threshold based on quality score
        let threshold = if self.config.quality_aware_threshold && seal_result.quality_score > 0.0 {
            let adjustment = 1.0 - (1.0 - seal_result.quality_score) * 0.3;
            self.config.injection_threshold * adjustment
        } else {
            self.config.injection_threshold
        };

        // Search using adaptive tiered search with resolved query
        let results = tiered_memory
            .search_adaptive(search_query, Some(conversation_id))
            .await?;

        // Filter by threshold
        let relevant: Vec<_> = results
            .into_iter()
            .filter(|r| r.score >= threshold)
            .take(self.config.max_inject_items)
            .collect();

        if relevant.is_empty() && seal_result.resolutions.is_empty() {
            return Ok(messages.to_vec());
        }

        // Build injection content with SEAL enhancements
        let injection = self.format_seal_tiered_injection(&relevant, &seal_result.resolutions);

        // Insert into message list
        let mut result = messages.to_vec();
        let insert_pos = self.find_injection_point(&result);

        let injection_message = Message {
            role: Role::System,
            content: MessageContent::Text(injection),
            name: None,
            metadata: None,
        };

        result.insert(insert_pos, injection_message);

        Ok(result)
    }

    /// Format retrieved messages for injection
    fn format_injection(&self, results: &[(MessageMetadata, f32)]) -> String {
        let mut parts = vec![
            "[Retrieved Context]".to_string(),
            "The following information was retrieved from earlier in this conversation:"
                .to_string(),
            String::new(),
        ];

        for (msg, score) in results {
            let truncated = self.truncate_content(&msg.content, 500);
            parts.push(format!(
                "- [{}] (relevance: {:.0}%): {}",
                msg.role,
                score * 100.0,
                truncated
            ));
        }

        parts.push(String::new());
        parts.push("[End Retrieved Context]".to_string());

        parts.join("\n")
    }

    /// Format tiered search results for injection
    fn format_tiered_injection(&self, results: &[TieredSearchResult]) -> String {
        let mut parts = vec![
            "[Retrieved Context]".to_string(),
            "The following information was retrieved from conversation history:".to_string(),
            String::new(),
        ];

        for result in results {
            let truncated = self.truncate_content(&result.content, 500);
            let tier_label = match result.tier {
                crate::storage::MemoryTier::Hot => "recent",
                crate::storage::MemoryTier::Warm => "summarized",
                crate::storage::MemoryTier::Cold => "archived",
                crate::storage::MemoryTier::MentalModel => "mental-model",
            };
            parts.push(format!(
                "- [{}] (relevance: {:.0}%): {}",
                tier_label,
                result.score * 100.0,
                truncated
            ));
        }

        parts.push(String::new());
        parts.push("[End Retrieved Context]".to_string());

        parts.join("\n")
    }

    /// Format retrieved messages with SEAL entity context for injection
    ///
    /// Enhances the standard injection format with resolved reference information
    /// to help the model understand entity relationships in the conversation.
    fn format_seal_injection(
        &self,
        results: &[(MessageMetadata, f32)],
        resolutions: &[ResolvedReference],
    ) -> String {
        let mut parts = Vec::new();

        // Add entity context header if we have resolutions
        if self.config.inject_entity_context && !resolutions.is_empty() {
            parts.push("[Entity Context]".to_string());
            parts.push("Resolved references from current query:".to_string());

            for resolution in resolutions {
                if resolution.confidence >= 0.5 {
                    parts.push(format!(
                        "- \"{}\" → {} ({:.0}% confidence)",
                        resolution.reference.text,
                        resolution.antecedent,
                        resolution.confidence * 100.0
                    ));
                }
            }
            parts.push(String::new());
        }

        // Add retrieved context if we have results
        if !results.is_empty() {
            parts.push("[Retrieved Context]".to_string());
            parts.push(
                "The following information was retrieved from earlier in this conversation:"
                    .to_string(),
            );
            parts.push(String::new());

            for (msg, score) in results {
                let truncated = self.truncate_content(&msg.content, 500);
                parts.push(format!(
                    "- [{}] (relevance: {:.0}%): {}",
                    msg.role,
                    score * 100.0,
                    truncated
                ));
            }

            parts.push(String::new());
            parts.push("[End Retrieved Context]".to_string());
        }

        parts.join("\n")
    }

    /// Format tiered search results with SEAL entity context for injection
    ///
    /// Combines tiered memory results with resolved reference information
    /// for enhanced context injection.
    fn format_seal_tiered_injection(
        &self,
        results: &[TieredSearchResult],
        resolutions: &[ResolvedReference],
    ) -> String {
        let mut parts = Vec::new();

        // Add entity context header if we have resolutions
        if self.config.inject_entity_context && !resolutions.is_empty() {
            parts.push("[Entity Context]".to_string());
            parts.push("Resolved references from current query:".to_string());

            for resolution in resolutions {
                if resolution.confidence >= 0.5 {
                    parts.push(format!(
                        "- \"{}\" → {} ({:.0}% confidence)",
                        resolution.reference.text,
                        resolution.antecedent,
                        resolution.confidence * 100.0
                    ));
                }
            }
            parts.push(String::new());
        }

        // Add retrieved context if we have results
        if !results.is_empty() {
            parts.push("[Retrieved Context]".to_string());
            parts.push(
                "The following information was retrieved from conversation history:".to_string(),
            );
            parts.push(String::new());

            for result in results {
                let truncated = self.truncate_content(&result.content, 500);
                let tier_label = match result.tier {
                    crate::storage::MemoryTier::Hot => "recent",
                    crate::storage::MemoryTier::Warm => "summarized",
                    crate::storage::MemoryTier::Cold => "archived",
                    crate::storage::MemoryTier::MentalModel => "mental-model",
                };
                parts.push(format!(
                    "- [{}] (relevance: {:.0}%): {}",
                    tier_label,
                    result.score * 100.0,
                    truncated
                ));
            }

            parts.push(String::new());
            parts.push("[End Retrieved Context]".to_string());
        }

        parts.join("\n")
    }

    /// Find the best position to inject retrieved context
    fn find_injection_point(&self, messages: &[Message]) -> usize {
        // Look for compaction summary and insert after it
        for (i, msg) in messages.iter().enumerate() {
            if msg.role == Role::System
                && let MessageContent::Text(text) = &msg.content
                && (text.contains("[Compacted Context]") || text.contains("Summary of earlier"))
            {
                return i + 1;
            }
        }

        // If no compaction summary, insert after system prompts
        let mut last_system = 0;
        for (i, msg) in messages.iter().enumerate() {
            if msg.role == Role::System {
                last_system = i + 1;
            } else {
                break;
            }
        }

        last_system
    }

    /// Truncate content to max chars while preserving word boundaries
    fn truncate_content(&self, content: &str, max_chars: usize) -> String {
        if content.len() <= max_chars {
            return content.to_string();
        }

        // Find last space before max_chars
        let truncate_at = content[..max_chars].rfind(' ').unwrap_or(max_chars);

        format!("{}...", &content[..truncate_at])
    }

    /// Estimate tokens in injection content
    pub fn estimate_injection_tokens(&self, content: &str) -> usize {
        // Rough estimate: 4 chars per token
        content.len().div_ceil(4)
    }

    /// Build context with local LLM re-ranking for improved relevance
    ///
    /// Uses the RelevanceScorer to semantically re-rank retrieved items,
    /// replacing the fixed threshold with intelligent scoring.
    pub async fn build_context_with_reranking(
        &self,
        messages: &[Message],
        user_query: &str,
        message_store: &MessageStore,
        conversation_id: &str,
        scorer: Option<&RelevanceScorer>,
    ) -> Result<Vec<Message>> {
        // Check if retrieval is needed
        if !self.should_retrieve(user_query, messages) {
            return Ok(messages.to_vec());
        }

        // Search for relevant historical content (get more candidates for re-ranking)
        let results = message_store
            .search_conversation(
                conversation_id,
                user_query,
                self.config.max_inject_items * 2, // Get more for re-ranking
                0.4,                              // Lower initial threshold
            )
            .await?;

        if results.is_empty() {
            return Ok(messages.to_vec());
        }

        // Re-rank with local LLM if scorer is available
        let relevant = if let Some(scorer) = scorer {
            let items: Vec<_> = results
                .iter()
                .map(|(msg, score)| (msg.content.as_str(), *score))
                .collect();

            let reranked = scorer.rerank(user_query, &items).await;

            // Map back to MessageMetadata with new scores
            reranked
                .into_iter()
                .take(self.config.max_inject_items)
                .filter_map(|r| {
                    results
                        .get(r.original_index)
                        .map(|(msg, _)| (msg.clone(), r.relevance_score))
                })
                .collect::<Vec<_>>()
        } else {
            // Fallback to threshold-based filtering
            results
                .into_iter()
                .filter(|(_, score)| *score >= self.config.injection_threshold)
                .take(self.config.max_inject_items)
                .collect()
        };

        if relevant.is_empty() {
            return Ok(messages.to_vec());
        }

        // Build injection content
        let injection = self.format_injection(&relevant);

        // Insert into message list
        let mut result = messages.to_vec();
        let insert_pos = self.find_injection_point(&result);

        let injection_message = Message {
            role: Role::System,
            content: MessageContent::Text(injection),
            name: None,
            metadata: None,
        };

        result.insert(insert_pos, injection_message);

        Ok(result)
    }

    /// Build context using tiered memory with re-ranking
    pub async fn build_context_tiered_with_reranking(
        &self,
        messages: &[Message],
        user_query: &str,
        tiered_memory: &mut TieredMemory,
        conversation_id: &str,
        scorer: Option<&RelevanceScorer>,
    ) -> Result<Vec<Message>> {
        // Check if retrieval is needed
        if !self.should_retrieve(user_query, messages) {
            return Ok(messages.to_vec());
        }

        // Search using adaptive tiered search
        let results = tiered_memory
            .search_adaptive(user_query, Some(conversation_id))
            .await?;

        if results.is_empty() {
            return Ok(messages.to_vec());
        }

        // Re-rank with local LLM if scorer is available
        let relevant = if let Some(scorer) = scorer {
            let items: Vec<_> = results
                .iter()
                .map(|r| (r.content.as_str(), r.score))
                .collect();

            let reranked = scorer.rerank(user_query, &items).await;

            // Map back to TieredSearchResult with new scores
            reranked
                .into_iter()
                .take(self.config.max_inject_items)
                .filter_map(|r| {
                    results
                        .get(r.original_index)
                        .map(|orig| TieredSearchResult {
                            content: orig.content.clone(),
                            score: r.relevance_score,
                            tier: orig.tier,
                            original_message_id: orig.original_message_id.clone(),
                            metadata: orig.metadata.clone(),
                            multi_factor_score: None,
                        })
                })
                .collect::<Vec<_>>()
        } else {
            // Fallback to threshold-based filtering
            results
                .into_iter()
                .filter(|r| r.score >= self.config.injection_threshold)
                .take(self.config.max_inject_items)
                .collect()
        };

        if relevant.is_empty() {
            return Ok(messages.to_vec());
        }

        // Build injection content
        let injection = self.format_tiered_injection(&relevant);

        // Insert into message list
        let mut result = messages.to_vec();
        let insert_pos = self.find_injection_point(&result);

        let injection_message = Message {
            role: Role::System,
            content: MessageContent::Text(injection),
            name: None,
            metadata: None,
        };

        result.insert(insert_pos, injection_message);

        Ok(result)
    }

    /// Extract file suggestions from retrieved messages.
    /// Returns paths mentioned in the content that exist on disk.
    pub fn suggest_files_from_results(
        &self,
        results: &[(MessageMetadata, f32)],
    ) -> Vec<std::path::PathBuf> {
        use crate::types::WorkingSet;

        let mut suggestions = Vec::new();
        for (msg, _) in results {
            let files = WorkingSet::extract_file_references(&msg.content);
            for f in files {
                if !suggestions.contains(&f) {
                    suggestions.push(f);
                }
            }
        }
        suggestions
    }

    /// Extract file suggestions from tiered search results.
    pub fn suggest_files_from_tiered(
        &self,
        results: &[TieredSearchResult],
    ) -> Vec<std::path::PathBuf> {
        use crate::types::WorkingSet;

        let mut suggestions = Vec::new();
        for result in results {
            let files = WorkingSet::extract_file_references(&result.content);
            for f in files {
                if !suggestions.contains(&f) {
                    suggestions.push(f);
                }
            }
        }
        suggestions
    }

    /// Build personal knowledge context injection string
    ///
    /// Loads relevant personal facts from the PKS cache and formats them
    /// for injection into the conversation context.
    ///
    /// # Arguments
    /// * `user_query` - Optional query to match relevant facts against
    ///
    /// # Returns
    /// Formatted string for injection, or empty string if PKS disabled or no facts
    pub fn build_personal_context(&self, user_query: Option<&str>) -> String {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::personal::{
            PersonalFactMatcher, PersonalKnowledgeCache,
        };

        if !self.config.enable_personal_knowledge {
            return String::new();
        }

        // Load personal facts from cache
        let facts = match PlatformPaths::personal_knowledge_db() {
            Ok(db_path) => match PersonalKnowledgeCache::new(&db_path, 100) {
                Ok(cache) => cache.all_facts().cloned().collect::<Vec<_>>(),
                Err(e) => {
                    tracing::warn!("Failed to load personal knowledge cache: {}", e);
                    return String::new();
                }
            },
            Err(e) => {
                tracing::warn!("Failed to get personal knowledge db path: {}", e);
                return String::new();
            }
        };

        if facts.is_empty() {
            return String::new();
        }

        // Create matcher and get relevant facts
        let matcher = PersonalFactMatcher::new(
            self.config.personal_fact_min_confidence,
            self.config.personal_fact_max_items,
            self.config.include_context_facts,
        );

        let relevant: Vec<_> = matcher
            .get_relevant_facts(facts.iter(), user_query)
            .into_iter()
            .collect();

        if relevant.is_empty() {
            return String::new();
        }

        // Format for context
        matcher.format_for_context(&relevant)
    }

    /// Build messages with personal knowledge context prepended
    ///
    /// Adds a system message with the user's profile information at the
    /// optimal injection point (after system prompts, before conversation).
    ///
    /// # Arguments
    /// * `messages` - Current conversation messages
    /// * `user_query` - Optional query to match relevant facts against
    ///
    /// # Returns
    /// Enhanced message list with personal context, or original if PKS disabled
    pub fn inject_personal_context(
        &self,
        messages: &[Message],
        user_query: Option<&str>,
    ) -> Vec<Message> {
        let personal_context = self.build_personal_context(user_query);

        if personal_context.is_empty() {
            return messages.to_vec();
        }

        // Create personal context message
        let personal_message = Message {
            role: Role::System,
            content: MessageContent::Text(personal_context),
            name: None,
            metadata: None,
        };

        // Insert at optimal position
        let mut result = messages.to_vec();
        let insert_pos = self.find_injection_point(&result);
        result.insert(insert_pos, personal_message);

        result
    }

    /// Build complete context with both retrieval and personal knowledge
    ///
    /// This is the recommended method for building context as it combines:
    /// 1. Personal knowledge (user profile, preferences)
    /// 2. Retrieved conversation history (if gating allows)
    ///
    /// # Arguments
    /// * `messages` - Current conversation messages
    /// * `user_query` - User's current query
    /// * `message_store` - Message store for retrieval
    /// * `conversation_id` - Current conversation ID
    ///
    /// # Returns
    /// Fully enhanced message list
    pub async fn build_full_context(
        &self,
        messages: &[Message],
        user_query: &str,
        message_store: &MessageStore,
        conversation_id: &str,
    ) -> Result<Vec<Message>> {
        // Start with personal knowledge injection
        let with_personal = self.inject_personal_context(messages, Some(user_query));

        // Then add retrieval context
        self.build_context(&with_personal, user_query, message_store, conversation_id)
            .await
    }

    /// Build complete context with tiered memory and personal knowledge
    pub async fn build_full_context_tiered(
        &self,
        messages: &[Message],
        user_query: &str,
        tiered_memory: &mut TieredMemory,
        conversation_id: &str,
    ) -> Result<Vec<Message>> {
        // Start with personal knowledge injection
        let with_personal = self.inject_personal_context(messages, Some(user_query));

        // Then add retrieval context
        self.build_context_tiered(&with_personal, user_query, tiered_memory, conversation_id)
            .await
    }

    /// Build context with full SEAL + Knowledge integration
    ///
    /// This is the most comprehensive context building method, combining:
    /// 1. Personal Knowledge (PKS) - user profile, preferences, facts
    /// 2. Behavioral Knowledge (BKS) - shared truths and patterns
    /// 3. Entity Context - SEAL's coreference resolutions
    /// 4. Retrieved History - semantically relevant past messages
    /// 5. Quality-aware thresholds - adaptive based on SEAL quality scores
    ///
    /// # Arguments
    /// * `messages` - Current conversation messages
    /// * `user_query` - Original user query
    /// * `seal_result` - SEAL processing result with resolutions and quality score
    /// * `coordinator` - Knowledge coordinator for BKS/PKS access
    /// * `message_store` - Message store for retrieval
    /// * `conversation_id` - Current conversation ID
    ///
    /// # Returns
    /// Fully enhanced message list with all context sources integrated
    pub async fn build_context_with_seal_and_knowledge(
        &self,
        messages: &[Message],
        user_query: &str,
        seal_result: Option<&SealProcessingResult>,
        coordinator: Option<&brainwires_seal::SealKnowledgeCoordinator>,
        message_store: &MessageStore,
        conversation_id: &str,
    ) -> Result<Vec<Message>> {
        let mut enhanced_messages = messages.to_vec();

        // Step 1: Inject Personal Knowledge (PKS) if enabled
        if self.config.enable_personal_knowledge
            && let (Some(coordinator), Some(seal_res)) = (coordinator, seal_result)
            && let Ok(Some(pks_context)) = coordinator.get_pks_context(seal_res).await
        {
            // Insert PKS as system message at start
            let pks_message = Message {
                role: Role::System,
                content: MessageContent::Text(pks_context),
                name: None,
                metadata: None,
            };
            enhanced_messages.insert(0, pks_message);
        }

        // Step 2: Inject Entity Context (SEAL resolutions) if enabled
        if self.config.inject_entity_context
            && let Some(seal_res) = seal_result
            && !seal_res.resolutions.is_empty()
        {
            let entity_context = self.format_entity_resolutions(&seal_res.resolutions);
            let entity_message = Message {
                role: Role::System,
                content: MessageContent::Text(entity_context),
                name: None,
                metadata: None,
            };
            // Insert after PKS (if present) or at start
            let insert_pos = if self.config.enable_personal_knowledge && coordinator.is_some() {
                1
            } else {
                0
            };
            enhanced_messages.insert(insert_pos, entity_message);
        }

        // Step 3: Inject Behavioral Knowledge (BKS) if enabled and quality is high
        if let (Some(coordinator_ref), Some(seal_res)) = (coordinator, seal_result) {
            // Use coordinator's config instead of ContextBuilder's config
            if seal_res.quality_score >= coordinator_ref.config().min_seal_quality_for_bks_boost
                && let Ok(Some(bks_context)) = coordinator_ref.get_bks_context(user_query).await
            {
                let bks_message = Message {
                    role: Role::System,
                    content: MessageContent::Text(bks_context),
                    name: None,
                    metadata: None,
                };
                // Insert after PKS and entity context
                let insert_pos = if self.config.enable_personal_knowledge {
                    if self.config.inject_entity_context {
                        2
                    } else {
                        1
                    }
                } else if self.config.inject_entity_context {
                    1
                } else {
                    0
                };
                enhanced_messages.insert(insert_pos, bks_message);
            }
        }

        // Step 4: Perform semantic search with SEAL-enhanced query
        let search_query = if self.config.use_resolved_query {
            if let Some(seal_res) = seal_result {
                &seal_res.resolved_query
            } else {
                user_query
            }
        } else {
            user_query
        };

        // Check if retrieval is needed
        if !self.should_retrieve(search_query, &enhanced_messages) {
            return Ok(enhanced_messages);
        }

        // Adjust injection threshold based on SEAL quality
        let threshold = if let Some(seal_res) = seal_result {
            if self.config.quality_aware_threshold && seal_res.quality_score > 0.0 {
                coordinator
                    .map(|c| {
                        c.adjust_retrieval_threshold(
                            self.config.injection_threshold,
                            seal_res.quality_score,
                        )
                    })
                    .unwrap_or(self.config.injection_threshold)
            } else {
                self.config.injection_threshold
            }
        } else {
            self.config.injection_threshold
        };

        // Perform retrieval with adjusted threshold
        let results = message_store
            .search_conversation(
                conversation_id,
                search_query,
                self.config.max_inject_items + 2,
                0.5,
            )
            .await?;

        // Filter by threshold
        let relevant: Vec<_> = results
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .take(self.config.max_inject_items)
            .collect();

        // Step 5: Inject retrieved context if we have any
        if !relevant.is_empty() {
            let injection = self.format_injection(&relevant);
            let inject_pos = self.find_injection_point(&enhanced_messages);

            let injection_message = Message {
                role: Role::System,
                content: MessageContent::Text(injection),
                name: None,
                metadata: None,
            };

            enhanced_messages.insert(inject_pos, injection_message);
        }

        Ok(enhanced_messages)
    }

    /// Format entity resolutions for context injection
    fn format_entity_resolutions(&self, resolutions: &[ResolvedReference]) -> String {
        let mut parts = vec![
            "[Entity Context]".to_string(),
            "Resolved references from current query:".to_string(),
        ];

        for resolution in resolutions {
            if resolution.confidence >= 0.5 {
                parts.push(format!(
                    "- \"{}\" → {} ({:.0}% confidence)",
                    resolution.reference.text,
                    resolution.antecedent,
                    resolution.confidence * 100.0
                ));
            }
        }

        parts.push(String::new());
        parts.join("\n")
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(role: Role, content: &str) -> Message {
        Message {
            role,
            content: MessageContent::Text(content.to_string()),
            name: None,
            metadata: None,
        }
    }

    #[test]
    fn test_has_compaction_summary() {
        let messages = vec![
            make_message(Role::System, "You are a helpful assistant"),
            make_message(Role::User, "Hello"),
        ];
        assert!(!ContextBuilder::has_compaction_summary(&messages));

        let compacted = vec![
            make_message(
                Role::System,
                "[Compacted Context] Summary of earlier conversation...",
            ),
            make_message(Role::User, "Continue"),
        ];
        assert!(ContextBuilder::has_compaction_summary(&compacted));
    }

    #[test]
    fn test_should_retrieve_no_compaction() {
        let builder = ContextBuilder::new();
        let messages = vec![
            make_message(Role::User, "Hello"),
            make_message(Role::Assistant, "Hi there"),
        ];

        // Should not retrieve if no compaction
        assert!(!builder.should_retrieve("What did we discuss earlier?", &messages));
    }

    #[test]
    fn test_should_retrieve_with_compaction() {
        let builder = ContextBuilder::new();
        let messages = vec![
            make_message(
                Role::System,
                "[Compacted Context] Earlier we discussed auth...",
            ),
            make_message(Role::User, "Continue"),
        ];

        // Should retrieve with back-reference
        assert!(builder.should_retrieve("What did we discuss earlier?", &messages));
    }

    #[test]
    fn test_find_injection_point() {
        let builder = ContextBuilder::new();

        let messages = vec![
            make_message(Role::System, "System prompt"),
            make_message(Role::System, "[Compacted Context] Summary..."),
            make_message(Role::User, "Hello"),
        ];

        // Should insert after compaction summary
        assert_eq!(builder.find_injection_point(&messages), 2);
    }

    #[test]
    fn test_truncate_content() {
        let builder = ContextBuilder::new();

        let short = "Hello world";
        assert_eq!(builder.truncate_content(short, 100), short);

        let long = "This is a very long message that should be truncated at a word boundary";
        let truncated = builder.truncate_content(long, 30);
        assert!(truncated.ends_with("..."));
        assert!(truncated.len() <= 33); // 30 + "..."
    }

    #[test]
    fn test_config_defaults() {
        let config = ContextBuilderConfig::default();
        assert_eq!(config.injection_threshold, 0.75);
        assert_eq!(config.max_inject_items, 3);
        assert!(config.use_gating);
    }

    #[test]
    fn test_seal_config_defaults() {
        let config = ContextBuilderConfig::default();
        assert!(config.enable_seal_enhancement);
        assert!(config.use_resolved_query);
        assert!(config.inject_entity_context);
        assert!(config.quality_aware_threshold);
    }

    #[test]
    fn test_format_seal_injection_with_resolutions() {
        use crate::utils::entity_extraction::EntityType;
        use brainwires_seal::{ReferenceType, SalienceScore, UnresolvedReference};

        let builder = ContextBuilder::new();

        let resolutions = vec![ResolvedReference {
            reference: UnresolvedReference {
                text: "it".to_string(),
                ref_type: ReferenceType::SingularNeutral,
                start: 0,
                end: 2,
            },
            antecedent: "main.rs".to_string(),
            entity_type: EntityType::File,
            confidence: 0.85,
            salience: SalienceScore::default(),
        }];

        let results = vec![(
            MessageMetadata {
                message_id: "msg1".to_string(),
                conversation_id: "conv1".to_string(),
                role: "user".to_string(),
                content: "Let's look at main.rs".to_string(),
                created_at: 1000,
                token_count: Some(10),
                model_id: None,
                images: None,
                expires_at: None,
            },
            0.9f32,
        )];

        let injection = builder.format_seal_injection(&results, &resolutions);

        // Should contain entity context
        assert!(injection.contains("[Entity Context]"));
        assert!(injection.contains("\"it\" → main.rs"), "Got: {}", injection);
        assert!(injection.contains("85%"));

        // Should contain retrieved context
        assert!(injection.contains("[Retrieved Context]"));
        assert!(injection.contains("Let's look at main.rs"));
    }

    #[test]
    fn test_format_seal_injection_empty_resolutions() {
        let builder = ContextBuilder::new();

        let results = vec![(
            MessageMetadata {
                message_id: "msg1".to_string(),
                conversation_id: "conv1".to_string(),
                role: "assistant".to_string(),
                content: "Here's the code".to_string(),
                created_at: 1000,
                token_count: Some(5),
                model_id: None,
                images: None,
                expires_at: None,
            },
            0.8f32,
        )];

        let injection = builder.format_seal_injection(&results, &[]);

        // Should not contain entity context header
        assert!(!injection.contains("[Entity Context]"));

        // Should still contain retrieved context
        assert!(injection.contains("[Retrieved Context]"));
        assert!(injection.contains("Here's the code"));
    }

    #[test]
    fn test_format_seal_injection_low_confidence_filtered() {
        use crate::utils::entity_extraction::EntityType;
        use brainwires_seal::{ReferenceType, SalienceScore, UnresolvedReference};

        let builder = ContextBuilder::new();

        let resolutions = vec![
            ResolvedReference {
                reference: UnresolvedReference {
                    text: "it".to_string(),
                    ref_type: ReferenceType::SingularNeutral,
                    start: 0,
                    end: 2,
                },
                antecedent: "main.rs".to_string(),
                entity_type: EntityType::File,
                confidence: 0.3, // Below threshold
                salience: SalienceScore::default(),
            },
            ResolvedReference {
                reference: UnresolvedReference {
                    text: "the file".to_string(),
                    ref_type: ReferenceType::DefiniteNP {
                        entity_type: EntityType::File,
                    },
                    start: 10,
                    end: 18,
                },
                antecedent: "config.toml".to_string(),
                entity_type: EntityType::File,
                confidence: 0.7, // Above threshold
                salience: SalienceScore::default(),
            },
        ];

        let injection = builder.format_seal_injection(&[], &resolutions);

        // Low confidence resolution should be filtered
        assert!(!injection.contains("\"it\" → main.rs"));

        // High confidence resolution should be included
        assert!(injection.contains("\"the file\" → config.toml"));
    }

    #[test]
    fn test_format_seal_injection_disabled() {
        use crate::utils::entity_extraction::EntityType;
        use brainwires_seal::{ReferenceType, SalienceScore, UnresolvedReference};

        let config = ContextBuilderConfig {
            inject_entity_context: false,
            ..Default::default()
        };
        let builder = ContextBuilder::with_config(config);

        let resolutions = vec![ResolvedReference {
            reference: UnresolvedReference {
                text: "it".to_string(),
                ref_type: ReferenceType::SingularNeutral,
                start: 0,
                end: 2,
            },
            antecedent: "main.rs".to_string(),
            entity_type: EntityType::File,
            confidence: 0.9,
            salience: SalienceScore::default(),
        }];

        let results = vec![(
            MessageMetadata {
                message_id: "msg1".to_string(),
                conversation_id: "conv1".to_string(),
                role: "user".to_string(),
                content: "Test content".to_string(),
                created_at: 1000,
                token_count: Some(5),
                model_id: None,
                images: None,
                expires_at: None,
            },
            0.85f32,
        )];

        let injection = builder.format_seal_injection(&results, &resolutions);

        // Entity context should be disabled
        assert!(!injection.contains("[Entity Context]"));

        // Retrieved context should still work
        assert!(injection.contains("[Retrieved Context]"));
    }

    #[test]
    fn test_quality_aware_threshold_calculation() {
        // High quality (1.0) should keep threshold as-is
        // adjustment = 1.0 - (1.0 - 1.0) * 0.3 = 1.0
        // threshold = 0.75 * 1.0 = 0.75

        // Lower quality (0.5) should relax threshold
        // adjustment = 1.0 - (1.0 - 0.5) * 0.3 = 1.0 - 0.15 = 0.85
        // threshold = 0.75 * 0.85 = 0.6375

        let config = ContextBuilderConfig::default();
        let base_threshold = config.injection_threshold;

        // High quality
        let high_quality = 1.0f32;
        let high_adjustment = 1.0 - (1.0 - high_quality) * 0.3;
        let high_threshold = base_threshold * high_adjustment;
        assert!((high_threshold - 0.75).abs() < 0.01);

        // Medium quality
        let medium_quality = 0.5f32;
        let medium_adjustment = 1.0 - (1.0 - medium_quality) * 0.3;
        let medium_threshold = base_threshold * medium_adjustment;
        assert!((medium_threshold - 0.6375).abs() < 0.01);

        // Low quality (more context retrieved)
        let low_quality = 0.0f32;
        let low_adjustment = 1.0 - (1.0 - low_quality) * 0.3;
        let low_threshold = base_threshold * low_adjustment;
        assert!((low_threshold - 0.525).abs() < 0.01);
    }
}
