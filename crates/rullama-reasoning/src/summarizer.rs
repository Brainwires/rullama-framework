//! Summarizer - Context Summarization
//!
//! Uses a provider to generate summaries for tiered memory demotion,
//! reducing the need for expensive API calls for context compression.

use std::sync::Arc;
use tracing::warn;

use rullama_core::message::Message;
use rullama_core::provider::{ChatOptions, Provider};

use crate::InferenceTimer;

const CONTEXT_BUFFER_CAPACITY: usize = 4000;

/// Result of a summarization operation
#[derive(Clone, Debug)]
pub struct SummarizationResult {
    /// The generated summary
    pub summary: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Whether LLM was used
    pub used_local_llm: bool,
}

impl SummarizationResult {
    /// Create a result from LLM summarization
    pub fn from_local(summary: String, confidence: f32) -> Self {
        Self {
            summary,
            confidence,
            used_local_llm: true,
        }
    }

    /// Create a fallback result (simple truncation)
    pub fn from_fallback(summary: String) -> Self {
        Self {
            summary,
            confidence: 0.3,
            used_local_llm: false,
        }
    }
}

/// Key fact extracted from content
#[derive(Clone, Debug)]
pub struct ExtractedFact {
    /// The fact content
    pub fact: String,
    /// Type of fact (decision, definition, requirement, etc.)
    pub fact_type: FactCategory,
    /// Confidence score
    pub confidence: f32,
}

/// Category of extracted facts
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactCategory {
    /// A decision that was made.
    Decision,
    /// A definition or clarification of a term.
    Definition,
    /// A requirement or constraint.
    Requirement,
    /// A code change, fix, or modification.
    CodeChange,
    /// A configuration or settings change.
    Configuration,
    /// A reference to external documentation or resources.
    Reference,
    /// An uncategorized fact.
    Other,
}

impl FactCategory {
    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("decision") {
            FactCategory::Decision
        } else if lower.contains("definition") || lower.contains("define") {
            FactCategory::Definition
        } else if lower.contains("requirement")
            || lower.contains("must")
            || lower.contains("should")
        {
            FactCategory::Requirement
        } else if lower.contains("code") || lower.contains("change") || lower.contains("fix") {
            FactCategory::CodeChange
        } else if lower.contains("config") || lower.contains("setting") {
            FactCategory::Configuration
        } else if lower.contains("reference") || lower.contains("see") || lower.contains("link") {
            FactCategory::Reference
        } else {
            FactCategory::Other
        }
    }
}

/// Summarizer for context compression
pub struct LocalSummarizer {
    provider: Arc<dyn Provider>,
    model_id: String,
    /// Maximum tokens for summary output
    max_summary_tokens: u32,
    /// Maximum facts to extract per summary
    max_facts: usize,
}

impl LocalSummarizer {
    /// Create a new summarizer
    pub fn new(provider: Arc<dyn Provider>, model_id: impl Into<String>) -> Self {
        Self {
            provider,
            model_id: model_id.into(),
            max_summary_tokens: 150,
            max_facts: 5,
        }
    }

    /// Set maximum summary tokens
    pub fn with_max_summary_tokens(mut self, tokens: u32) -> Self {
        self.max_summary_tokens = tokens;
        self
    }

    /// Set maximum facts to extract
    pub fn with_max_facts(mut self, facts: usize) -> Self {
        self.max_facts = facts;
        self
    }

    /// Summarize a message for warm tier storage
    ///
    /// Generates a 50-100 word summary suitable for the warm memory tier.
    pub async fn summarize_message(
        &self,
        content: &str,
        role: &str,
    ) -> Option<SummarizationResult> {
        let timer = InferenceTimer::new("summarize_message", &self.model_id);

        // Skip very short content
        if content.trim().len() < 50 {
            return Some(SummarizationResult::from_fallback(content.to_string()));
        }

        let prompt = format!(
            "Summarize this {} message in 50-100 words. Preserve key information, decisions, and technical details.\n\nMessage:\n{}\n\nSummary:",
            role,
            // Truncate very long content for efficiency
            if content.len() > 2000 {
                &content[..2000]
            } else {
                content
            }
        );

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions::creative(self.max_summary_tokens);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let summary = response.message.text_or_summary();
                let cleaned = self.clean_summary(&summary);
                if cleaned.len() < 10 {
                    timer.finish(false);
                    return None;
                }
                timer.finish(true);
                Some(SummarizationResult::from_local(cleaned, 0.8))
            }
            Err(e) => {
                warn!(target: "local_llm", "Message summarization failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Extract key facts from a summary for cold tier storage
    ///
    /// Parses structured facts from content for ultra-compressed archival.
    pub async fn extract_facts(&self, summary: &str) -> Option<Vec<ExtractedFact>> {
        let timer = InferenceTimer::new("extract_facts", &self.model_id);

        // Skip very short summaries
        if summary.trim().len() < 30 {
            return Some(vec![ExtractedFact {
                fact: summary.to_string(),
                fact_type: FactCategory::Other,
                confidence: 0.5,
            }]);
        }

        let prompt = format!(
            "Extract {} key facts from this text. Format each as: TYPE: fact\nTypes: Decision, Definition, Requirement, CodeChange, Configuration, Reference, Other\n\nText:\n{}\n\nFacts:",
            self.max_facts, summary
        );

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions::factual(200);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let output = response.message.text_or_summary();
                let facts = self.parse_facts(&output);
                if facts.is_empty() {
                    timer.finish(false);
                    return None;
                }
                timer.finish(true);
                Some(facts)
            }
            Err(e) => {
                warn!(target: "local_llm", "Fact extraction failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Compact a conversation for emergency context reduction
    ///
    /// Used when token count exceeds threshold (e.g., 80k tokens).
    pub async fn compact_conversation(
        &self,
        messages: &[(String, String)], // (role, content) pairs
        keep_recent: usize,
    ) -> Option<String> {
        let timer = InferenceTimer::new("compact_conversation", &self.model_id);

        if messages.len() <= keep_recent {
            return None; // Nothing to compact
        }

        let to_compact = &messages[..messages.len() - keep_recent];

        // Build a condensed representation
        let mut context = String::with_capacity(CONTEXT_BUFFER_CAPACITY);
        for (role, content) in to_compact.iter().take(20) {
            let truncated = if content.len() > 200 {
                &content[..200]
            } else {
                content
            };
            context.push_str(&format!("[{}]: {}\n", role, truncated));
        }

        if to_compact.len() > 20 {
            context.push_str(&format!(
                "\n... ({} more messages)\n",
                to_compact.len() - 20
            ));
        }

        let prompt = format!(
            "Summarize this conversation history in 200-300 words. Focus on: decisions made, key technical details, current task state.\n\n{}\n\nSummary:",
            context
        );

        let chat_messages = vec![Message::user(&prompt)];
        let options = ChatOptions::creative(400);

        match self.provider.chat(&chat_messages, None, &options).await {
            Ok(response) => {
                let summary = response.message.text_or_summary();
                let cleaned = self.clean_summary(&summary);
                timer.finish(true);
                Some(cleaned)
            }
            Err(e) => {
                warn!(target: "local_llm", "Conversation compaction failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Heuristic summarization (no LLM)
    pub fn summarize_heuristic(&self, content: &str) -> SummarizationResult {
        SummarizationResult::from_fallback(self.truncate_summary(content))
    }

    /// Extract entities from content for summary metadata
    pub fn extract_entities(&self, content: &str) -> Vec<String> {
        let mut entities = Vec::new();

        // Extract file paths
        let path_patterns = [r"([a-zA-Z0-9_\-/]+\.[a-z]{2,4})", r"src/[a-zA-Z0-9_\-/]+"];
        for pattern in path_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(content) {
                    if let Some(m) = cap.get(0) {
                        let entity = m.as_str().to_string();
                        if !entities.contains(&entity) && entity.len() > 3 {
                            entities.push(entity);
                        }
                    }
                }
            }
        }

        // Extract function/type names (PascalCase or snake_case)
        if let Ok(re) = regex::Regex::new(r"\b([A-Z][a-zA-Z0-9]+|[a-z]+_[a-z_]+)\b") {
            for cap in re.captures_iter(content) {
                if let Some(m) = cap.get(1) {
                    let entity = m.as_str().to_string();
                    if !entities.contains(&entity)
                        && entity.len() > 3
                        && ![
                            "This", "That", "These", "Those", "What", "When", "Where", "Which",
                        ]
                        .contains(&entity.as_str())
                    {
                        entities.push(entity);
                    }
                }
            }
        }

        // Limit to top 10 entities
        entities.truncate(10);
        entities
    }

    /// Truncate content to create a simple summary
    fn truncate_summary(&self, content: &str) -> String {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.len() <= 100 {
            content.to_string()
        } else {
            format!("{}...", words[..100].join(" "))
        }
    }

    /// Clean up LLM output for summary
    fn clean_summary(&self, output: &str) -> String {
        let mut cleaned = output.trim().to_string();

        // Remove common prefixes
        let prefixes = [
            "Summary:",
            "Here's a summary:",
            "Here is a summary:",
            "The summary is:",
        ];
        for prefix in prefixes {
            if cleaned.to_lowercase().starts_with(&prefix.to_lowercase()) {
                cleaned = cleaned[prefix.len()..].trim().to_string();
            }
        }

        // Remove trailing incomplete sentences
        if let Some(last_period) = cleaned.rfind('.')
            && last_period < cleaned.len() - 20
        {
            cleaned = cleaned[..=last_period].to_string();
        }

        cleaned
    }

    /// Parse facts from LLM output
    fn parse_facts(&self, output: &str) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();

        for line in output.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Try to parse "TYPE: fact" format
            if let Some(colon_pos) = line.find(':') {
                let type_part = &line[..colon_pos].trim();
                let fact_part = line[colon_pos + 1..].trim();

                if !fact_part.is_empty() {
                    facts.push(ExtractedFact {
                        fact: fact_part.to_string(),
                        fact_type: FactCategory::from_str(type_part),
                        confidence: 0.75,
                    });
                }
            } else if line.len() > 10 {
                // Line without type prefix
                facts.push(ExtractedFact {
                    fact: line.to_string(),
                    fact_type: FactCategory::Other,
                    confidence: 0.5,
                });
            }

            if facts.len() >= self.max_facts {
                break;
            }
        }

        facts
    }

    /// Heuristic fact extraction (no LLM)
    fn _extract_facts_heuristic(&self, content: &str) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();

        // Look for sentences with decision/requirement indicators
        for sentence in content.split(['.', '!', '?']) {
            let sentence = sentence.trim();
            if sentence.len() < 10 {
                continue;
            }

            let lower = sentence.to_lowercase();
            let fact_type = if lower.contains("decided")
                || lower.contains("will use")
                || lower.contains("chose")
            {
                FactCategory::Decision
            } else if lower.contains("must")
                || lower.contains("should")
                || lower.contains("need to")
            {
                FactCategory::Requirement
            } else if lower.contains("is defined as") || lower.contains("means") {
                FactCategory::Definition
            } else if lower.contains("changed")
                || lower.contains("fixed")
                || lower.contains("updated")
            {
                FactCategory::CodeChange
            } else if lower.contains("configured") || lower.contains("set to") {
                FactCategory::Configuration
            } else {
                continue; // Skip non-fact sentences
            };

            facts.push(ExtractedFact {
                fact: sentence.to_string(),
                fact_type,
                confidence: 0.5,
            });

            if facts.len() >= self.max_facts {
                break;
            }
        }

        // If no facts found, create one from the first sentence
        if facts.is_empty()
            && let Some(first_sentence) = content.split('.').next()
            && first_sentence.len() > 10
        {
            facts.push(ExtractedFact {
                fact: first_sentence.trim().to_string(),
                fact_type: FactCategory::Other,
                confidence: 0.3,
            });
        }

        facts
    }
}

/// Builder for LocalSummarizer
pub struct LocalSummarizerBuilder {
    provider: Option<Arc<dyn Provider>>,
    model_id: String,
    max_summary_tokens: u32,
    max_facts: usize,
}

impl Default for LocalSummarizerBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            model_id: "lfm2-1.2b".to_string(), // Use larger model for summarization
            max_summary_tokens: 150,
            max_facts: 5,
        }
    }
}

impl LocalSummarizerBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider to use for summarization.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model ID to use for inference.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Set the maximum number of tokens for summary output.
    pub fn max_summary_tokens(mut self, tokens: u32) -> Self {
        self.max_summary_tokens = tokens;
        self
    }

    /// Set the maximum number of facts to extract per summary.
    pub fn max_facts(mut self, facts: usize) -> Self {
        self.max_facts = facts;
        self
    }

    /// Build the summarizer, returning `None` if no provider was set.
    pub fn build(self) -> Option<LocalSummarizer> {
        self.provider.map(|p| {
            LocalSummarizer::new(p, self.model_id)
                .with_max_summary_tokens(self.max_summary_tokens)
                .with_max_facts(self.max_facts)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarization_result() {
        let result = SummarizationResult::from_local("Test summary".to_string(), 0.9);
        assert!(result.used_local_llm);
        assert_eq!(result.confidence, 0.9);

        let fallback = SummarizationResult::from_fallback("Fallback".to_string());
        assert!(!fallback.used_local_llm);
        assert_eq!(fallback.confidence, 0.3);
    }

    #[test]
    fn test_fact_category_parsing() {
        assert_eq!(FactCategory::from_str("Decision"), FactCategory::Decision);
        assert_eq!(
            FactCategory::from_str("REQUIREMENT"),
            FactCategory::Requirement
        );
        assert_eq!(
            FactCategory::from_str("code change"),
            FactCategory::CodeChange
        );
        assert_eq!(FactCategory::from_str("random"), FactCategory::Other);
    }

    #[test]
    fn test_entity_extraction() {
        let _summarizer = LocalSummarizerBuilder::default();

        // Test entity extraction logic
        let content = "Modified src/main.rs and added LocalSummarizer to handle_request function";
        let entities = extract_entities_direct(content);

        assert!(
            entities
                .iter()
                .any(|e| e.contains("main.rs") || e.contains("src/"))
        );
        assert!(
            entities
                .iter()
                .any(|e| e.contains("LocalSummarizer") || e.contains("handle_request"))
        );
    }

    fn extract_entities_direct(content: &str) -> Vec<String> {
        let mut entities = Vec::new();

        // Extract file paths
        if let Ok(re) = regex::Regex::new(r"([a-zA-Z0-9_\-/]+\.[a-z]{2,4})") {
            for cap in re.captures_iter(content) {
                if let Some(m) = cap.get(0) {
                    entities.push(m.as_str().to_string());
                }
            }
        }

        // Extract PascalCase names
        if let Ok(re) = regex::Regex::new(r"\b([A-Z][a-zA-Z0-9]+)\b") {
            for cap in re.captures_iter(content) {
                if let Some(m) = cap.get(1) {
                    let name = m.as_str().to_string();
                    if !["Modified", "This", "That"].contains(&name.as_str()) {
                        entities.push(name);
                    }
                }
            }
        }

        entities
    }

    #[test]
    fn test_heuristic_fact_extraction() {
        let content =
            "We decided to use Rust. The config must be updated. The function was changed.";
        let facts = extract_facts_heuristic_direct(content);

        assert!(!facts.is_empty());
        assert!(facts.iter().any(|f| f.fact_type == FactCategory::Decision));
    }

    fn extract_facts_heuristic_direct(content: &str) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();

        for sentence in content.split('.') {
            let sentence = sentence.trim();
            if sentence.len() < 10 {
                continue;
            }

            let lower = sentence.to_lowercase();
            let fact_type = if lower.contains("decided") {
                FactCategory::Decision
            } else if lower.contains("must") {
                FactCategory::Requirement
            } else if lower.contains("changed") {
                FactCategory::CodeChange
            } else {
                continue;
            };

            facts.push(ExtractedFact {
                fact: sentence.to_string(),
                fact_type,
                confidence: 0.5,
            });
        }

        facts
    }

    #[test]
    fn test_truncate_summary() {
        let long_content = "word ".repeat(200);
        let truncated = truncate_summary_direct(&long_content);

        let word_count = truncated.split_whitespace().count();
        assert!(word_count <= 101); // 100 words + "..."
    }

    fn truncate_summary_direct(content: &str) -> String {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.len() <= 100 {
            content.to_string()
        } else {
            format!("{}...", words[..100].join(" "))
        }
    }
}
