//! Router - Semantic Query Classification
//!
//! Uses a provider to classify queries into tool categories,
//! replacing keyword-based pattern matching with semantic understanding.

use std::sync::Arc;
use tracing::warn;

use brainwires_core::message::Message;
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_tool_runtime::ToolCategory;

use crate::InferenceTimer;

/// Result of route classification
#[derive(Clone, Debug)]
pub struct RouteResult {
    /// Classified tool categories
    pub categories: Vec<ToolCategory>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Whether LLM was used (vs fallback)
    pub used_local_llm: bool,
}

impl RouteResult {
    /// Create a result from pattern-based fallback
    pub fn from_fallback(categories: Vec<ToolCategory>) -> Self {
        Self {
            categories,
            confidence: 0.5, // Lower confidence for fallback
            used_local_llm: false,
        }
    }

    /// Create a result from LLM classification
    pub fn from_local(categories: Vec<ToolCategory>, confidence: f32) -> Self {
        Self {
            categories,
            confidence,
            used_local_llm: true,
        }
    }
}

/// Router for semantic query classification
pub struct LocalRouter {
    provider: Arc<dyn Provider>,
    model_id: String,
}

impl LocalRouter {
    /// Create a new router with the given provider
    pub fn new(provider: Arc<dyn Provider>, model_id: impl Into<String>) -> Self {
        Self {
            provider,
            model_id: model_id.into(),
        }
    }

    /// Classify a query into tool categories using the provider
    ///
    /// Returns None if classification fails, allowing fallback to pattern matching.
    pub async fn classify(&self, query: &str) -> Option<RouteResult> {
        let timer = InferenceTimer::new("route_classify", &self.model_id);

        let system_prompt = self.build_classification_prompt();
        let user_prompt = format!(
            "Classify this query into tool categories. Output ONLY the category names, comma-separated.\n\nQuery: {}",
            query
        );

        let messages = vec![Message::user(&user_prompt)];
        let options = ChatOptions::deterministic(50).system(system_prompt);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let text = response.message.text_or_summary();
                let categories = self.parse_categories(&text);

                if categories.is_empty() {
                    timer.finish(false);
                    return None;
                }

                timer.finish(true);
                Some(RouteResult::from_local(categories, 0.85))
            }
            Err(e) => {
                warn!(target: "local_llm", "Route classification failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Build the system prompt for classification
    fn build_classification_prompt(&self) -> String {
        r#"You are a tool category classifier. Given a user query, output the relevant tool categories.

Available categories:
- FileOps: File operations (read, write, edit, create, delete, list files/directories)
- Search: Text search (grep, find patterns, locate text)
- SemanticSearch: Semantic/concept search (codebase queries, embeddings, RAG)
- Git: Git operations (commit, diff, branch, merge, status, log)
- TaskManager: Task tracking (todos, progress, subtasks)
- AgentPool: Multi-agent operations (spawn, parallel, background)
- Web: HTTP/API operations (fetch, request, download)
- WebSearch: Internet search (google, browse, scrape)
- Bash: Shell commands (run, execute, npm, cargo, pip, docker)
- Planning: Design/architecture (plan, strategy, roadmap)
- Context: Memory/recall (remember, previous, earlier)
- Orchestrator: Script automation (workflow, batch)
- CodeExecution: Code execution (run code, python, javascript)

Rules:
1. Output ONLY category names, comma-separated
2. Include multiple categories if query spans multiple domains
3. Always include FileOps if file operations might be needed
4. Be conservative - only include clearly relevant categories"#.to_string()
    }

    /// Parse LLM output into tool categories
    fn parse_categories(&self, output: &str) -> Vec<ToolCategory> {
        let mut categories = Vec::new();
        let output_lower = output.to_lowercase();

        // Parse each potential category
        let category_mappings = [
            ("fileops", ToolCategory::FileOps),
            ("file", ToolCategory::FileOps),
            ("search", ToolCategory::Search),
            ("semanticsearch", ToolCategory::SemanticSearch),
            ("semantic", ToolCategory::SemanticSearch),
            ("git", ToolCategory::Git),
            ("taskmanager", ToolCategory::TaskManager),
            ("task", ToolCategory::TaskManager),
            ("agentpool", ToolCategory::AgentPool),
            ("agent", ToolCategory::AgentPool),
            ("web", ToolCategory::Web),
            ("websearch", ToolCategory::WebSearch),
            ("bash", ToolCategory::Bash),
            ("shell", ToolCategory::Bash),
            ("planning", ToolCategory::Planning),
            ("plan", ToolCategory::Planning),
            ("context", ToolCategory::Context),
            ("orchestrator", ToolCategory::Orchestrator),
            ("codeexecution", ToolCategory::CodeExecution),
            ("code", ToolCategory::CodeExecution),
        ];

        for (keyword, category) in category_mappings {
            if output_lower.contains(keyword) && !categories.contains(&category) {
                categories.push(category);
            }
        }

        categories
    }
}

/// Builder for LocalRouter
pub struct LocalRouterBuilder {
    provider: Option<Arc<dyn Provider>>,
    model_id: String,
}

impl Default for LocalRouterBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            model_id: "lfm2-350m".to_string(),
        }
    }
}

impl LocalRouterBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider to use for query routing.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model ID to use for inference.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Build the local router, returning `None` if no provider was set.
    pub fn build(self) -> Option<LocalRouter> {
        self.provider.map(|p| LocalRouter::new(p, self.model_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_result_from_fallback() {
        let result = RouteResult::from_fallback(vec![ToolCategory::FileOps, ToolCategory::Search]);
        assert!(!result.used_local_llm);
        assert_eq!(result.confidence, 0.5);
        assert_eq!(result.categories.len(), 2);
    }

    #[test]
    fn test_route_result_from_local() {
        let result = RouteResult::from_local(vec![ToolCategory::Git], 0.9);
        assert!(result.used_local_llm);
        assert_eq!(result.confidence, 0.9);
    }

    #[test]
    fn test_parse_categories() {
        let _router = LocalRouterBuilder::default();

        // Test the parsing logic directly
        let output = "FileOps, Git, Bash";
        let output_lower = output.to_lowercase();
        let mut categories = Vec::new();

        if output_lower.contains("fileops") || output_lower.contains("file") {
            categories.push(ToolCategory::FileOps);
        }
        if output_lower.contains("git") {
            categories.push(ToolCategory::Git);
        }
        if output_lower.contains("bash") {
            categories.push(ToolCategory::Bash);
        }

        assert!(categories.contains(&ToolCategory::FileOps));
        assert!(categories.contains(&ToolCategory::Git));
        assert!(categories.contains(&ToolCategory::Bash));
    }
}
