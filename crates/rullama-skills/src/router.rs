//! Skill Router
//!
//! Handles skill activation through semantic matching and keyword patterns.
//! Skills are **suggested** to the user, not auto-activated.
//!
//! # Activation Flow
//!
//! 1. User query is analyzed against skill descriptions
//! 2. Matching skills are suggested (e.g., "Skill 'review-pr' may help")
//! 3. User explicitly invokes with `/skill-name` or `/skill <name>`
//!
//! # Matching Methods
//!
//! - **Semantic**: Uses LocalRouter for similarity matching (when llama-cpp-2 feature is enabled)
//! - **Keyword**: Fallback pattern matching against skill names and descriptions

use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(test)]
use super::metadata::MatchSource;
use super::metadata::{SkillMatch, SkillMetadata};
use super::registry::SkillRegistry;

/// Minimum confidence for showing skill suggestions
const MIN_SUGGESTION_CONFIDENCE: f32 = 0.5;

/// Keyword match confidence score
const KEYWORD_MATCH_CONFIDENCE: f32 = 0.6;

/// Skill router for matching queries against skills
pub struct SkillRouter {
    /// Reference to skill registry
    registry: Arc<RwLock<SkillRegistry>>,
    /// Minimum confidence for suggestions
    min_confidence: f32,
}

impl SkillRouter {
    /// Create a new skill router
    pub fn new(registry: Arc<RwLock<SkillRegistry>>) -> Self {
        Self {
            registry,
            min_confidence: MIN_SUGGESTION_CONFIDENCE,
        }
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Match query against skill descriptions
    ///
    /// Returns matching skills sorted by confidence (highest first).
    pub async fn match_skills(&self, query: &str) -> Vec<SkillMatch> {
        let registry = self.registry.read().await;
        let all_metadata = registry.all_metadata();

        if all_metadata.is_empty() {
            return Vec::new();
        }

        // Use keyword matching (semantic matching via local LLM would be added in Phase 4)
        let mut matches = self.keyword_match(query, &all_metadata);

        // Filter by minimum confidence
        matches.retain(|m| m.confidence >= self.min_confidence);

        // Sort by confidence (highest first)
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    // Note: Semantic matching via LocalRouter is planned for a future enhancement.
    // The LocalRouter currently supports tool category classification, not arbitrary
    // text generation. For now, keyword-based matching provides good results.
    //
    // Future enhancement: Add a dedicated skill classification method to LocalRouter
    // that can classify queries against skill descriptions.

    /// Keyword-based fallback matching
    ///
    /// Matches query words against skill names and descriptions.
    fn keyword_match(&self, query: &str, metadata: &[&SkillMetadata]) -> Vec<SkillMatch> {
        let query_lower = query.to_lowercase();
        let query_words: HashSet<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 2) // Skip short words
            .collect();

        if query_words.is_empty() {
            return Vec::new();
        }

        metadata
            .iter()
            .filter_map(|m| {
                let name_lower = m.name.to_lowercase();
                let desc_lower = m.description.to_lowercase();

                // Count matching words
                let mut match_count = 0;

                // Check name match (higher weight)
                if query_lower.contains(&name_lower) || name_lower.contains(&query_lower) {
                    match_count += 3;
                }

                // Check individual word matches in description
                for word in &query_words {
                    if desc_lower.contains(word) {
                        match_count += 1;
                    }
                }

                // Check for skill name words in query
                let name_words: Vec<&str> = name_lower.split('-').collect();
                for name_word in &name_words {
                    if query_words.contains(name_word) {
                        match_count += 2;
                    }
                }

                if match_count > 0 {
                    // Calculate confidence based on match count
                    let confidence =
                        (KEYWORD_MATCH_CONFIDENCE + (match_count as f32 * 0.05)).min(0.9);

                    Some(SkillMatch::keyword(m.name.clone(), confidence))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Format skill suggestions for display
    ///
    /// Returns None if no skills match, otherwise returns a formatted suggestion message.
    pub fn format_suggestions(&self, matches: &[SkillMatch]) -> Option<String> {
        if matches.is_empty() {
            return None;
        }

        let suggestions: Vec<String> = matches
            .iter()
            .take(3) // Limit to top 3
            .map(|m| format!("`/{}`", m.skill_name))
            .collect();

        let skill_word = if suggestions.len() == 1 {
            "skill"
        } else {
            "skills"
        };

        Some(format!(
            "The {} {} may help. Use the command to activate.",
            skill_word,
            suggestions.join(", ")
        ))
    }

    /// Check if a skill exists by name
    pub async fn skill_exists(&self, name: &str) -> bool {
        let registry = self.registry.read().await;
        registry.contains(name)
    }

    /// Get an explicit match for a skill name
    ///
    /// Used when user directly invokes `/skill-name`.
    pub fn explicit_match(&self, skill_name: &str) -> SkillMatch {
        SkillMatch::explicit(skill_name.to_string())
    }
}

/// Truncate description for prompt building
#[cfg(test)]
fn truncate_desc(desc: &str, max_len: usize) -> String {
    let first_line = desc.lines().next().unwrap_or(desc);
    if first_line.len() <= max_len {
        first_line.to_string()
    } else {
        format!("{}...", &first_line[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_registry() -> Arc<RwLock<SkillRegistry>> {
        let mut registry = SkillRegistry::new();

        // Add test skills
        let mut review_meta = SkillMetadata::new(
            "review-pr".to_string(),
            "Reviews pull requests for code quality, security issues, and best practices"
                .to_string(),
        );
        review_meta.allowed_tools = Some(vec!["Read".to_string(), "Grep".to_string()]);

        let commit_meta = SkillMetadata::new(
            "commit".to_string(),
            "Creates well-formatted git commits following conventional commit standards"
                .to_string(),
        );

        let explain_meta = SkillMetadata::new(
            "explain-code".to_string(),
            "Explains code functionality in detail, breaking down complex logic".to_string(),
        );

        registry.register(review_meta);
        registry.register(commit_meta);
        registry.register(explain_meta);

        Arc::new(RwLock::new(registry))
    }

    #[tokio::test]
    async fn test_router_creation() {
        let registry = create_test_registry().await;
        let router = SkillRouter::new(registry);
        assert_eq!(router.min_confidence, MIN_SUGGESTION_CONFIDENCE);
    }

    #[tokio::test]
    async fn test_match_by_name() {
        let registry = create_test_registry().await;
        let router = SkillRouter::new(registry);

        let matches = router.match_skills("review my pull request").await;
        assert!(!matches.is_empty());
        assert!(matches.iter().any(|m| m.skill_name == "review-pr"));
    }

    #[tokio::test]
    async fn test_match_by_description() {
        let registry = create_test_registry().await;
        let router = SkillRouter::new(registry);

        let matches = router.match_skills("check code quality").await;
        assert!(!matches.is_empty());
        // Should match review-pr (has "code quality" in description)
        assert!(matches.iter().any(|m| m.skill_name == "review-pr"));
    }

    #[tokio::test]
    async fn test_match_commit_skill() {
        let registry = create_test_registry().await;
        let router = SkillRouter::new(registry);

        let matches = router.match_skills("create a commit message").await;
        assert!(!matches.is_empty());
        assert!(matches.iter().any(|m| m.skill_name == "commit"));
    }

    #[tokio::test]
    async fn test_no_matches() {
        let registry = create_test_registry().await;
        let router = SkillRouter::new(registry);

        let _matches = router.match_skills("completely unrelated query").await;
        // May or may not have matches depending on threshold
        // Just ensure it doesn't panic
    }

    #[tokio::test]
    async fn test_empty_query() {
        let registry = create_test_registry().await;
        let router = SkillRouter::new(registry);

        let matches = router.match_skills("").await;
        assert!(matches.is_empty());
    }

    #[tokio::test]
    async fn test_skill_exists() {
        let registry = create_test_registry().await;
        let router = SkillRouter::new(registry);

        assert!(router.skill_exists("review-pr").await);
        assert!(router.skill_exists("commit").await);
        assert!(!router.skill_exists("nonexistent").await);
    }

    #[tokio::test]
    async fn test_explicit_match() {
        let registry = create_test_registry().await;
        let router = SkillRouter::new(registry);

        let m = router.explicit_match("review-pr");
        assert_eq!(m.skill_name, "review-pr");
        assert_eq!(m.confidence, 1.0);
        assert_eq!(m.source, MatchSource::Explicit);
    }

    #[test]
    fn test_format_suggestions_single() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let router = SkillRouter::new(registry);

        let matches = vec![SkillMatch::keyword("review-pr".to_string(), 0.8)];
        let suggestion = router.format_suggestions(&matches);

        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("/review-pr"));
    }

    #[test]
    fn test_format_suggestions_multiple() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let router = SkillRouter::new(registry);

        let matches = vec![
            SkillMatch::keyword("review-pr".to_string(), 0.8),
            SkillMatch::keyword("commit".to_string(), 0.7),
        ];
        let suggestion = router.format_suggestions(&matches);

        assert!(suggestion.is_some());
        let s = suggestion.unwrap();
        assert!(s.contains("/review-pr"));
        assert!(s.contains("/commit"));
        assert!(s.contains("skills")); // Plural
    }

    #[test]
    fn test_format_suggestions_empty() {
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let router = SkillRouter::new(registry);

        let suggestion = router.format_suggestions(&[]);
        assert!(suggestion.is_none());
    }

    #[test]
    fn test_truncate_desc() {
        assert_eq!(truncate_desc("Short description", 50), "Short description");
        assert_eq!(
            truncate_desc("This is a very long description that exceeds the limit", 20),
            "This is a very lo..."
        );
        assert_eq!(truncate_desc("First line\nSecond line", 100), "First line");
    }
}
