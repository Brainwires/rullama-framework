//! Personal Fact Matcher
//!
//! Matches relevant personal facts to current context for injection into prompts.

use super::fact::{PersonalFact, PersonalFactCategory};
use std::collections::HashSet;

/// Matcher for selecting relevant personal facts for context injection
pub struct PersonalFactMatcher {
    /// Minimum confidence threshold for matching
    min_confidence: f32,
    /// Maximum number of facts to include
    max_facts: usize,
    /// Whether to include context facts (which are more transient)
    include_context: bool,
}

impl Default for PersonalFactMatcher {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_facts: 15,
            include_context: true,
        }
    }
}

impl PersonalFactMatcher {
    /// Create a new matcher with custom settings
    pub fn new(min_confidence: f32, max_facts: usize, include_context: bool) -> Self {
        Self {
            min_confidence,
            max_facts,
            include_context,
        }
    }

    /// Get relevant facts for context injection
    ///
    /// Returns facts sorted by relevance (category priority + confidence)
    pub fn get_relevant_facts<'a>(
        &self,
        facts: impl Iterator<Item = &'a PersonalFact>,
        context: Option<&str>,
    ) -> Vec<&'a PersonalFact> {
        let mut relevant: Vec<&PersonalFact> = facts
            .filter(|f| {
                // Filter by confidence
                if f.decayed_confidence() < self.min_confidence {
                    return false;
                }

                // Optionally filter out context facts
                if !self.include_context && f.category == PersonalFactCategory::Context {
                    return false;
                }

                true
            })
            .collect();

        // Score and sort by relevance
        relevant.sort_by(|a, b| {
            let score_a = self.relevance_score(a, context);
            let score_b = self.relevance_score(b, context);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to max facts
        relevant.truncate(self.max_facts);

        relevant
    }

    /// Calculate relevance score for a fact
    fn relevance_score(&self, fact: &PersonalFact, context: Option<&str>) -> f32 {
        let mut score = fact.decayed_confidence();

        // Boost by category priority
        score *= self.category_priority(&fact.category);

        // Boost by reinforcements (but capped)
        let reinforcement_boost = (fact.reinforcements.min(10) as f32) * 0.02;
        score += reinforcement_boost;

        // Boost if matches current context
        if let Some(ctx) = context
            && self.matches_context(fact, ctx)
        {
            score *= 1.3;
        }

        score
    }

    /// Get priority multiplier for category
    fn category_priority(&self, category: &PersonalFactCategory) -> f32 {
        match category {
            // Identity facts are most important for personalization
            PersonalFactCategory::Identity => 1.2,
            // Preferences directly affect assistant behavior
            PersonalFactCategory::Preference => 1.15,
            // Capabilities help adjust explanations
            PersonalFactCategory::Capability => 1.1,
            // Constraints are critical to respect
            PersonalFactCategory::Constraint => 1.2,
            // Context is valuable but transient
            PersonalFactCategory::Context => 1.0,
            // Relationships are supplementary
            PersonalFactCategory::Relationship => 0.9,
            // Ambiguity type preferences affect question quality
            PersonalFactCategory::AmbiguityTypePreference => 1.1,
        }
    }

    /// Check if a fact matches the current context
    fn matches_context(&self, fact: &PersonalFact, context: &str) -> bool {
        let context_lower = context.to_lowercase();

        // Check key matches
        if context_lower.contains(&fact.key.to_lowercase()) {
            return true;
        }

        // Check value matches
        if fact
            .value
            .to_lowercase()
            .split_whitespace()
            .any(|word| word.len() > 3 && context_lower.contains(word))
        {
            return true;
        }

        // Check fact context matches
        if let Some(ref fact_ctx) = fact.context
            && context_lower.contains(&fact_ctx.to_lowercase())
        {
            return true;
        }

        false
    }

    /// Format facts for context injection
    pub fn format_for_context(&self, facts: &[&PersonalFact]) -> String {
        if facts.is_empty() {
            return String::new();
        }

        let mut lines = Vec::new();
        lines.push("[User Profile]".to_string());

        // Group by category for cleaner output
        let mut by_category: std::collections::HashMap<PersonalFactCategory, Vec<&PersonalFact>> =
            std::collections::HashMap::new();

        for fact in facts {
            by_category.entry(fact.category).or_default().push(fact);
        }

        // Output in category order
        let category_order = [
            PersonalFactCategory::Identity,
            PersonalFactCategory::Preference,
            PersonalFactCategory::Capability,
            PersonalFactCategory::Context,
            PersonalFactCategory::Constraint,
            PersonalFactCategory::Relationship,
        ];

        for category in &category_order {
            if let Some(cat_facts) = by_category.get(category) {
                for fact in cat_facts {
                    lines.push(format!("- {}", fact.to_context_string()));
                }
            }
        }

        lines.join("\n")
    }

    /// Get a summary of the user's profile
    pub fn format_profile_summary(&self, facts: &[&PersonalFact]) -> String {
        if facts.is_empty() {
            return "No profile information available.".to_string();
        }

        let mut sections = Vec::new();

        // Find name
        if let Some(name_fact) = facts
            .iter()
            .find(|f| f.key == "name" || f.key == "preferred_name")
        {
            sections.push(format!("Name: {}", name_fact.value));
        }

        // Find role/organization
        let role = facts
            .iter()
            .find(|f| f.key == "role")
            .map(|f| f.value.as_str());
        let org = facts
            .iter()
            .find(|f| f.key == "organization")
            .map(|f| f.value.as_str());
        match (role, org) {
            (Some(r), Some(o)) => sections.push(format!("Role: {} at {}", r, o)),
            (Some(r), None) => sections.push(format!("Role: {}", r)),
            (None, Some(o)) => sections.push(format!("Organization: {}", o)),
            _ => {}
        }

        // Find current project
        if let Some(project) = facts.iter().find(|f| f.key == "current_project") {
            sections.push(format!("Current Project: {}", project.value));
        }

        // Count preferences and capabilities
        let pref_count = facts
            .iter()
            .filter(|f| f.category == PersonalFactCategory::Preference)
            .count();
        let cap_count = facts
            .iter()
            .filter(|f| f.category == PersonalFactCategory::Capability)
            .count();

        if pref_count > 0 {
            sections.push(format!("Preferences: {} recorded", pref_count));
        }
        if cap_count > 0 {
            sections.push(format!("Skills/Capabilities: {} recorded", cap_count));
        }

        if sections.is_empty() {
            return "Profile has some facts but no key information.".to_string();
        }

        sections.join("\n")
    }
}

/// Extract keywords from a message for context matching
pub fn extract_keywords(text: &str) -> HashSet<String> {
    let stopwords: HashSet<&str> = [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "of",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
    ]
    .iter()
    .cloned()
    .collect();

    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|word| word.len() > 2 && !stopwords.contains(word))
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::bks_pks::personal::fact::PersonalFactSource;

    fn create_test_fact(category: PersonalFactCategory, key: &str, value: &str) -> PersonalFact {
        PersonalFact::new(
            category,
            key.to_string(),
            value.to_string(),
            None,
            PersonalFactSource::ExplicitStatement,
            false,
        )
    }

    #[test]
    fn test_matcher_creation() {
        let matcher = PersonalFactMatcher::default();
        assert_eq!(matcher.min_confidence, 0.5);
        assert_eq!(matcher.max_facts, 15);
    }

    #[test]
    fn test_get_relevant_facts() {
        let matcher = PersonalFactMatcher::default();

        let facts = [
            create_test_fact(PersonalFactCategory::Identity, "name", "John"),
            create_test_fact(PersonalFactCategory::Preference, "language", "Rust"),
            create_test_fact(PersonalFactCategory::Context, "project", "brainwires"),
        ];

        let relevant: Vec<_> = matcher
            .get_relevant_facts(facts.iter(), None)
            .into_iter()
            .collect();

        assert_eq!(relevant.len(), 3);
    }

    #[test]
    fn test_context_matching() {
        let matcher = PersonalFactMatcher::default();

        let rust_fact = create_test_fact(PersonalFactCategory::Capability, "language", "Rust");
        let python_fact = create_test_fact(PersonalFactCategory::Capability, "language", "Python");

        let facts = [rust_fact.clone(), python_fact.clone()];

        let relevant: Vec<_> = matcher
            .get_relevant_facts(facts.iter(), Some("working with Rust"))
            .into_iter()
            .collect();

        // Rust fact should come first due to context match
        assert!(!relevant.is_empty());
        assert_eq!(relevant[0].value, "Rust");
    }

    #[test]
    fn test_format_for_context() {
        let matcher = PersonalFactMatcher::default();

        let facts = [
            create_test_fact(PersonalFactCategory::Identity, "name", "John"),
            create_test_fact(PersonalFactCategory::Preference, "editor", "VSCode"),
        ];

        let refs: Vec<&PersonalFact> = facts.iter().collect();
        let formatted = matcher.format_for_context(&refs);

        assert!(formatted.contains("[User Profile]"));
        assert!(formatted.contains("name: John"));
        assert!(formatted.contains("editor: VSCode"));
    }

    #[test]
    fn test_extract_keywords() {
        let keywords = extract_keywords("I am working on a Rust project called brainwires");

        assert!(keywords.contains("working"));
        assert!(keywords.contains("rust"));
        assert!(keywords.contains("project"));
        assert!(keywords.contains("brainwires"));
        // Stopwords should be filtered
        assert!(!keywords.contains("i"));
        assert!(!keywords.contains("am"));
        assert!(!keywords.contains("on"));
        assert!(!keywords.contains("a"));
    }
}
