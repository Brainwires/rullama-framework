//! Personal Fact Collector
//!
//! Detects implicit personal facts from conversation patterns.
//! Recognizes phrases like "I prefer...", "I'm working on...", "My name is...", etc.

use super::fact::{PersonalFact, PersonalFactCategory, PersonalFactSource};
use regex::Regex;

/// Collector for detecting personal facts from conversation
pub struct PersonalFactCollector {
    /// Patterns for detecting identity facts
    identity_patterns: Vec<PatternRule>,
    /// Patterns for detecting preference facts
    preference_patterns: Vec<PatternRule>,
    /// Patterns for detecting capability facts
    capability_patterns: Vec<PatternRule>,
    /// Patterns for detecting context facts
    context_patterns: Vec<PatternRule>,
    /// Patterns for detecting constraint facts
    constraint_patterns: Vec<PatternRule>,
    /// Minimum confidence for inferred facts
    min_confidence: f32,
    /// Whether implicit detection is enabled
    enabled: bool,
}

/// A pattern rule for detecting facts
struct PatternRule {
    /// Compiled regex pattern
    pattern: Regex,
    /// Key to use for the detected fact
    key_template: String,
    /// Category for detected facts
    category: PersonalFactCategory,
    /// Confidence score for matches
    confidence: f32,
    /// Group index for the value (1-based)
    value_group: usize,
    /// Optional group index for additional context
    context_group: Option<usize>,
}

impl Default for PersonalFactCollector {
    fn default() -> Self {
        Self::new(0.7, true)
    }
}

impl PersonalFactCollector {
    /// Create a new collector with default patterns
    pub fn new(min_confidence: f32, enabled: bool) -> Self {
        let mut collector = Self {
            identity_patterns: Vec::new(),
            preference_patterns: Vec::new(),
            capability_patterns: Vec::new(),
            context_patterns: Vec::new(),
            constraint_patterns: Vec::new(),
            min_confidence,
            enabled,
        };

        collector.init_patterns();
        collector
    }

    /// Initialize detection patterns
    fn init_patterns(&mut self) {
        // Identity patterns
        self.identity_patterns = vec![
            PatternRule::new(
                r"(?i)my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                "name",
                PersonalFactCategory::Identity,
                0.9,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)call me\s+([A-Z][a-z]+)",
                "preferred_name",
                PersonalFactCategory::Identity,
                0.85,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i(?:'m| am) (?:a |an )?([a-z]+(?:\s+[a-z]+)*)\s+(?:at|for|with)\s+([A-Za-z0-9]+)",
                "role",
                PersonalFactCategory::Identity,
                0.8,
                1,
                Some(2),
            ),
            PatternRule::new(
                r"(?i)i work (?:at|for|with)\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)",
                "organization",
                PersonalFactCategory::Identity,
                0.8,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i(?:'m| am) on the\s+([a-z]+(?:\s+[a-z]+)*)\s+team",
                "team",
                PersonalFactCategory::Identity,
                0.8,
                1,
                None,
            ),
        ];

        // Preference patterns
        self.preference_patterns = vec![
            PatternRule::new(
                r"(?i)i prefer\s+(.+?)(?:\s+over|\s+to|\s*[,.]|$)",
                "preference",
                PersonalFactCategory::Preference,
                0.85,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i (?:really |always |usually )?like\s+(?:using |working with )?(.+?)(?:\s+for|\s*[,.]|$)",
                "liked_tool",
                PersonalFactCategory::Preference,
                0.7,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i(?:'d| would) rather\s+(.+?)(?:\s+than|\s*[,.]|$)",
                "preference",
                PersonalFactCategory::Preference,
                0.75,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)my favorite\s+([a-z]+)\s+is\s+([A-Za-z0-9]+)",
                "favorite_{1}",
                PersonalFactCategory::Preference,
                0.8,
                2,
                None,
            ),
            PatternRule::new(
                r"(?i)i use\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?)\s+(?:as my |for )([a-z]+)",
                "{2}_tool",
                PersonalFactCategory::Preference,
                0.75,
                1,
                None,
            ),
            // Decision/approach patterns from code conversations
            PatternRule::new(
                r"(?i)let's go with\s+(.+?)(?:\s*[,.]|$)",
                "decision",
                PersonalFactCategory::Preference,
                0.8,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)we should use\s+(.+?)(?:\s+for|\s*[,.]|$)",
                "approach",
                PersonalFactCategory::Preference,
                0.75,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)(?:decided|going) (?:to |with )(.+?)(?:\s+because|\s*[,.]|$)",
                "decision",
                PersonalFactCategory::Preference,
                0.8,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)always use\s+(.+?)\s+when",
                "rule",
                PersonalFactCategory::Preference,
                0.85,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)the approach is\s+(.+?)(?:\s*[,.]|$)",
                "approach",
                PersonalFactCategory::Preference,
                0.7,
                1,
                None,
            ),
        ];

        // Capability patterns
        self.capability_patterns = vec![
            PatternRule::new(
                r"(?i)i(?:'m| am) (?:fluent|proficient|experienced) (?:in|with)\s+([A-Za-z0-9#+]+)",
                "proficient_in",
                PersonalFactCategory::Capability,
                0.8,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i know\s+([A-Za-z0-9#+]+)(?:\s+(?:well|pretty well))?",
                "knows",
                PersonalFactCategory::Capability,
                0.7,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i(?:'ve| have) (?:been )?(?:using|working with)\s+([A-Za-z0-9#+]+)\s+for\s+(\d+)\s+years?",
                "experience_{1}",
                PersonalFactCategory::Capability,
                0.85,
                1,
                Some(2),
            ),
            PatternRule::new(
                r"(?i)i(?:'m| am) (?:a |an )?expert (?:in|at|with)\s+([A-Za-z0-9#+]+)",
                "expert_in",
                PersonalFactCategory::Capability,
                0.85,
                1,
                None,
            ),
        ];

        // Context patterns
        self.context_patterns = vec![
            PatternRule::new(
                r"(?i)i(?:'m| am) (?:currently )?working on\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)*)",
                "current_project",
                PersonalFactCategory::Context,
                0.8,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)my (?:current )?project is\s+([A-Za-z0-9_-]+)",
                "current_project",
                PersonalFactCategory::Context,
                0.85,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i(?:'m| am) (?:trying to|working to)\s+(.+?)(?:\s*[,.]|$)",
                "current_goal",
                PersonalFactCategory::Context,
                0.7,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)today i(?:'m| am)\s+(.+?)(?:\s*[,.]|$)",
                "current_task",
                PersonalFactCategory::Context,
                0.65,
                1,
                None,
            ),
        ];

        // Constraint patterns
        self.constraint_patterns = vec![
            PatternRule::new(
                r"(?i)i (?:can't|cannot|don't have access to)\s+(.+?)(?:\s*[,.]|$)",
                "cannot_access",
                PersonalFactCategory::Constraint,
                0.8,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i(?:'m| am) in (?:the )?([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+time ?zone",
                "timezone",
                PersonalFactCategory::Constraint,
                0.85,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i(?:'m| am) limited (?:to|by)\s+(.+?)(?:\s*[,.]|$)",
                "limitation",
                PersonalFactCategory::Constraint,
                0.75,
                1,
                None,
            ),
            PatternRule::new(
                r"(?i)i(?:'m| am) not allowed to\s+(.+?)(?:\s*[,.]|$)",
                "restriction",
                PersonalFactCategory::Constraint,
                0.8,
                1,
                None,
            ),
        ];
    }

    /// Process user message and extract any personal facts
    pub fn process_message(&self, message: &str) -> Vec<PersonalFact> {
        if !self.enabled {
            return Vec::new();
        }

        let mut facts = Vec::new();

        // Check all pattern categories
        facts.extend(self.check_patterns(message, &self.identity_patterns));
        facts.extend(self.check_patterns(message, &self.preference_patterns));
        facts.extend(self.check_patterns(message, &self.capability_patterns));
        facts.extend(self.check_patterns(message, &self.context_patterns));
        facts.extend(self.check_patterns(message, &self.constraint_patterns));

        // Filter by minimum confidence
        facts
            .into_iter()
            .filter(|f| f.confidence >= self.min_confidence)
            .collect()
    }

    /// Check message against a set of patterns
    fn check_patterns(&self, message: &str, patterns: &[PatternRule]) -> Vec<PersonalFact> {
        let mut facts = Vec::new();

        for rule in patterns {
            if let Some(captures) = rule.pattern.captures(message)
                && let Some(value_match) = captures.get(rule.value_group)
            {
                let value = value_match.as_str().trim().to_string();

                // Skip very short or very long values
                if value.len() < 2 || value.len() > 100 {
                    continue;
                }

                // Build the key (may contain template placeholders)
                let key = self.build_key(&rule.key_template, &captures);

                // Get optional context
                let context = rule
                    .context_group
                    .and_then(|g| captures.get(g).map(|m| m.as_str().trim().to_string()));

                let fact = PersonalFact::new(
                    rule.category,
                    key,
                    value,
                    context,
                    PersonalFactSource::InferredFromBehavior,
                    false, // Default to synced, not local-only
                );

                // Adjust confidence based on rule
                let mut adjusted_fact = fact;
                adjusted_fact.confidence = rule.confidence;

                facts.push(adjusted_fact);
            }
        }

        facts
    }

    /// Build a key from a template, replacing {n} with capture groups
    fn build_key(&self, template: &str, captures: &regex::Captures) -> String {
        let mut key = template.to_string();

        // Replace {n} patterns with capture groups
        use std::sync::LazyLock;
        static RE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"\{(\d+)\}").expect("valid regex"));
        let re = &*RE;
        for cap in re.captures_iter(template) {
            if let Ok(group_num) = cap[1].parse::<usize>()
                && let Some(value) = captures.get(group_num)
            {
                let replacement = value.as_str().to_lowercase().replace(' ', "_");
                key = key.replace(&cap[0], &replacement);
            }
        }

        key.to_lowercase().replace(' ', "_")
    }

    /// Enable or disable the collector
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the collector is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set minimum confidence threshold
    pub fn set_min_confidence(&mut self, confidence: f32) {
        self.min_confidence = confidence;
    }
}

impl PatternRule {
    fn new(
        pattern: &str,
        key_template: &str,
        category: PersonalFactCategory,
        confidence: f32,
        value_group: usize,
        context_group: Option<usize>,
    ) -> Self {
        Self {
            pattern: Regex::new(pattern).expect("Invalid pattern regex"),
            key_template: key_template.to_string(),
            category,
            confidence,
            value_group,
            context_group,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_creation() {
        let collector = PersonalFactCollector::default();
        assert!(collector.is_enabled());
    }

    #[test]
    fn test_name_detection() {
        let collector = PersonalFactCollector::default();
        let facts = collector.process_message("My name is John Smith");

        assert!(!facts.is_empty());
        let name_fact = facts.iter().find(|f| f.key == "name").unwrap();
        assert_eq!(name_fact.value, "John Smith");
        assert_eq!(name_fact.category, PersonalFactCategory::Identity);
    }

    #[test]
    fn test_preference_detection() {
        let collector = PersonalFactCollector::default();
        let facts = collector.process_message("I prefer Rust over Python");

        assert!(!facts.is_empty());
        let pref_fact = facts.iter().find(|f| f.key == "preference").unwrap();
        assert!(pref_fact.value.contains("Rust"));
        assert_eq!(pref_fact.category, PersonalFactCategory::Preference);
    }

    #[test]
    fn test_current_project_detection() {
        let collector = PersonalFactCollector::default();
        let facts = collector.process_message("I'm working on rullama-cli");

        assert!(!facts.is_empty());
        let project_fact = facts.iter().find(|f| f.key == "current_project").unwrap();
        assert_eq!(project_fact.value, "rullama-cli");
        assert_eq!(project_fact.category, PersonalFactCategory::Context);
    }

    #[test]
    fn test_organization_detection() {
        let collector = PersonalFactCollector::default();
        let facts = collector.process_message("I work at Anthropic");

        assert!(!facts.is_empty());
        let org_fact = facts.iter().find(|f| f.key == "organization").unwrap();
        assert_eq!(org_fact.value, "Anthropic");
        assert_eq!(org_fact.category, PersonalFactCategory::Identity);
    }

    #[test]
    fn test_capability_detection() {
        let collector = PersonalFactCollector::default();
        let facts = collector.process_message("I'm proficient in Rust");

        assert!(!facts.is_empty());
        let cap_fact = facts.iter().find(|f| f.key == "proficient_in").unwrap();
        assert_eq!(cap_fact.value, "Rust");
        assert_eq!(cap_fact.category, PersonalFactCategory::Capability);
    }

    #[test]
    fn test_disabled_collector() {
        let mut collector = PersonalFactCollector::default();
        collector.set_enabled(false);

        let facts = collector.process_message("My name is John");
        assert!(facts.is_empty());
    }

    #[test]
    fn test_confidence_filtering() {
        let collector = PersonalFactCollector::new(0.95, true);
        // Most patterns have confidence < 0.95, so should filter out
        let facts = collector.process_message("I prefer Rust");
        // May or may not have results depending on pattern confidences
        for fact in &facts {
            assert!(fact.confidence >= 0.95);
        }
    }
}
