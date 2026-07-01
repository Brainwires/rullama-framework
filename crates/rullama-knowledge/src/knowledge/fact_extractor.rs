use crate::knowledge::thought::ThoughtCategory;
use crate::knowledge::types::{EvidenceCheckResult, MemorySearchResult};
use regex::Regex;

/// Auto-detect the category of a thought from its text content.
///
/// Uses simple keyword/pattern matching — no LLM call needed.
pub fn detect_category(text: &str) -> ThoughtCategory {
    let lower = text.to_lowercase();

    // Decision indicators
    if contains_any(
        &lower,
        &[
            "decided",
            "chose",
            "going with",
            "picked",
            "selected",
            "settled on",
            "committed to",
        ],
    ) {
        return ThoughtCategory::Decision;
    }

    // Person indicators — capitalized names after relational keywords (check before action items
    // because phrases like "spoke to Sarah about the deadline" should be Person, not ActionItem)
    static PERSON_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r"(?i)\b(?:spoke to|met with|talked to|met|told)\s+[A-Z][a-z]+")
            .expect("valid regex")
    });
    if PERSON_RE.is_match(text) {
        return ThoughtCategory::Person;
    }

    // Insight indicators (check before meeting notes because "async" contains "sync")
    if contains_any(
        &lower,
        &[
            "noticed",
            "realized",
            "learned",
            "discovered",
            "turns out",
            "interesting that",
            "observation",
        ],
    ) {
        return ThoughtCategory::Insight;
    }

    // Action item indicators
    if contains_any(
        &lower,
        &[
            "need to",
            "todo:",
            "todo ",
            "must ",
            "action item",
            "follow up",
            "by friday",
            "by monday",
            "by end of",
        ],
    ) {
        return ThoughtCategory::ActionItem;
    }

    // Idea indicators
    if contains_any(
        &lower,
        &[
            "what if",
            "idea:",
            "could we",
            "how about",
            "maybe we",
            "brainstorm",
            "experiment with",
        ],
    ) {
        return ThoughtCategory::Idea;
    }

    // Meeting note indicators (use word-boundary-aware matching for "sync")
    static MEETING_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r"(?i)\b(?:standup|meeting|discussed|retro|sprint|call with|1:1)\b|\bsync\b")
            .expect("valid regex")
    });
    if MEETING_RE.is_match(text) {
        return ThoughtCategory::MeetingNote;
    }

    // Reference indicators
    static URL_RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"https?://").expect("valid regex"));
    if URL_RE.is_match(text)
        || contains_any(&lower, &["docs at", "reference:", "link:", "see also"])
    {
        return ThoughtCategory::Reference;
    }

    // Auto-captured conversation turns from hooks
    if text.starts_with("[assistant]") || text.starts_with("[user]") {
        return ThoughtCategory::Conversation;
    }

    ThoughtCategory::General
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

// Negation words used to detect contradictions.
const NEGATION_WORDS: &[&str] = &[
    "not ",
    "never",
    "no ",
    "don't",
    "doesn't",
    "isn't",
    "aren't",
    "won't",
    "can't",
    "cannot",
    "didn't",
    "wasn't",
    "weren't",
    "shouldn't",
    "wouldn't",
];

/// Classify a set of search results as corroborations for a new thought.
///
/// A result is a corroboration when its score is ≥ `threshold`.
/// The returned `EvidenceCheckResult` populates only `corroborations`; call
/// [`check_contradiction`] separately to fill `contradictions`.
pub fn check_corroboration(results: &[MemorySearchResult], threshold: f32) -> EvidenceCheckResult {
    let corroborations = results
        .iter()
        .filter(|r| r.score >= threshold)
        .filter_map(|r| r.thought_id.clone())
        .collect();

    EvidenceCheckResult {
        corroborations,
        contradictions: Vec::new(),
    }
}

/// Identify which search results contradict a new thought.
///
/// A result is a contradiction candidate when its score is ≥ `threshold`
/// (i.e. it is semantically similar) **and** one of the two pieces of content
/// contains negation language that is absent in the other.  This is a
/// lightweight heuristic — no NLP required.
pub fn check_contradiction(
    new_content: &str,
    results: &[MemorySearchResult],
    threshold: f32,
) -> Vec<String> {
    let new_lower = new_content.to_lowercase();
    let new_has_negation = contains_any(&new_lower, NEGATION_WORDS);

    results
        .iter()
        .filter(|r| r.score >= threshold)
        .filter(|r| {
            let existing_lower = r.content.to_lowercase();
            let existing_has_negation = contains_any(&existing_lower, NEGATION_WORDS);
            // Contradiction: one side negates while the other does not.
            new_has_negation != existing_has_negation
        })
        .filter_map(|r| r.thought_id.clone())
        .collect()
}

/// Extract auto-tags from thought text.
///
/// Pulls out hashtags, @-mentions, and significant capitalised terms.
pub fn extract_tags(text: &str) -> Vec<String> {
    let mut tags = Vec::new();

    // #hashtag extraction
    static HASHTAG_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r"#([A-Za-z][A-Za-z0-9_-]{1,30})").expect("valid regex")
    });
    for cap in HASHTAG_RE.captures_iter(text) {
        let tag = cap[1].to_lowercase();
        if !tags.contains(&tag) {
            tags.push(tag);
        }
    }

    // Common tech name detection
    static TECH_NAMES: &[(&str, &str)] = &[
        ("react", "react"),
        ("nextjs", "nextjs"),
        ("next.js", "nextjs"),
        ("postgresql", "postgresql"),
        ("postgres", "postgresql"),
        ("sqlite", "sqlite"),
        ("redis", "redis"),
        ("mongodb", "mongodb"),
        ("docker", "docker"),
        ("kubernetes", "kubernetes"),
        ("k8s", "kubernetes"),
        ("typescript", "typescript"),
        ("javascript", "javascript"),
        ("python", "python"),
        ("rust", "rust"),
        ("golang", "golang"),
        ("graphql", "graphql"),
        ("grpc", "grpc"),
        ("websocket", "websocket"),
        ("supabase", "supabase"),
        ("firebase", "firebase"),
        ("aws", "aws"),
        ("terraform", "terraform"),
        ("nginx", "nginx"),
        ("linux", "linux"),
        ("git", "git"),
        ("github", "github"),
        ("claude", "claude"),
        ("openai", "openai"),
        ("lancedb", "lancedb"),
        ("tokio", "tokio"),
    ];

    let lower = text.to_lowercase();
    for &(pattern, tag) in TECH_NAMES {
        if lower.contains(pattern) && !tags.contains(&tag.to_string()) {
            tags.push(tag.to_string());
        }
    }

    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_detection() {
        assert_eq!(
            detect_category("Decided to use PostgreSQL for the auth service"),
            ThoughtCategory::Decision
        );
        assert_eq!(
            detect_category("Going with React for the frontend"),
            ThoughtCategory::Decision
        );
    }

    #[test]
    fn test_person_detection() {
        assert_eq!(
            detect_category("Spoke to Sarah about the deadline"),
            ThoughtCategory::Person
        );
        assert_eq!(
            detect_category("Met with John to discuss the architecture"),
            ThoughtCategory::Person
        );
    }

    #[test]
    fn test_insight_detection() {
        assert_eq!(
            detect_category("Noticed that batch processing is 3x faster with async"),
            ThoughtCategory::Insight
        );
        assert_eq!(
            detect_category("Realized the bottleneck is in the serialization"),
            ThoughtCategory::Insight
        );
    }

    #[test]
    fn test_meeting_note_detection() {
        assert_eq!(
            detect_category("Standup: team agreed to prioritize the auth refactor"),
            ThoughtCategory::MeetingNote
        );
    }

    #[test]
    fn test_idea_detection() {
        assert_eq!(
            detect_category("What if we used WebSockets instead of polling?"),
            ThoughtCategory::Idea
        );
        assert_eq!(
            detect_category("Idea: cache the embeddings in Redis"),
            ThoughtCategory::Idea
        );
    }

    #[test]
    fn test_action_item_detection() {
        assert_eq!(
            detect_category("Need to review PR #234 before Friday"),
            ThoughtCategory::ActionItem
        );
        assert_eq!(
            detect_category("TODO: update the API docs"),
            ThoughtCategory::ActionItem
        );
    }

    #[test]
    fn test_reference_detection() {
        assert_eq!(
            detect_category("The API docs are at https://docs.example.com"),
            ThoughtCategory::Reference
        );
    }

    #[test]
    fn test_general_fallback() {
        assert_eq!(
            detect_category("Just a random note"),
            ThoughtCategory::General
        );
    }

    #[test]
    fn test_tag_extraction() {
        let tags = extract_tags("Working on #rust and #mcp-server today");
        assert!(tags.contains(&"rust".to_string()));
        assert!(tags.contains(&"mcp-server".to_string()));
    }
}
