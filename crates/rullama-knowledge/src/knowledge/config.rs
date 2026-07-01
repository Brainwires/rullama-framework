//! Memory bank configuration: mission, directives, and disposition traits.
//!
//! [`MemoryBankConfig`] shapes how a [`crate::knowledge::BrainClient`] annotates
//! stored thoughts and filters/scores search results.  The default configuration
//! is a no-op — everything works exactly as before unless you opt in.

use serde::{Deserialize, Serialize};

/// Behavioral reasoning traits that bias search result scoring.
///
/// Each active trait applies a small score delta (positive or negative) to
/// each result based on content characteristics.  Deltas are summed and
/// clamped to `[-0.1, 0.1]` before being applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispositionTrait {
    /// Boost structured, data-driven, or reasoned content.
    Analytical,
    /// Penalise very long content; prefer concise results.
    Concise,
    /// Boost content that expresses caution or uncertainty.
    Cautious,
    /// Boost content that presents novel ideas or possibilities.
    Creative,
    /// Boost content that follows sequential or procedural structure.
    Systematic,
}

/// Configuration that shapes how a [`crate::knowledge::BrainClient`] memory
/// bank behaves.
///
/// ## Mission
/// A short statement of the agent's purpose.  When set, every captured thought
/// is tagged with a normalised mission slug (e.g. `"mission:security_assistant"`)
/// so thoughts can be scoped by client identity.
///
/// ## Directives
/// Compliance or safety rules.  Directives that start with `"Never "` or
/// `"Do not "` are parsed as content-blocking rules: any search result whose
/// content contains words from the directive's object phrase is removed from
/// the response.  Example: `"Never store PII"` will remove results that
/// contain the words "store" **and** "Pii" (case-insensitive).
///
/// ## Disposition
/// A set of reasoning traits that apply small score adjustments (±0.1) based
/// on the content characteristics of each result.  The net delta is clamped
/// so no single result can gain or lose more than 0.1.
///
/// # Default
/// All fields are empty/`None` — the config is a no-op.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryBankConfig {
    /// Agent mission statement (used for auto-tagging).
    pub mission: Option<String>,
    /// Content blocking / compliance directives.
    pub directives: Vec<String>,
    /// Reasoning trait biases applied to search scores.
    pub disposition: Vec<DispositionTrait>,
}

impl MemoryBankConfig {
    /// Create an empty (no-op) config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the mission (builder-style).
    pub fn with_mission(mut self, mission: impl Into<String>) -> Self {
        self.mission = Some(mission.into());
        self
    }

    /// Append a directive (builder-style).
    pub fn with_directive(mut self, directive: impl Into<String>) -> Self {
        self.directives.push(directive.into());
        self
    }

    /// Append a disposition trait (builder-style).
    pub fn with_disposition(mut self, trait_: DispositionTrait) -> Self {
        if !self.disposition.contains(&trait_) {
            self.disposition.push(trait_);
        }
        self
    }

    /// Returns `true` when the config has no effect (all fields empty/None).
    pub fn is_noop(&self) -> bool {
        self.mission.is_none() && self.directives.is_empty() && self.disposition.is_empty()
    }

    /// Returns a mission slug suitable for use as a tag.
    ///
    /// Lowercases the mission and replaces whitespace with `_`.
    pub fn mission_tag(&self) -> Option<String> {
        self.mission.as_ref().map(|m| {
            let slug = m
                .to_lowercase()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join("_");
            format!("mission:{slug}")
        })
    }

    /// Returns `true` if the given content should be excluded by a blocking
    /// directive.
    ///
    /// A directive is treated as blocking when it starts with `"Never "` or
    /// `"Do not "`.  The object phrase (the part after the prefix) is split
    /// into words; the content is blocked if it contains **all** of those
    /// words (case-insensitive).
    pub fn blocks_content(&self, content: &str) -> bool {
        let lower_content = content.to_lowercase();
        for directive in &self.directives {
            let object = if let Some(rest) = directive.strip_prefix("Never ") {
                rest
            } else if let Some(rest) = directive.strip_prefix("Do not ") {
                rest
            } else {
                continue; // not a blocking directive
            };

            let words: Vec<&str> = object.split_whitespace().collect();
            if !words.is_empty()
                && words
                    .iter()
                    .all(|w| lower_content.contains(&w.to_lowercase()))
            {
                return true;
            }
        }
        false
    }

    /// Compute a score delta in `[-0.1, 0.1]` based on disposition traits and
    /// content characteristics.
    pub fn disposition_score_delta(&self, content: &str) -> f32 {
        if self.disposition.is_empty() {
            return 0.0;
        }

        let lower = content.to_lowercase();
        let mut delta: f32 = 0.0;

        for trait_ in &self.disposition {
            delta += match trait_ {
                DispositionTrait::Analytical => {
                    // Boost content with structured markers: numbers, code blocks, bullet points
                    let has_numbers = lower.chars().any(|c| c.is_ascii_digit());
                    let has_code = lower.contains("```") || lower.contains("    ");
                    let has_bullets = lower.contains("- ") || lower.contains("* ");
                    if has_numbers || has_code || has_bullets {
                        0.05
                    } else {
                        0.0
                    }
                }
                DispositionTrait::Concise => {
                    // Penalise very long content
                    if content.len() > 500 { -0.05 } else { 0.0 }
                }
                DispositionTrait::Cautious => {
                    // Boost hedging language
                    let hedges = ["might", "could", "consider", "perhaps", "possibly", "maybe"];
                    if hedges.iter().any(|h| lower.contains(h)) {
                        0.05
                    } else {
                        0.0
                    }
                }
                DispositionTrait::Creative => {
                    // Boost generative / ideation phrasing
                    let creative = [
                        "idea",
                        "what if",
                        "novel",
                        "alternative",
                        "propose",
                        "imagine",
                    ];
                    if creative.iter().any(|c| lower.contains(c)) {
                        0.05
                    } else {
                        0.0
                    }
                }
                DispositionTrait::Systematic => {
                    // Boost sequential / procedural structure
                    let sequential = ["first", "then", "finally", "step ", "1.", "2.", "3."];
                    if sequential.iter().any(|s| lower.contains(s)) {
                        0.05
                    } else {
                        0.0
                    }
                }
            };
        }

        delta.clamp(-0.1, 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_noop() {
        assert!(MemoryBankConfig::default().is_noop());
        assert!(MemoryBankConfig::new().is_noop());
    }

    #[test]
    fn test_builder_chain() {
        let cfg = MemoryBankConfig::new()
            .with_mission("Security assistant")
            .with_directive("Never store PII")
            .with_disposition(DispositionTrait::Analytical);
        assert!(!cfg.is_noop());
        assert_eq!(cfg.mission.as_deref(), Some("Security assistant"));
        assert_eq!(cfg.directives.len(), 1);
        assert_eq!(cfg.disposition.len(), 1);
    }

    #[test]
    fn test_mission_tag() {
        let cfg = MemoryBankConfig::new().with_mission("Security Assistant");
        assert_eq!(cfg.mission_tag(), Some("mission:security_assistant".into()));
        assert!(MemoryBankConfig::new().mission_tag().is_none());
    }

    #[test]
    fn test_blocks_content_never() {
        let cfg = MemoryBankConfig::new().with_directive("Never store PII");
        assert!(cfg.blocks_content("we should store user PII here"));
        assert!(!cfg.blocks_content("authentication token handling"));
    }

    #[test]
    fn test_blocks_content_do_not() {
        let cfg = MemoryBankConfig::new().with_directive("Do not log passwords");
        assert!(cfg.blocks_content("log passwords to the debug output"));
        assert!(!cfg.blocks_content("log request headers"));
    }

    #[test]
    fn test_blocks_content_non_blocking_directive() {
        let cfg = MemoryBankConfig::new().with_directive("Prefer Rust over Python");
        // Not a "Never" or "Do not" directive — should never block
        assert!(!cfg.blocks_content("Prefer Rust over Python everywhere"));
    }

    #[test]
    fn test_disposition_concise_penalty() {
        let cfg = MemoryBankConfig::new().with_disposition(DispositionTrait::Concise);
        let long_content = "x".repeat(501);
        let short_content = "short";
        assert!(cfg.disposition_score_delta(&long_content) < 0.0);
        assert_eq!(cfg.disposition_score_delta(short_content), 0.0);
    }

    #[test]
    fn test_disposition_analytical_boost() {
        let cfg = MemoryBankConfig::new().with_disposition(DispositionTrait::Analytical);
        assert!(cfg.disposition_score_delta("Step 1. Use 42 requests") > 0.0);
        assert_eq!(cfg.disposition_score_delta("casual chat"), 0.0);
    }

    #[test]
    fn test_disposition_delta_clamp() {
        // All traits at once on matching content — delta stays in [-0.1, 0.1]
        let cfg = MemoryBankConfig::new()
            .with_disposition(DispositionTrait::Analytical)
            .with_disposition(DispositionTrait::Cautious)
            .with_disposition(DispositionTrait::Creative)
            .with_disposition(DispositionTrait::Systematic)
            .with_disposition(DispositionTrait::Concise);
        let content = "first idea: might use 42 steps - consider alternatives";
        let delta = cfg.disposition_score_delta(content);
        assert!((-0.1..=0.1).contains(&delta));
    }
}
