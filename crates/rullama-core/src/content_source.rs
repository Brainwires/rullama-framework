use serde::{Deserialize, Serialize};

/// Trust level / origin of content injected into an agent's context.
///
/// Variants are ordered from most-trusted (0) to least-trusted (3) using
/// `#[repr(u8)]`, enabling ordering comparisons:
///
/// ```rust
/// use rullama_core::ContentSource;
/// assert!(ContentSource::SystemPrompt < ContentSource::ExternalContent);
/// assert!(ContentSource::SystemPrompt.can_override(ContentSource::UserInput));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum ContentSource {
    /// Highest trust — operator-defined instructions. Cannot be overridden.
    SystemPrompt = 0,
    /// High trust — content originating directly from the human turn.
    UserInput = 1,
    /// Medium trust — content produced by the agent itself during reasoning.
    AgentReasoning = 2,
    /// Lowest trust — content fetched from the web, external APIs, or
    /// any tool that retrieves data from outside the trusted principal
    /// hierarchy.  Always sanitized before injection.
    ExternalContent = 3,
}

impl ContentSource {
    /// Returns `true` for sources that must be sanitised before injection.
    #[inline]
    pub fn requires_sanitization(self) -> bool {
        self == ContentSource::ExternalContent
    }

    /// Returns `true` if this source is allowed to override `other`.
    ///
    /// A higher-priority (lower numeric value) source can override a
    /// lower-priority one.
    #[inline]
    pub fn can_override(self, other: ContentSource) -> bool {
        self < other
    }
}

impl std::fmt::Display for ContentSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentSource::SystemPrompt => write!(f, "system_prompt"),
            ContentSource::UserInput => write!(f, "user_input"),
            ContentSource::AgentReasoning => write!(f, "agent_reasoning"),
            ContentSource::ExternalContent => write!(f, "external_content"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_is_trust_descending() {
        assert!(ContentSource::SystemPrompt < ContentSource::UserInput);
        assert!(ContentSource::UserInput < ContentSource::AgentReasoning);
        assert!(ContentSource::AgentReasoning < ContentSource::ExternalContent);
    }

    #[test]
    fn requires_sanitization_only_for_external() {
        assert!(!ContentSource::SystemPrompt.requires_sanitization());
        assert!(!ContentSource::UserInput.requires_sanitization());
        assert!(!ContentSource::AgentReasoning.requires_sanitization());
        assert!(ContentSource::ExternalContent.requires_sanitization());
    }

    #[test]
    fn can_override_respects_trust_order() {
        assert!(ContentSource::SystemPrompt.can_override(ContentSource::UserInput));
        assert!(ContentSource::SystemPrompt.can_override(ContentSource::ExternalContent));
        assert!(!ContentSource::ExternalContent.can_override(ContentSource::SystemPrompt));
        assert!(!ContentSource::UserInput.can_override(ContentSource::UserInput));
    }

    #[test]
    fn display_names() {
        assert_eq!(ContentSource::SystemPrompt.to_string(), "system_prompt");
        assert_eq!(ContentSource::UserInput.to_string(), "user_input");
        assert_eq!(ContentSource::AgentReasoning.to_string(), "agent_reasoning");
        assert_eq!(
            ContentSource::ExternalContent.to_string(),
            "external_content"
        );
    }
}
