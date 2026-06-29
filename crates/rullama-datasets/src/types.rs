use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Role in a training conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainingRole {
    /// System prompt role.
    System,
    /// User message role.
    User,
    /// Assistant response role.
    Assistant,
    /// Tool output role.
    Tool,
}

impl std::fmt::Display for TrainingRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

/// A single message in a training conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMessage {
    /// Role of the message sender.
    pub role: TrainingRole,
    /// Text content of the message.
    pub content: String,
    /// Optional tool calls made by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// ID of the tool call this message responds to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional name of the sender.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl TrainingMessage {
    /// Create a new training message with the given role and content.
    pub fn new(role: TrainingRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(TrainingRole::System, content)
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(TrainingRole::User, content)
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(TrainingRole::Assistant, content)
    }

    /// Create a tool response message.
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: TrainingRole::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
        }
    }

    /// Estimated token count (rough: ~4 chars per token).
    pub fn estimated_tokens(&self) -> usize {
        self.content.len() / 4 + 1
    }
}

/// A training example consisting of a multi-turn conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Unique identifier for this example.
    #[serde(default = "generate_id")]
    pub id: String,
    /// Ordered list of messages in the conversation.
    pub messages: Vec<TrainingMessage>,
    /// Arbitrary metadata attached to this example.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

fn generate_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

impl TrainingExample {
    /// Create a new training example with an auto-generated ID.
    pub fn new(messages: Vec<TrainingMessage>) -> Self {
        Self {
            id: generate_id(),
            messages,
            metadata: HashMap::new(),
        }
    }

    /// Create a new training example with a specific ID.
    pub fn with_id(id: impl Into<String>, messages: Vec<TrainingMessage>) -> Self {
        Self {
            id: id.into(),
            messages,
            metadata: HashMap::new(),
        }
    }

    /// Total estimated token count across all messages.
    pub fn estimated_tokens(&self) -> usize {
        self.messages.iter().map(|m| m.estimated_tokens()).sum()
    }

    /// Check if this example has a system message.
    pub fn has_system_message(&self) -> bool {
        self.messages.iter().any(|m| m.role == TrainingRole::System)
    }

    /// Check if the last message is from the assistant (completion target).
    pub fn ends_with_assistant(&self) -> bool {
        self.messages
            .last()
            .is_some_and(|m| m.role == TrainingRole::Assistant)
    }
}

/// A preference pair for DPO/ORPO training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePair {
    /// Unique identifier for this preference pair.
    #[serde(default = "generate_id")]
    pub id: String,
    /// The shared prompt messages.
    pub prompt: Vec<TrainingMessage>,
    /// The preferred (chosen) response messages.
    pub chosen: Vec<TrainingMessage>,
    /// The rejected response messages.
    pub rejected: Vec<TrainingMessage>,
    /// Arbitrary metadata attached to this pair.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PreferencePair {
    /// Create a new preference pair with an auto-generated ID.
    pub fn new(
        prompt: Vec<TrainingMessage>,
        chosen: Vec<TrainingMessage>,
        rejected: Vec<TrainingMessage>,
    ) -> Self {
        Self {
            id: generate_id(),
            prompt,
            chosen,
            rejected,
            metadata: HashMap::new(),
        }
    }

    /// Total estimated tokens for prompt + chosen + rejected.
    pub fn estimated_tokens(&self) -> usize {
        let prompt_tokens: usize = self.prompt.iter().map(|m| m.estimated_tokens()).sum();
        let chosen_tokens: usize = self.chosen.iter().map(|m| m.estimated_tokens()).sum();
        let rejected_tokens: usize = self.rejected.iter().map(|m| m.estimated_tokens()).sum();
        prompt_tokens + chosen_tokens + rejected_tokens
    }
}

/// Supported data formats for import/export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataFormat {
    /// OpenAI fine-tuning format.
    OpenAI,
    /// Together AI fine-tuning format.
    Together,
    /// Alpaca instruction-following format.
    Alpaca,
    /// ShareGPT conversation format.
    ShareGpt,
    /// ChatML template format.
    ChatMl,
}

impl std::fmt::Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAI => write!(f, "openai"),
            Self::Together => write!(f, "together"),
            Self::Alpaca => write!(f, "alpaca"),
            Self::ShareGpt => write!(f, "sharegpt"),
            Self::ChatMl => write!(f, "chatml"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_message_creation() {
        let msg = TrainingMessage::system("You are a helpful assistant.");
        assert_eq!(msg.role, TrainingRole::System);
        assert_eq!(msg.content, "You are a helpful assistant.");
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn test_training_example() {
        let example = TrainingExample::new(vec![
            TrainingMessage::system("You are helpful."),
            TrainingMessage::user("Hello"),
            TrainingMessage::assistant("Hi there!"),
        ]);
        assert_eq!(example.messages.len(), 3);
        assert!(example.has_system_message());
        assert!(example.ends_with_assistant());
        assert!(example.estimated_tokens() > 0);
    }

    #[test]
    fn test_preference_pair() {
        let pair = PreferencePair::new(
            vec![TrainingMessage::user("What is 2+2?")],
            vec![TrainingMessage::assistant("4")],
            vec![TrainingMessage::assistant("22")],
        );
        assert_eq!(pair.prompt.len(), 1);
        assert_eq!(pair.chosen.len(), 1);
        assert_eq!(pair.rejected.len(), 1);
    }

    #[test]
    fn test_training_role_display() {
        assert_eq!(TrainingRole::System.to_string(), "system");
        assert_eq!(TrainingRole::User.to_string(), "user");
        assert_eq!(TrainingRole::Assistant.to_string(), "assistant");
        assert_eq!(TrainingRole::Tool.to_string(), "tool");
    }

    #[test]
    fn test_data_format_display() {
        assert_eq!(DataFormat::OpenAI.to_string(), "openai");
        assert_eq!(DataFormat::Together.to_string(), "together");
        assert_eq!(DataFormat::ShareGpt.to_string(), "sharegpt");
    }

    #[test]
    fn test_training_message_serialization() {
        let msg = TrainingMessage::assistant("Hello world");
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: TrainingMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, TrainingRole::Assistant);
        assert_eq!(parsed.content, "Hello world");
    }
}
