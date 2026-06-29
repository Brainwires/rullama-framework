//! Shared email types used by IMAP and SMTP clients.

use serde::{Deserialize, Serialize};

/// An email message with headers, body, and optional attachments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailMessage {
    /// Sender address.
    pub from: String,
    /// Primary recipients.
    pub to: Vec<String>,
    /// Carbon-copy recipients.
    #[serde(default)]
    pub cc: Vec<String>,
    /// Blind carbon-copy recipients.
    #[serde(default)]
    pub bcc: Vec<String>,
    /// Message subject line.
    pub subject: String,
    /// Plain-text body.
    #[serde(default)]
    pub body: Option<String>,
    /// HTML body.
    #[serde(default)]
    pub body_html: Option<String>,
    /// File attachments.
    #[serde(default)]
    pub attachments: Vec<EmailAttachment>,
    /// RFC-2822 date string.
    #[serde(default)]
    pub date: Option<String>,
    /// IMAP message UID.
    #[serde(default)]
    pub uid: Option<u32>,
    /// RFC message-id header.
    #[serde(default)]
    pub message_id: Option<String>,
    /// IMAP flags (e.g. `\Seen`, `\Flagged`).
    #[serde(default)]
    pub flags: Vec<String>,
}

/// An email attachment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailAttachment {
    /// File name.
    pub filename: String,
    /// MIME content type.
    pub content_type: String,
    /// Raw attachment data (base64-encoded when serialized to JSON).
    #[serde(with = "base64_bytes")]
    pub data: Vec<u8>,
}

/// An IMAP folder (mailbox).
#[allow(dead_code)] // reason: public API, surfaced via list_folders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailFolder {
    /// Folder name (e.g. `INBOX`).
    pub name: String,
    /// Total number of messages.
    pub total_messages: u32,
    /// Number of unread messages.
    pub unread: u32,
}

/// Search criteria for IMAP search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmailSearchQuery {
    /// Filter by sender address.
    #[serde(default)]
    pub from: Option<String>,
    /// Filter by recipient address.
    #[serde(default)]
    pub to: Option<String>,
    /// Filter by subject text.
    #[serde(default)]
    pub subject: Option<String>,
    /// Filter by body text.
    #[serde(default)]
    pub body: Option<String>,
    /// Messages on or after this date (RFC-3339).
    #[serde(default)]
    pub since: Option<String>,
    /// Messages before this date (RFC-3339).
    #[serde(default)]
    pub before: Option<String>,
    /// Filter by IMAP flags.
    #[serde(default)]
    pub flags: Vec<String>,
}

/// Helper module for base64-encoding `Vec<u8>` in serde.
mod base64_bytes {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(data))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(deserializer)?;
        STANDARD.decode(s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_message_serde_roundtrip() {
        let msg = EmailMessage {
            from: "alice@example.com".to_string(),
            to: vec!["bob@example.com".to_string()],
            cc: vec![],
            bcc: vec![],
            subject: "Hello".to_string(),
            body: Some("Hi Bob!".to_string()),
            body_html: None,
            attachments: vec![],
            date: Some("2025-01-01T00:00:00Z".to_string()),
            uid: Some(42),
            message_id: Some("<msg@example.com>".to_string()),
            flags: vec!["\\Seen".to_string()],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: EmailMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.from, "alice@example.com");
        assert_eq!(deserialized.subject, "Hello");
        assert_eq!(deserialized.uid, Some(42));
    }

    #[test]
    fn test_email_attachment_serde_roundtrip() {
        let att = EmailAttachment {
            filename: "test.txt".to_string(),
            content_type: "text/plain".to_string(),
            data: b"hello world".to_vec(),
        };
        let json = serde_json::to_string(&att).unwrap();
        let deserialized: EmailAttachment = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.filename, "test.txt");
        assert_eq!(deserialized.data, b"hello world");
    }

    #[test]
    fn test_email_folder_serde_roundtrip() {
        let folder = EmailFolder {
            name: "INBOX".to_string(),
            total_messages: 100,
            unread: 5,
        };
        let json = serde_json::to_string(&folder).unwrap();
        let deserialized: EmailFolder = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "INBOX");
        assert_eq!(deserialized.total_messages, 100);
    }

    #[test]
    fn test_email_search_query_serde_roundtrip() {
        let query = EmailSearchQuery {
            from: Some("alice@example.com".to_string()),
            subject: Some("Hello".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&query).unwrap();
        let deserialized: EmailSearchQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.from, Some("alice@example.com".to_string()));
    }
}
