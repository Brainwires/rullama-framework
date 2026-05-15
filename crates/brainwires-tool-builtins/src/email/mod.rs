//! Email tools: send, search, read, and list email messages via IMAP/SMTP.
//!
//! Inbound email can be ingested two ways:
//!
//! - **IMAP polling** — the historical path, driven by [`ImapClient`].
//! - **Gmail push** — the low-latency path, driven by [`gmail_push::GmailPushHandler`].
//!   Gmail delivers Pub/Sub webhooks to the BrainClaw gateway which
//!   authenticates Google's signed JWT and pulls the new messages via the
//!   Gmail REST API.
//!
//! When both are configured for the same account, Gmail push wins — see
//! [`EmailSource`] and the startup warning emitted by the daemon.

pub mod gmail_push;
pub mod imap_client;
pub mod smtp_client;
pub mod types;

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

use self::imap_client::ImapClient;
use self::smtp_client::SmtpClient;
use self::types::EmailSearchQuery;

/// Where inbound email for an account is pulled from.
///
/// This is an operator-facing configuration pivot: each account may be
/// connected via classical IMAP polling *or* Gmail Pub/Sub push. When
/// both are configured for the same address, BrainClaw prefers push and
/// suppresses IMAP to avoid double delivery. See the daemon startup logs
/// for the warning that documents the choice.
#[derive(Debug, Clone)]
pub enum EmailSource {
    /// Classical IMAP polling — the historical path.
    Imap(EmailConfig),
    /// Gmail push via Google Pub/Sub — the low-latency path.
    GmailPush(gmail_push::GmailPushConfig),
}

/// Email provider configuration variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmailProvider {
    /// IMAP + SMTP provider (most common).
    ImapSmtp {
        /// IMAP server hostname.
        imap_host: String,
        /// IMAP server port (993 for TLS).
        imap_port: u16,
        /// SMTP server hostname.
        smtp_host: String,
        /// SMTP server port (587 for STARTTLS, 465 for TLS).
        smtp_port: u16,
        /// Login username.
        username: String,
        /// Login password.
        password: String,
        /// Whether to use TLS.
        tls: bool,
    },
}

/// Configuration for the email tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    /// Email provider settings.
    pub provider: EmailProvider,
    /// Default "from" address for sending.
    pub from_address: String,
}

/// Email tool implementation providing send, search, read, and list operations.
pub struct EmailTool;

impl EmailTool {
    /// Return tool definitions for email operations.
    pub fn get_tools() -> Vec<Tool> {
        vec![
            Self::email_send_tool(),
            Self::email_search_tool(),
            Self::email_read_tool(),
            Self::email_list_tool(),
        ]
    }

    fn email_send_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "to".to_string(),
            json!({"type": "array", "items": {"type": "string"}, "description": "Recipient email addresses"}),
        );
        properties.insert(
            "subject".to_string(),
            json!({"type": "string", "description": "Email subject line"}),
        );
        properties.insert(
            "body".to_string(),
            json!({"type": "string", "description": "Plain-text email body"}),
        );
        properties.insert(
            "cc".to_string(),
            json!({"type": "array", "items": {"type": "string"}, "description": "CC recipients"}),
        );
        properties.insert(
            "bcc".to_string(),
            json!({"type": "array", "items": {"type": "string"}, "description": "BCC recipients"}),
        );
        Tool {
            name: "email_send".to_string(),
            description: "Send an email message via SMTP.".to_string(),
            input_schema: ToolInputSchema::object(
                properties,
                vec!["to".to_string(), "subject".to_string(), "body".to_string()],
            ),
            requires_approval: true,
            ..Default::default()
        }
    }

    fn email_search_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "folder".to_string(),
            json!({"type": "string", "description": "IMAP folder to search (default: INBOX)"}),
        );
        properties.insert(
            "from".to_string(),
            json!({"type": "string", "description": "Filter by sender address"}),
        );
        properties.insert(
            "to".to_string(),
            json!({"type": "string", "description": "Filter by recipient address"}),
        );
        properties.insert(
            "subject".to_string(),
            json!({"type": "string", "description": "Filter by subject text"}),
        );
        properties.insert(
            "body".to_string(),
            json!({"type": "string", "description": "Filter by body text"}),
        );
        properties.insert(
            "since".to_string(),
            json!({"type": "string", "description": "Messages on or after this date (IMAP date format)"}),
        );
        properties.insert(
            "before".to_string(),
            json!({"type": "string", "description": "Messages before this date (IMAP date format)"}),
        );
        Tool {
            name: "email_search".to_string(),
            description: "Search email messages in an IMAP folder using filter criteria."
                .to_string(),
            input_schema: ToolInputSchema::object(properties, vec![]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn email_read_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "uid".to_string(),
            json!({"type": "integer", "description": "IMAP message UID to read"}),
        );
        properties.insert(
            "folder".to_string(),
            json!({"type": "string", "description": "IMAP folder containing the message (default: INBOX)"}),
        );
        Tool {
            name: "email_read".to_string(),
            description: "Read a full email message by UID, including body and attachments."
                .to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["uid".to_string()]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn email_list_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "folder".to_string(),
            json!({"type": "string", "description": "IMAP folder to list (default: INBOX)"}),
        );
        properties.insert(
            "limit".to_string(),
            json!({"type": "integer", "description": "Maximum number of messages to return (default: 20)"}),
        );
        properties.insert(
            "offset".to_string(),
            json!({"type": "integer", "description": "Offset for pagination (default: 0)"}),
        );
        Tool {
            name: "email_list".to_string(),
            description: "List email message summaries from an IMAP folder.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec![]),
            requires_approval: false,
            ..Default::default()
        }
    }

    /// Execute an email tool by name.
    #[tracing::instrument(name = "tool.execute", skip(input, context), fields(tool_name))]
    pub async fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "email_send" => Self::handle_send(input, context).await,
            "email_search" => Self::handle_search(input, context).await,
            "email_read" => Self::handle_read(input, context).await,
            "email_list" => Self::handle_list(input, context).await,
            _ => Err(anyhow::anyhow!("Unknown email tool: {}", tool_name)),
        };
        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Email operation failed: {}", e),
            ),
        }
    }

    // ── Handler implementations ─────────────────────────────────────────────

    async fn handle_send(input: &Value, context: &ToolContext) -> Result<String> {
        let config = Self::get_config(context)?;

        match &config.provider {
            EmailProvider::ImapSmtp {
                smtp_host,
                smtp_port,
                username,
                password,
                tls,
                ..
            } => {
                let client = SmtpClient::new(
                    smtp_host,
                    *smtp_port,
                    username,
                    password,
                    *tls,
                    &config.from_address,
                )?;

                #[derive(Deserialize)]
                struct SendInput {
                    to: Vec<String>,
                    subject: String,
                    body: String,
                    #[serde(default)]
                    cc: Vec<String>,
                    #[serde(default)]
                    bcc: Vec<String>,
                }

                let params: SendInput = serde_json::from_value(input.clone())?;
                client
                    .send_email(
                        &params.to,
                        &params.cc,
                        &params.bcc,
                        &params.subject,
                        &params.body,
                        &[],
                    )
                    .await
            }
        }
    }

    async fn handle_search(input: &Value, context: &ToolContext) -> Result<String> {
        let config = Self::get_config(context)?;

        match &config.provider {
            EmailProvider::ImapSmtp {
                imap_host,
                imap_port,
                username,
                password,
                tls,
                ..
            } => {
                let mut client =
                    ImapClient::connect(imap_host, *imap_port, username, password, *tls).await?;

                let folder = input
                    .get("folder")
                    .and_then(|v| v.as_str())
                    .unwrap_or("INBOX");

                let query = EmailSearchQuery {
                    from: input.get("from").and_then(|v| v.as_str()).map(String::from),
                    to: input.get("to").and_then(|v| v.as_str()).map(String::from),
                    subject: input
                        .get("subject")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    body: input.get("body").and_then(|v| v.as_str()).map(String::from),
                    since: input
                        .get("since")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    before: input
                        .get("before")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    flags: vec![],
                };

                let uids = client.search_messages(&query, folder).await?;
                let _ = client.logout().await;

                Ok(serde_json::to_string_pretty(&uids)?)
            }
        }
    }

    async fn handle_read(input: &Value, context: &ToolContext) -> Result<String> {
        let config = Self::get_config(context)?;

        match &config.provider {
            EmailProvider::ImapSmtp {
                imap_host,
                imap_port,
                username,
                password,
                tls,
                ..
            } => {
                let mut client =
                    ImapClient::connect(imap_host, *imap_port, username, password, *tls).await?;

                let uid = input
                    .get("uid")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow::anyhow!("'uid' is required"))?
                    as u32;

                let folder = input
                    .get("folder")
                    .and_then(|v| v.as_str())
                    .unwrap_or("INBOX");

                // Select the folder before reading
                client.list_messages(folder, 0, 0).await.ok();

                let msg = client.read_message(uid).await?;
                let _ = client.logout().await;

                Ok(serde_json::to_string_pretty(&msg)?)
            }
        }
    }

    async fn handle_list(input: &Value, context: &ToolContext) -> Result<String> {
        let config = Self::get_config(context)?;

        match &config.provider {
            EmailProvider::ImapSmtp {
                imap_host,
                imap_port,
                username,
                password,
                tls,
                ..
            } => {
                let mut client =
                    ImapClient::connect(imap_host, *imap_port, username, password, *tls).await?;

                let folder = input
                    .get("folder")
                    .and_then(|v| v.as_str())
                    .unwrap_or("INBOX");
                let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as u32;
                let offset = input.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

                let messages = client.list_messages(folder, limit, offset).await?;
                let _ = client.logout().await;

                Ok(serde_json::to_string_pretty(&messages)?)
            }
        }
    }

    /// Extract email configuration from the tool context metadata.
    fn get_config(context: &ToolContext) -> Result<EmailConfig> {
        let config_json = context.metadata.get("email_config").ok_or_else(|| {
            anyhow::anyhow!(
                "Email configuration not found. Set 'email_config' in ToolContext.metadata."
            )
        })?;
        let config: EmailConfig = serde_json::from_str(config_json)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tools() {
        let tools = EmailTool::get_tools();
        assert_eq!(tools.len(), 4);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"email_send"));
        assert!(names.contains(&"email_search"));
        assert!(names.contains(&"email_read"));
        assert!(names.contains(&"email_list"));
    }

    #[test]
    fn test_email_send_requires_approval() {
        let tools = EmailTool::get_tools();
        let send = tools.iter().find(|t| t.name == "email_send").unwrap();
        assert!(send.requires_approval);
    }

    #[test]
    fn test_email_send_required_fields() {
        let tools = EmailTool::get_tools();
        let send = tools.iter().find(|t| t.name == "email_send").unwrap();
        let required = send.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"to".to_string()));
        assert!(required.contains(&"subject".to_string()));
        assert!(required.contains(&"body".to_string()));
    }

    #[test]
    fn test_email_read_required_fields() {
        let tools = EmailTool::get_tools();
        let read = tools.iter().find(|t| t.name == "email_read").unwrap();
        let required = read.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"uid".to_string()));
    }

    #[test]
    fn test_email_config_serde_roundtrip() {
        let config = EmailConfig {
            provider: EmailProvider::ImapSmtp {
                imap_host: "imap.example.com".to_string(),
                imap_port: 993,
                smtp_host: "smtp.example.com".to_string(),
                smtp_port: 587,
                username: "user@example.com".to_string(),
                password: "secret".to_string(),
                tls: true,
            },
            from_address: "user@example.com".to_string(),
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EmailConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.from_address, "user@example.com");
    }

    #[tokio::test]
    async fn test_execute_unknown_tool() {
        let context = ToolContext {
            working_directory: ".".to_string(),
            ..Default::default()
        };
        let input = json!({});
        let result = EmailTool::execute("1", "unknown_email_tool", &input, &context).await;
        assert!(result.is_error);
    }
}
