//! Internet-facing email identity for agents.
//!
//! Gives an agent a real email address so it can participate in asynchronous
//! communication with humans, services, and other agents over the open
//! internet — no shared infrastructure required.
//!
//! ## Design
//!
//! - [`EmailIdentity`] stores the agent's email address and display name.
//! - [`EmailMessage`] is a language-agnostic representation of an email.
//! - [`EmailProvider`] is an async trait for send + poll backends.
//! - [`HttpEmailProvider`] is a generic REST client that works with any
//!   transactional email service: AgentMail, Mailgun, Postmark, Resend,
//!   SendGrid, or a custom endpoint.
//!
//! ## Example
//!
//! ```rust,no_run
//! use brainwires_network::identity::email::{
//!     EmailIdentity, EmailMessage, HttpEmailProvider, HttpEmailConfig, EmailProvider,
//! };
//!
//! # async fn example() -> anyhow::Result<()> {
//! let identity = EmailIdentity::new("my-agent@agents.example.com", "My Agent");
//!
//! let config = HttpEmailConfig::mailgun("mg.example.com", "key-abc123");
//! let provider = HttpEmailProvider::new(config);
//!
//! let msg = EmailMessage::new(
//!     &identity,
//!     "user@example.com",
//!     "Task complete",
//!     "I finished reviewing the PR.",
//! );
//! provider.send(msg).await?;
//!
//! let inbox = provider.poll(&identity).await?;
//! for email in inbox {
//!     println!("From: {} — {}", email.from, email.subject);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ── Types ─────────────────────────────────────────────────────────────────────

/// An agent's email identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailIdentity {
    /// The agent's email address (e.g. `"agent-42@agents.example.com"`).
    pub address: String,
    /// Display name (e.g. `"Code Review Agent"`).
    pub display_name: String,
}

impl EmailIdentity {
    /// Create a new email identity.
    pub fn new(address: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            display_name: display_name.into(),
        }
    }

    /// Format as `"Display Name <address>"`.
    pub fn formatted(&self) -> String {
        format!("{} <{}>", self.display_name, self.address)
    }
}

/// An email message — either outgoing (to send) or incoming (received).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailMessage {
    /// Message-ID assigned by the provider (empty for outgoing until sent).
    pub id: Option<String>,
    /// Sender address.
    pub from: String,
    /// Primary recipient.
    pub to: String,
    /// CC recipients.
    #[serde(default)]
    pub cc: Vec<String>,
    /// Subject line.
    pub subject: String,
    /// Plain-text body.
    pub body_text: String,
    /// HTML body (optional).
    pub body_html: Option<String>,
    /// When the message was sent / received (UTC).
    pub timestamp: Option<DateTime<Utc>>,
    /// Arbitrary headers / metadata.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Whether this message has been read.
    #[serde(default)]
    pub read: bool,
    /// Thread / conversation ID (provider-specific).
    pub thread_id: Option<String>,
}

impl EmailMessage {
    /// Create an outgoing message.
    pub fn new(
        from: &EmailIdentity,
        to: impl Into<String>,
        subject: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            id: None,
            from: from.formatted(),
            to: to.into(),
            cc: vec![],
            subject: subject.into(),
            body_text: body.into(),
            body_html: None,
            timestamp: Some(Utc::now()),
            headers: HashMap::new(),
            read: false,
            thread_id: None,
        }
    }

    /// Add HTML body.
    pub fn with_html(mut self, html: impl Into<String>) -> Self {
        self.body_html = Some(html.into());
        self
    }

    /// Add CC recipients.
    pub fn with_cc(mut self, cc: Vec<String>) -> Self {
        self.cc = cc;
        self
    }

    /// Reply-to header shorthand.
    pub fn with_reply_to(mut self, reply_to: impl Into<String>) -> Self {
        self.headers.insert("Reply-To".to_string(), reply_to.into());
        self
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Pluggable email backend for an agent.
///
/// Implement this to connect any SMTP service, REST API, or mock backend.
#[async_trait]
pub trait EmailProvider: Send + Sync + 'static {
    /// Send an email message.
    ///
    /// Returns the provider-assigned message ID on success.
    async fn send(&self, message: EmailMessage) -> Result<String, EmailError>;

    /// Poll the inbox for new messages addressed to `identity`.
    ///
    /// Implementations should return only unread messages; callers are
    /// responsible for marking messages read via [`EmailProvider::mark_read`].
    async fn poll(&self, identity: &EmailIdentity) -> Result<Vec<EmailMessage>, EmailError>;

    /// Mark a message as read by its provider message ID.
    async fn mark_read(&self, message_id: &str) -> Result<(), EmailError>;
}

/// Errors returned by [`EmailProvider`] operations.
#[derive(Debug, thiserror::Error)]
pub enum EmailError {
    /// The HTTP request to the email API failed.
    #[error("email API request failed: {0}")]
    Http(String),
    /// The API returned an unexpected response.
    #[error("email API error ({status}): {body}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },
    /// Response body could not be deserialized.
    #[error("failed to parse email API response: {0}")]
    Parse(String),
    /// Feature not supported by this provider implementation.
    #[error("not supported by this provider: {0}")]
    Unsupported(String),
}

// ── HttpEmailProvider ─────────────────────────────────────────────────────────

/// Which REST API flavor to use for sending.
#[derive(Debug, Clone)]
pub enum HttpEmailBackend {
    /// Mailgun — `POST /v3/{domain}/messages` with form encoding.
    Mailgun {
        /// Mailgun sending domain (e.g. `"mg.example.com"`).
        domain: String,
    },
    /// Postmark — `POST /email` with JSON body.
    Postmark,
    /// Resend — `POST /emails` with JSON body.
    Resend,
    /// Custom REST endpoint — `POST {url}` with JSON body (`EmailMessage`).
    Custom {
        /// Full URL including path.
        url: String,
    },
}

/// Configuration for [`HttpEmailProvider`].
#[derive(Debug, Clone)]
pub struct HttpEmailConfig {
    /// Which API backend to target.
    pub backend: HttpEmailBackend,
    /// API key / token sent as `Authorization: Bearer <key>` (or `X-Postmark-Server-Token`).
    pub api_key: String,
    /// Inbox poll URL override (optional; `GET {inbox_url}?to={address}`).
    pub inbox_url: Option<String>,
    /// Additional headers sent with every request.
    pub extra_headers: HashMap<String, String>,
}

impl HttpEmailConfig {
    /// Mailgun configuration.
    pub fn mailgun(domain: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            backend: HttpEmailBackend::Mailgun {
                domain: domain.into(),
            },
            api_key: api_key.into(),
            inbox_url: None,
            extra_headers: HashMap::new(),
        }
    }

    /// Postmark configuration.
    pub fn postmark(server_token: impl Into<String>) -> Self {
        Self {
            backend: HttpEmailBackend::Postmark,
            api_key: server_token.into(),
            inbox_url: None,
            extra_headers: HashMap::new(),
        }
    }

    /// Resend configuration.
    pub fn resend(api_key: impl Into<String>) -> Self {
        Self {
            backend: HttpEmailBackend::Resend,
            api_key: api_key.into(),
            inbox_url: None,
            extra_headers: HashMap::new(),
        }
    }

    /// Custom REST endpoint for both send and poll.
    pub fn custom(send_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            backend: HttpEmailBackend::Custom {
                url: send_url.into(),
            },
            api_key: api_key.into(),
            inbox_url: None,
            extra_headers: HashMap::new(),
        }
    }

    /// Override the inbox poll URL.
    pub fn with_inbox_url(mut self, url: impl Into<String>) -> Self {
        self.inbox_url = Some(url.into());
        self
    }

    /// Add an extra header sent with every request.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.insert(name.into(), value.into());
        self
    }
}

/// Generic HTTP email provider that targets the configured REST backend.
///
/// Covers the send path for Mailgun, Postmark, Resend, and any custom
/// AgentMail-compatible endpoint. For inbox polling, set `inbox_url` in
/// [`HttpEmailConfig`] to point at your inbound webhook / message store API.
pub struct HttpEmailProvider {
    config: HttpEmailConfig,
    client: reqwest::Client,
}

impl HttpEmailProvider {
    /// Build a provider from the given config.
    pub fn new(config: HttpEmailConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn _auth_header(&self) -> String {
        match &self.config.backend {
            HttpEmailBackend::Postmark => self.config.api_key.clone(),
            _ => format!("Bearer {}", self.config.api_key),
        }
    }

    async fn send_mailgun(
        &self,
        domain: &str,
        message: &EmailMessage,
    ) -> Result<String, EmailError> {
        let url = format!("https://api.mailgun.net/v3/{domain}/messages");
        let mut form = vec![
            ("from", message.from.clone()),
            ("to", message.to.clone()),
            ("subject", message.subject.clone()),
            ("text", message.body_text.clone()),
        ];
        if let Some(html) = &message.body_html {
            form.push(("html", html.clone()));
        }
        for cc in &message.cc {
            form.push(("cc", cc.clone()));
        }

        let resp = self
            .client
            .post(&url)
            .basic_auth("api", Some(&self.config.api_key))
            .form(&form)
            .send()
            .await
            .map_err(|e| EmailError::Http(e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        if !(200..300).contains(&status) {
            return Err(EmailError::Api { status, body });
        }

        // Mailgun returns {"id": "<msg-id@domain>", "message": "Queued. Thank you."}
        let v: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| EmailError::Parse(e.to_string()))?;
        Ok(v["id"].as_str().unwrap_or("").to_string())
    }

    async fn send_postmark(&self, message: &EmailMessage) -> Result<String, EmailError> {
        #[derive(Serialize)]
        struct PostmarkBody<'a> {
            #[serde(rename = "From")]
            from: &'a str,
            #[serde(rename = "To")]
            to: &'a str,
            #[serde(rename = "Subject")]
            subject: &'a str,
            #[serde(rename = "TextBody")]
            text_body: &'a str,
            #[serde(rename = "HtmlBody", skip_serializing_if = "Option::is_none")]
            html_body: Option<&'a str>,
        }

        let body = PostmarkBody {
            from: &message.from,
            to: &message.to,
            subject: &message.subject,
            text_body: &message.body_text,
            html_body: message.body_html.as_deref(),
        };

        let resp = self
            .client
            .post("https://api.postmarkapp.com/email")
            .header("X-Postmark-Server-Token", &self.config.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| EmailError::Http(e.to_string()))?;

        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if !(200..300).contains(&status) {
            return Err(EmailError::Api { status, body: text });
        }

        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| EmailError::Parse(e.to_string()))?;
        Ok(v["MessageID"].as_str().unwrap_or("").to_string())
    }

    async fn send_resend(&self, message: &EmailMessage) -> Result<String, EmailError> {
        #[derive(Serialize)]
        struct ResendBody<'a> {
            from: &'a str,
            to: Vec<&'a str>,
            subject: &'a str,
            text: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            html: Option<&'a str>,
        }

        let body = ResendBody {
            from: &message.from,
            to: vec![message.to.as_str()],
            subject: &message.subject,
            text: &message.body_text,
            html: message.body_html.as_deref(),
        };

        let resp = self
            .client
            .post("https://api.resend.com/emails")
            .bearer_auth(&self.config.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| EmailError::Http(e.to_string()))?;

        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if !(200..300).contains(&status) {
            return Err(EmailError::Api { status, body: text });
        }

        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| EmailError::Parse(e.to_string()))?;
        Ok(v["id"].as_str().unwrap_or("").to_string())
    }

    async fn send_custom(&self, url: &str, message: &EmailMessage) -> Result<String, EmailError> {
        let mut req = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .json(message);
        for (k, v) in &self.config.extra_headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let resp = req
            .send()
            .await
            .map_err(|e| EmailError::Http(e.to_string()))?;

        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if !(200..300).contains(&status) {
            return Err(EmailError::Api { status, body: text });
        }

        // Accept either {"id": "..."} or just a bare string
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            return Ok(v["id"].as_str().unwrap_or(&text).to_string());
        }
        Ok(text)
    }
}

#[async_trait]
impl EmailProvider for HttpEmailProvider {
    async fn send(&self, message: EmailMessage) -> Result<String, EmailError> {
        match &self.config.backend {
            HttpEmailBackend::Mailgun { domain } => {
                let domain = domain.clone();
                self.send_mailgun(&domain, &message).await
            }
            HttpEmailBackend::Postmark => self.send_postmark(&message).await,
            HttpEmailBackend::Resend => self.send_resend(&message).await,
            HttpEmailBackend::Custom { url } => {
                let url = url.clone();
                self.send_custom(&url, &message).await
            }
        }
    }

    async fn poll(&self, identity: &EmailIdentity) -> Result<Vec<EmailMessage>, EmailError> {
        let url = self.config.inbox_url.as_deref().ok_or_else(|| {
            EmailError::Unsupported(
                "inbox polling requires inbox_url to be set in HttpEmailConfig".to_string(),
            )
        })?;

        let resp = self
            .client
            .get(url)
            .bearer_auth(&self.config.api_key)
            .query(&[("to", &identity.address)])
            .send()
            .await
            .map_err(|e| EmailError::Http(e.to_string()))?;

        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if !(200..300).contains(&status) {
            return Err(EmailError::Api { status, body: text });
        }

        serde_json::from_str::<Vec<EmailMessage>>(&text)
            .map_err(|e| EmailError::Parse(e.to_string()))
    }

    async fn mark_read(&self, message_id: &str) -> Result<(), EmailError> {
        let base = self.config.inbox_url.as_deref().ok_or_else(|| {
            EmailError::Unsupported(
                "mark_read requires inbox_url to be set in HttpEmailConfig".to_string(),
            )
        })?;

        let url = format!("{base}/{message_id}/read");
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .send()
            .await
            .map_err(|e| EmailError::Http(e.to_string()))?;

        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let body = resp.text().await.unwrap_or_default();
            return Err(EmailError::Api { status, body });
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn email_identity_formatted() {
        let id = EmailIdentity::new("agent@example.com", "My Agent");
        assert_eq!(id.formatted(), "My Agent <agent@example.com>");
    }

    #[test]
    fn email_message_builder() {
        let from = EmailIdentity::new("agent@example.com", "Agent");
        let msg = EmailMessage::new(&from, "user@example.com", "Hello", "Hi there")
            .with_html("<p>Hi there</p>")
            .with_reply_to("noreply@example.com");

        assert_eq!(msg.from, "Agent <agent@example.com>");
        assert_eq!(msg.to, "user@example.com");
        assert_eq!(msg.subject, "Hello");
        assert_eq!(msg.body_text, "Hi there");
        assert_eq!(msg.body_html.as_deref(), Some("<p>Hi there</p>"));
        assert_eq!(msg.headers.get("Reply-To").unwrap(), "noreply@example.com");
    }

    #[test]
    fn http_config_mailgun() {
        let cfg = HttpEmailConfig::mailgun("mg.example.com", "key-123");
        matches!(cfg.backend, HttpEmailBackend::Mailgun { .. });
        assert_eq!(cfg.api_key, "key-123");
    }

    #[test]
    fn http_config_custom_with_inbox() {
        let cfg = HttpEmailConfig::custom("https://api.agentmail.to/send", "tok-abc")
            .with_inbox_url("https://api.agentmail.to/inbox");
        assert_eq!(
            cfg.inbox_url.as_deref(),
            Some("https://api.agentmail.to/inbox")
        );
    }

    #[test]
    fn provider_construction() {
        let cfg = HttpEmailConfig::resend("re_abc123");
        let _provider = HttpEmailProvider::new(cfg);
    }
}
