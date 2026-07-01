//! SMTP client for sending email messages.

use anyhow::{Context, Result};
use lettre::message::header::ContentType;
use lettre::message::{Attachment, MultiPart, SinglePart};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};

use super::types::EmailAttachment;

/// SMTP client wrapping a lettre async transport.
pub struct SmtpClient {
    transport: AsyncSmtpTransport<Tokio1Executor>,
    from: String,
}

impl SmtpClient {
    /// Create a new SMTP client and connect to the server.
    pub fn new(
        host: &str,
        port: u16,
        username: &str,
        password: &str,
        tls: bool,
        from: &str,
    ) -> Result<Self> {
        let credentials = Credentials::new(username.to_string(), password.to_string());

        let transport = if tls {
            AsyncSmtpTransport::<Tokio1Executor>::relay(host)
                .context("Failed to create SMTP relay")?
                .port(port)
                .credentials(credentials)
                .build()
        } else {
            AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(host)
                .port(port)
                .credentials(credentials)
                .build()
        };

        Ok(Self {
            transport,
            from: from.to_string(),
        })
    }

    /// Send an email message.
    pub async fn send_email(
        &self,
        to: &[String],
        cc: &[String],
        bcc: &[String],
        subject: &str,
        body: &str,
        attachments: &[EmailAttachment],
    ) -> Result<String> {
        let mut builder = Message::builder()
            .from(self.from.parse().context("Invalid 'from' address")?)
            .subject(subject);

        for addr in to {
            builder = builder.to(addr.parse().context("Invalid 'to' address")?);
        }
        for addr in cc {
            builder = builder.cc(addr.parse().context("Invalid 'cc' address")?);
        }
        for addr in bcc {
            builder = builder.bcc(addr.parse().context("Invalid 'bcc' address")?);
        }

        let message = if attachments.is_empty() {
            builder
                .body(body.to_string())
                .context("Failed to build email message")?
        } else {
            let text_part = SinglePart::builder()
                .content_type(ContentType::TEXT_PLAIN)
                .body(body.to_string());

            let mut multipart = MultiPart::mixed().singlepart(text_part);

            for att in attachments {
                let content_type: ContentType =
                    ContentType::parse(&att.content_type).unwrap_or(ContentType::TEXT_PLAIN);
                let attachment =
                    Attachment::new(att.filename.clone()).body(att.data.clone(), content_type);
                multipart = multipart.singlepart(attachment);
            }

            builder
                .multipart(multipart)
                .context("Failed to build multipart email")?
        };

        let response = self
            .transport
            .send(message)
            .await
            .context("Failed to send email")?;

        Ok(format!(
            "Email sent successfully (code: {})",
            response.code()
        ))
    }
}
