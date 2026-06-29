//! IMAP client for reading email messages.

use anyhow::{Context, Result};
use async_imap::Session;
use tokio::net::TcpStream;
use tokio_native_tls::TlsStream;

use super::types::{EmailFolder, EmailMessage, EmailSearchQuery};

/// IMAP client wrapping an async-imap session over a TLS-encrypted TCP stream.
pub struct ImapClient {
    session: Session<TlsStream<TcpStream>>,
}

impl ImapClient {
    /// Connect and authenticate to an IMAP server with TLS.
    pub async fn connect(
        host: &str,
        port: u16,
        username: &str,
        password: &str,
        tls: bool,
    ) -> Result<Self> {
        if !tls {
            anyhow::bail!("Non-TLS IMAP connections are not supported; use TLS (port 993)");
        }

        let tcp = TcpStream::connect((host, port))
            .await
            .context("Failed to connect to IMAP server")?;

        let tls_connector = tokio_native_tls::TlsConnector::from(
            native_tls::TlsConnector::new().context("Failed to create TLS connector")?,
        );
        let tls_stream = tls_connector
            .connect(host, tcp)
            .await
            .context("TLS handshake failed")?;

        let client = async_imap::Client::new(tls_stream);
        let session = client
            .login(username, password)
            .await
            .map_err(|(e, _)| e)
            .context("IMAP login failed")?;

        Ok(Self { session })
    }

    /// List message summaries from a folder.
    pub async fn list_messages(
        &mut self,
        folder: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EmailMessage>> {
        self.session
            .select(folder)
            .await
            .context("Failed to select folder")?;

        let start = offset.saturating_add(1);
        let end = offset.saturating_add(limit);
        let range = format!("{}:{}", start, end);

        let messages_stream = self
            .session
            .fetch(&range, "(UID FLAGS ENVELOPE)")
            .await
            .context("Failed to fetch messages")?;

        let mut result = Vec::new();
        let messages: Vec<_> = {
            use futures::TryStreamExt;
            messages_stream
                .try_collect()
                .await
                .context("Failed to collect messages")?
        };

        for msg in &messages {
            let envelope = msg.envelope();
            let uid = msg.uid;
            let flags: Vec<String> = msg.flags().map(|f| format!("{:?}", f)).collect();

            let (from, subject, date) = if let Some(env) = envelope {
                let from = env
                    .from
                    .as_ref()
                    .and_then(|addrs| addrs.first())
                    .map(|a| {
                        let mailbox = a
                            .mailbox
                            .as_ref()
                            .map(|m| String::from_utf8_lossy(m).to_string())
                            .unwrap_or_default();
                        let host = a
                            .host
                            .as_ref()
                            .map(|h| String::from_utf8_lossy(h).to_string())
                            .unwrap_or_default();
                        format!("{}@{}", mailbox, host)
                    })
                    .unwrap_or_default();
                let subject = env
                    .subject
                    .as_ref()
                    .map(|s| String::from_utf8_lossy(s).to_string())
                    .unwrap_or_default();
                let date = env
                    .date
                    .as_ref()
                    .map(|d| String::from_utf8_lossy(d).to_string());
                (from, subject, date)
            } else {
                (String::new(), String::new(), None)
            };

            result.push(EmailMessage {
                from,
                to: vec![],
                cc: vec![],
                bcc: vec![],
                subject,
                body: None,
                body_html: None,
                attachments: vec![],
                date,
                uid,
                message_id: None,
                flags,
            });
        }

        Ok(result)
    }

    /// Read a full message by UID, including body and attachments.
    pub async fn read_message(&mut self, uid: u32) -> Result<EmailMessage> {
        let messages_stream = self
            .session
            .uid_fetch(uid.to_string(), "(UID FLAGS ENVELOPE BODY[])")
            .await
            .context("Failed to fetch message")?;

        let messages: Vec<_> = {
            use futures::TryStreamExt;
            messages_stream
                .try_collect()
                .await
                .context("Failed to collect message")?
        };

        let msg = messages
            .first()
            .ok_or_else(|| anyhow::anyhow!("Message UID {} not found", uid))?;

        let body_bytes = msg.body().unwrap_or_default();
        let parsed = mailparse::parse_mail(body_bytes).context("Failed to parse message body")?;

        let mut body_text = None;
        let mut body_html = None;
        let mut attachments = Vec::new();

        Self::extract_parts(&parsed, &mut body_text, &mut body_html, &mut attachments);

        let envelope = msg.envelope();
        let (from, to, cc, subject, date, message_id) = if let Some(env) = envelope {
            let from = Self::format_first_address(env.from.as_deref());
            let to = Self::format_address_list(env.to.as_deref());
            let cc = Self::format_address_list(env.cc.as_deref());
            let subject = env
                .subject
                .as_ref()
                .map(|s| String::from_utf8_lossy(s).to_string())
                .unwrap_or_default();
            let date = env
                .date
                .as_ref()
                .map(|d| String::from_utf8_lossy(d).to_string());
            let message_id = env
                .message_id
                .as_ref()
                .map(|m| String::from_utf8_lossy(m).to_string());
            (from, to, cc, subject, date, message_id)
        } else {
            (String::new(), vec![], vec![], String::new(), None, None)
        };

        let flags: Vec<String> = msg.flags().map(|f| format!("{:?}", f)).collect();

        Ok(EmailMessage {
            from,
            to,
            cc,
            bcc: vec![],
            subject,
            body: body_text,
            body_html,
            attachments,
            date,
            uid: msg.uid,
            message_id,
            flags,
        })
    }

    /// Search messages in a folder using IMAP search criteria.
    pub async fn search_messages(
        &mut self,
        query: &EmailSearchQuery,
        folder: &str,
    ) -> Result<Vec<u32>> {
        self.session
            .select(folder)
            .await
            .context("Failed to select folder")?;

        let mut criteria = Vec::new();
        if let Some(ref from) = query.from {
            criteria.push(format!("FROM \"{}\"", from));
        }
        if let Some(ref to) = query.to {
            criteria.push(format!("TO \"{}\"", to));
        }
        if let Some(ref subject) = query.subject {
            criteria.push(format!("SUBJECT \"{}\"", subject));
        }
        if let Some(ref body) = query.body {
            criteria.push(format!("BODY \"{}\"", body));
        }
        if let Some(ref since) = query.since {
            criteria.push(format!("SINCE \"{}\"", since));
        }
        if let Some(ref before) = query.before {
            criteria.push(format!("BEFORE \"{}\"", before));
        }
        for flag in &query.flags {
            criteria.push(format!("KEYWORD {}", flag));
        }

        let search_str = if criteria.is_empty() {
            "ALL".to_string()
        } else {
            criteria.join(" ")
        };

        let uids = self
            .session
            .uid_search(&search_str)
            .await
            .context("IMAP search failed")?;

        Ok(uids.into_iter().collect())
    }

    /// List available IMAP folders (mailboxes).
    #[allow(dead_code)] // reason: public API, currently unused internally.
    pub async fn list_folders(&mut self) -> Result<Vec<EmailFolder>> {
        let names_stream = self
            .session
            .list(None, Some("*"))
            .await
            .context("Failed to list folders")?;

        let names: Vec<_> = {
            use futures::TryStreamExt;
            names_stream
                .try_collect()
                .await
                .context("Failed to collect folder list")?
        };

        let mut folders = Vec::new();
        for name in &names {
            let folder_name = name.name().to_string();
            if let Ok(mailbox) = self.session.examine(&folder_name).await {
                // `unseen` is the sequence number of the first unseen message,
                // not the count. We approximate unread as (exists - unseen + 1)
                // when available, but this is imprecise. For accurate counts,
                // a STATUS command would be needed.
                let unread = mailbox
                    .unseen
                    .map(|seq| mailbox.exists.saturating_sub(seq).saturating_add(1))
                    .unwrap_or(0);
                folders.push(EmailFolder {
                    name: folder_name,
                    total_messages: mailbox.exists,
                    unread,
                });
            }
        }
        Ok(folders)
    }

    /// Gracefully close the IMAP session.
    pub async fn logout(mut self) -> Result<()> {
        self.session.logout().await.context("IMAP logout failed")?;
        Ok(())
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn extract_parts(
        parsed: &mailparse::ParsedMail<'_>,
        body_text: &mut Option<String>,
        body_html: &mut Option<String>,
        attachments: &mut Vec<super::types::EmailAttachment>,
    ) {
        let content_type = parsed.ctype.mimetype.to_lowercase();
        let disposition = parsed
            .headers
            .iter()
            .find(|h| h.get_key().eq_ignore_ascii_case("content-disposition"))
            .map(|h| h.get_value());

        let is_attachment = disposition
            .as_ref()
            .is_some_and(|d| d.to_lowercase().starts_with("attachment"));

        if is_attachment {
            if let Ok(data) = parsed.get_body_raw() {
                let filename = parsed
                    .ctype
                    .params
                    .get("name")
                    .cloned()
                    .unwrap_or_else(|| "attachment".to_string());
                attachments.push(super::types::EmailAttachment {
                    filename,
                    content_type: content_type.clone(),
                    data,
                });
            }
        } else if content_type == "text/plain" && body_text.is_none() {
            *body_text = parsed.get_body().ok();
        } else if content_type == "text/html" && body_html.is_none() {
            *body_html = parsed.get_body().ok();
        }

        for part in &parsed.subparts {
            Self::extract_parts(part, body_text, body_html, attachments);
        }
    }

    fn format_first_address(addrs: Option<&[imap_proto::Address<'_>]>) -> String {
        addrs
            .and_then(|a| a.first())
            .map(|a| {
                let mailbox = a
                    .mailbox
                    .as_ref()
                    .map(|m| String::from_utf8_lossy(m).to_string())
                    .unwrap_or_default();
                let host = a
                    .host
                    .as_ref()
                    .map(|h| String::from_utf8_lossy(h).to_string())
                    .unwrap_or_default();
                format!("{}@{}", mailbox, host)
            })
            .unwrap_or_default()
    }

    fn format_address_list(addrs: Option<&[imap_proto::Address<'_>]>) -> Vec<String> {
        addrs
            .unwrap_or_default()
            .iter()
            .map(|a| {
                let mailbox = a
                    .mailbox
                    .as_ref()
                    .map(|m| String::from_utf8_lossy(m).to_string())
                    .unwrap_or_default();
                let host = a
                    .host
                    .as_ref()
                    .map(|h| String::from_utf8_lossy(h).to_string())
                    .unwrap_or_default();
                format!("{}@{}", mailbox, host)
            })
            .collect()
    }
}
