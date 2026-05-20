//! IRC protocol helpers that don't depend on the live client.
//!
//! Split out so the happy-path can be unit-tested without a real socket.

use brainwires_network::channels::{ChannelMessage, ConversationId, MessageContent, MessageId};
use chrono::Utc;
use std::collections::HashMap;

/// IRC line limit is 512 bytes *including* the protocol wrapper. We cap
/// the payload at 400 bytes to stay well inside the limit after the
/// `PRIVMSG <target> :` prefix and the trailing `\r\n`.
pub const MAX_PRIVMSG_BYTES: usize = 400;

/// What an incoming IRC PRIVMSG boils down to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InboundMessage {
    /// A channel message (target starts with `#`, `&`, `+`, or `!`).
    Channel {
        /// Channel name, e.g. `#chat`.
        channel: String,
        /// Sender nick.
        nick: String,
        /// Message body (with any prefix already stripped).
        body: String,
        /// Was this a CTCP ACTION (`/me`) frame?
        is_action: bool,
    },
    /// A private message targeted at the bot.
    Private {
        /// Sender nick.
        nick: String,
        /// Message body.
        body: String,
        /// Was this a CTCP ACTION?
        is_action: bool,
    },
    /// Not a PRIVMSG — callers typically ignore.
    Ignored,
}

/// Parse a raw PRIVMSG target/body pair.
///
/// `bot_nick` is compared case-insensitively to `target` to detect PMs.
/// `prefix` is the string that must appear at the start of a channel
/// message for us to forward it — PMs bypass the prefix check.
pub fn classify_privmsg(
    target: &str,
    body: &str,
    sender_nick: &str,
    bot_nick: &str,
    channel_prefix: &str,
) -> InboundMessage {
    let (stripped, is_action) = strip_ctcp_action(body);
    let is_channel_target = target
        .chars()
        .next()
        .map(|c| matches!(c, '#' | '&' | '+' | '!'))
        .unwrap_or(false);

    if !is_channel_target && target.eq_ignore_ascii_case(bot_nick) {
        return InboundMessage::Private {
            nick: sender_nick.to_string(),
            body: stripped.to_string(),
            is_action,
        };
    }

    if is_channel_target {
        // Prefix filter — public channels only forward if the message
        // starts with the prefix, which we strip from the body before
        // forwarding.
        if channel_prefix.is_empty() {
            return InboundMessage::Channel {
                channel: target.to_string(),
                nick: sender_nick.to_string(),
                body: stripped.to_string(),
                is_action,
            };
        }
        if let Some(remainder) = stripped.strip_prefix(channel_prefix) {
            return InboundMessage::Channel {
                channel: target.to_string(),
                nick: sender_nick.to_string(),
                body: remainder.trim_start().to_string(),
                is_action,
            };
        }
    }

    InboundMessage::Ignored
}

/// Detect and strip a CTCP ACTION wrapper.
///
/// CTCP ACTION frames are `\x01ACTION <text>\x01`. Returns `(inner, true)`
/// when the wrapper is present, `(raw, false)` otherwise.
pub fn strip_ctcp_action(body: &str) -> (&str, bool) {
    let bytes = body.as_bytes();
    if bytes.len() >= 9
        && bytes[0] == 0x01
        && body.starts_with("\u{0001}ACTION ")
        && bytes.last() == Some(&0x01)
    {
        let inner = &body["\u{0001}ACTION ".len()..body.len() - 1];
        return (inner, true);
    }
    (body, false)
}

/// Wrap an outbound "action" string with the CTCP ACTION envelope.
pub fn build_ctcp_action(text: &str) -> String {
    format!("\u{0001}ACTION {}\u{0001}", text)
}

/// Split an outbound message into UTF-8-safe chunks sized for IRC.
///
/// Always returns at least one chunk (possibly empty if `text` is empty).
pub fn chunk_for_privmsg(text: &str, max_bytes: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for line in text.split('\n') {
        // If a single line is too long, split on char boundaries.
        if line.len() > max_bytes {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            let mut start = 0;
            while start < line.len() {
                // Walk forward up to max_bytes without cutting a UTF-8
                // codepoint. `floor_char_boundary` is nightly, so we do
                // the equivalent explicitly.
                let end = if start + max_bytes >= line.len() {
                    line.len()
                } else {
                    let mut e = start + max_bytes;
                    while e > start && !line.is_char_boundary(e) {
                        e -= 1;
                    }
                    e
                };
                out.push(line[start..end].to_string());
                start = end;
            }
            continue;
        }
        // Can this line fit into the current chunk?
        if current.is_empty() {
            current = line.to_string();
        } else if current.len() + 1 + line.len() <= max_bytes {
            current.push('\n');
            current.push_str(line);
        } else {
            out.push(std::mem::take(&mut current));
            current = line.to_string();
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Render an [`InboundMessage`] as a BrainClaw [`ChannelMessage`] bound
/// to the given IRC server.
pub fn to_channel_message(server: &str, msg: &InboundMessage) -> Option<ChannelMessage> {
    let (channel_id, nick, body, is_action) = match msg {
        InboundMessage::Channel {
            channel,
            nick,
            body,
            is_action,
        } => (channel.clone(), nick.clone(), body.clone(), *is_action),
        InboundMessage::Private {
            nick,
            body,
            is_action,
        } => (format!("pm:{nick}"), nick.clone(), body.clone(), *is_action),
        InboundMessage::Ignored => return None,
    };

    let display = if is_action {
        format!("*{nick} {body}*")
    } else {
        body
    };

    let mut metadata = HashMap::new();
    metadata.insert("irc.server".into(), server.to_string());
    if is_action {
        metadata.insert("irc.action".into(), "true".into());
    }

    Some(ChannelMessage {
        id: MessageId::new(uuid::Uuid::new_v4().to_string()),
        conversation: ConversationId {
            platform: "irc".into(),
            channel_id,
            server_id: Some(server.to_string()),
        },
        author: nick,
        content: MessageContent::Text(display),
        thread_id: None,
        reply_to: None,
        timestamp: Utc::now(),
        attachments: vec![],
        metadata,
    })
}

/// Build a session identifier the gateway can use as a key.
pub fn session_id(server: &str, target: &str) -> String {
    format!("irc:{server}:{target}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_channel_requires_prefix() {
        let m = classify_privmsg("#test", "hi there", "alice", "bot", "brainclaw: ");
        assert_eq!(m, InboundMessage::Ignored);

        let m = classify_privmsg(
            "#test",
            "brainclaw: tell me a joke",
            "alice",
            "bot",
            "brainclaw: ",
        );
        match m {
            InboundMessage::Channel { body, channel, .. } => {
                assert_eq!(channel, "#test");
                assert_eq!(body, "tell me a joke");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn classify_pm_forwards_unconditionally() {
        let m = classify_privmsg("bot", "anything", "alice", "bot", "brainclaw: ");
        match m {
            InboundMessage::Private { nick, body, .. } => {
                assert_eq!(nick, "alice");
                assert_eq!(body, "anything");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn classify_pm_case_insensitive_nick() {
        let m = classify_privmsg("Bot", "hello", "alice", "bot", "brainclaw: ");
        assert!(matches!(m, InboundMessage::Private { .. }));
    }

    #[test]
    fn classify_unknown_target_is_ignored() {
        // PRIVMSG target is some other user — not the bot, not a channel.
        let m = classify_privmsg("someone_else", "hi", "alice", "bot", "brainclaw: ");
        assert_eq!(m, InboundMessage::Ignored);
    }

    #[test]
    fn empty_prefix_means_forward_everything() {
        let m = classify_privmsg("#test", "just chatting", "alice", "bot", "");
        match m {
            InboundMessage::Channel { body, .. } => assert_eq!(body, "just chatting"),
            _ => panic!(),
        }
    }

    #[test]
    fn strip_ctcp_action_detects_me() {
        let raw = "\u{0001}ACTION waves\u{0001}";
        let (inner, is_action) = strip_ctcp_action(raw);
        assert!(is_action);
        assert_eq!(inner, "waves");
    }

    #[test]
    fn strip_ctcp_action_passes_plain_through() {
        let (inner, is_action) = strip_ctcp_action("plain");
        assert!(!is_action);
        assert_eq!(inner, "plain");
    }

    #[test]
    fn build_ctcp_action_wraps() {
        let out = build_ctcp_action("jumps");
        assert_eq!(out, "\u{0001}ACTION jumps\u{0001}");
    }

    #[test]
    fn chunk_for_privmsg_keeps_short_messages_intact() {
        let out = chunk_for_privmsg("hello world", MAX_PRIVMSG_BYTES);
        assert_eq!(out, vec!["hello world"]);
    }

    #[test]
    fn chunk_for_privmsg_splits_on_newlines_first() {
        let text = "line1\nline2";
        let out = chunk_for_privmsg(text, 5);
        assert_eq!(out, vec!["line1", "line2"]);
    }

    #[test]
    fn chunk_for_privmsg_respects_utf8_boundaries() {
        // 10 bytes of a 2-byte codepoint — splitting mid-codepoint would
        // panic. With limit 5 we should still get valid UTF-8 chunks.
        let text = "ééééé"; // 5 × 2 = 10 bytes.
        let chunks = chunk_for_privmsg(text, 5);
        assert!(chunks.iter().all(|c| c.is_char_boundary(c.len())));
        let rejoined: String = chunks.join("");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn chunk_for_privmsg_empty_input_returns_one_empty_chunk() {
        assert_eq!(chunk_for_privmsg("", 100), vec![String::new()]);
    }

    #[test]
    fn chunk_for_privmsg_single_long_line() {
        let text = "a".repeat(1000);
        let chunks = chunk_for_privmsg(&text, 400);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| c.len() <= 400));
    }

    #[test]
    fn to_channel_message_for_channel_sets_ids() {
        let m = InboundMessage::Channel {
            channel: "#test".into(),
            nick: "alice".into(),
            body: "hi".into(),
            is_action: false,
        };
        let cm = to_channel_message("irc.libera.chat", &m).unwrap();
        assert_eq!(cm.conversation.platform, "irc");
        assert_eq!(cm.conversation.channel_id, "#test");
        assert_eq!(
            cm.conversation.server_id.as_deref(),
            Some("irc.libera.chat")
        );
        assert_eq!(cm.author, "alice");
        match cm.content {
            MessageContent::Text(t) => assert_eq!(t, "hi"),
            _ => panic!(),
        }
    }

    #[test]
    fn to_channel_message_action_prettifies() {
        let m = InboundMessage::Channel {
            channel: "#test".into(),
            nick: "alice".into(),
            body: "waves".into(),
            is_action: true,
        };
        let cm = to_channel_message("irc.example.net", &m).unwrap();
        match cm.content {
            MessageContent::Text(t) => assert_eq!(t, "*alice waves*"),
            _ => panic!(),
        }
        assert_eq!(
            cm.metadata.get("irc.action").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn to_channel_message_pm_uses_nick_in_channel_id() {
        let m = InboundMessage::Private {
            nick: "alice".into(),
            body: "hi".into(),
            is_action: false,
        };
        let cm = to_channel_message("irc.example.net", &m).unwrap();
        assert_eq!(cm.conversation.channel_id, "pm:alice");
    }

    #[test]
    fn session_id_is_stable() {
        assert_eq!(
            session_id("irc.libera.chat", "#test"),
            "irc:irc.libera.chat:#test"
        );
    }
}
