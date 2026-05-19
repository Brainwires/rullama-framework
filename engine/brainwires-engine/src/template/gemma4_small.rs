//! Gemma 4 chat template (small variant).
//!
//! Mirrors the active path of `model/renderers/gemma4.go` (gemma4-small) for a chat
//! made of `system` / `user` / `model` turns. Tool-call / thinking-channel rendering
//! is deferred — those don't matter for a simple PWA chat.
//!
//! Output is exactly what's tokenized end-to-end; tokens 105 (`<|turn>`),
//! 106 (`<turn|>`) and the role names (`user`, `model`, `system`) come out of our
//! tokenizer the same way Ollama's does (M1 golden tests).

use crate::api::{ChatMessage, ChatRole};

/// Render messages as a complete chat prompt ready for tokenization. The trailing
/// `<|turn>model\n` opens the assistant turn — the model continues from there.
///
/// `with_bos` prepends the BOS token marker (`<bos>`). Gemma 4's default is to NOT
/// add BOS automatically, so include it here when you want the conventional layout.
pub fn render_for_completion(messages: &[ChatMessage], with_bos: bool) -> String {
    let mut out = String::new();
    if with_bos {
        out.push_str("<bos>");
    }
    for msg in messages {
        let role = role_name(msg.role);
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str("<turn|>\n");
    }
    // Open the assistant turn so the model generates inside it.
    out.push_str("<|turn>model\n");
    out
}

/// Render the `<turn|>` close marker that ends an assistant reply. Used when the
/// caller needs to stitch a sampled reply back into the message history (e.g. for
/// a multi-turn conversation).
pub fn end_of_turn() -> &'static str {
    "<turn|>\n"
}

/// Render messages for *continuation* of an interrupted assistant turn.
///
/// If the last message is `role: Model`, its content is rendered without
/// the trailing `<turn|>\n` close marker — leaving the assistant turn
/// open so the next `step()` continues generating *that* response
/// instead of starting a new one. All earlier messages render with
/// their normal close markers.
///
/// If the last message is NOT `Model` (e.g. just user history), this
/// behaves identically to [`render_for_completion`] — it opens a fresh
/// assistant turn.
///
/// Used by the suspend/resume path to rebuild KV cache from a
/// conversation that includes a partial assistant response.
pub fn render_for_continuation(messages: &[ChatMessage], with_bos: bool) -> String {
    let mut out = String::new();
    if with_bos {
        out.push_str("<bos>");
    }
    let last_idx = messages.len().saturating_sub(1);
    let last_is_model = messages
        .last()
        .map(|m| m.role == ChatRole::Model)
        .unwrap_or(false);

    for (idx, msg) in messages.iter().enumerate() {
        let role = role_name(msg.role);
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&msg.content);
        // Final model turn stays open so generation continues *inside* it.
        if !(idx == last_idx && last_is_model) {
            out.push_str("<turn|>\n");
        }
    }
    if !last_is_model {
        out.push_str("<|turn>model\n");
    }
    out
}

fn role_name(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Model => "model",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ChatMessage, ChatRole};

    #[test]
    fn renders_single_user_turn() {
        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hi".to_string(),
        }];
        let s = render_for_completion(&msgs, false);
        assert_eq!(s, "<|turn>user\nHi<turn|>\n<|turn>model\n");
    }

    #[test]
    fn renders_with_bos_and_system() {
        let msgs = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are friendly.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hi".to_string(),
            },
        ];
        let s = render_for_completion(&msgs, true);
        assert_eq!(
            s,
            "<bos><|turn>system\nYou are friendly.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn render_for_continuation_keeps_last_model_turn_open() {
        let msgs = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "Hi".to_string(),
            },
            ChatMessage {
                role: ChatRole::Model,
                content: "Hello! How can I he".to_string(),
            },
        ];
        let s = render_for_continuation(&msgs, false);
        assert_eq!(
            s,
            "<|turn>user\nHi<turn|>\n<|turn>model\nHello! How can I he"
        );
    }

    #[test]
    fn render_for_continuation_without_trailing_model_acts_like_completion() {
        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hi".to_string(),
        }];
        let s_cont = render_for_continuation(&msgs, false);
        let s_full = render_for_completion(&msgs, false);
        assert_eq!(s_cont, s_full);
    }

    #[test]
    fn render_for_continuation_with_bos_and_system_preserves_history_closes() {
        let msgs = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are friendly.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hi".to_string(),
            },
            ChatMessage {
                role: ChatRole::Model,
                content: "Hi! How can".to_string(),
            },
        ];
        let s = render_for_continuation(&msgs, true);
        assert_eq!(
            s,
            "<bos><|turn>system\nYou are friendly.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHi! How can"
        );
    }

    #[test]
    fn round_trip_through_tokenizer_with_real_gguf() {
        let path = "/Users/nightness/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: gemma4 GGUF not available");
            return;
        }
        let bytes = std::fs::read(path).unwrap();
        let r = crate::gguf::GgufReader::new(bytes).unwrap();
        let tok = crate::tokenizer::BpeTokenizer::from_gguf(&r).unwrap();
        let msgs = vec![ChatMessage {
            role: ChatRole::User,
            content: "Hi".to_string(),
        }];
        let s = render_for_completion(&msgs, false);
        let ids = tok.encode(&s);
        // Should match the canonical layout we used in M3 manually:
        //   "<|turn>user\nHi<turn|>\n<|turn>model\n"
        //   = [105, 2364, 107, 10979, 106, 107, 105, 4368, 107]
        assert_eq!(ids, vec![105, 2364, 107, 10979, 106, 107, 105, 4368, 107]);
    }
}
