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
pub fn end_of_turn() -> &'static str { "<turn|>\n" }

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
        let msgs = vec![ChatMessage { role: ChatRole::User, content: "Hi".to_string() }];
        let s = render_for_completion(&msgs, false);
        assert_eq!(s, "<|turn>user\nHi<turn|>\n<|turn>model\n");
    }

    #[test]
    fn renders_with_bos_and_system() {
        let msgs = vec![
            ChatMessage { role: ChatRole::System, content: "You are friendly.".to_string() },
            ChatMessage { role: ChatRole::User,   content: "Hi".to_string() },
        ];
        let s = render_for_completion(&msgs, true);
        assert_eq!(
            s,
            "<bos><|turn>system\nYou are friendly.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
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
        let msgs = vec![ChatMessage { role: ChatRole::User, content: "Hi".to_string() }];
        let s = render_for_completion(&msgs, false);
        let ids = tok.encode(&s);
        // Should match the canonical layout we used in M3 manually:
        //   "<|turn>user\nHi<turn|>\n<|turn>model\n"
        //   = [105, 2364, 107, 10979, 106, 107, 105, 4368, 107]
        assert_eq!(ids, vec![105, 2364, 107, 10979, 106, 107, 105, 4368, 107]);
    }
}
