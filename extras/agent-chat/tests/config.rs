//! Smoke tests for `agent-chat` configuration round-trips and the shared
//! slash-command parser used by the plain REPL and TUI.

use agent_chat::commands::{SlashCommand, parse_slash_command};
use agent_chat::config::ChatConfig;
use tempfile::TempDir;

#[test]
fn config_load_defaults_when_missing() {
    let dir = TempDir::new().expect("create tempdir");
    let path = dir.path().join("config.toml");

    // File does not exist yet — `load_from` must yield a default config
    // rather than erroring or creating anything on disk.
    let cfg = ChatConfig::load_from(&path).expect("load defaults");
    let defaults = ChatConfig::default();

    assert_eq!(cfg.default_provider, defaults.default_provider);
    assert_eq!(cfg.default_model, defaults.default_model);
    assert_eq!(cfg.permission_mode, defaults.permission_mode);
    assert_eq!(cfg.max_tokens, defaults.max_tokens);
    assert!(!path.exists(), "load_from must not write to disk");
}

#[test]
fn config_roundtrip() {
    let dir = TempDir::new().expect("create tempdir");
    let path = dir.path().join("nested").join("config.toml");

    let original = ChatConfig {
        default_provider: "openai".into(),
        default_model: "gpt-4o-mini".into(),
        system_prompt: Some("You are a terse assistant.".into()),
        permission_mode: "always".into(),
        max_tokens: 8192,
        temperature: 0.2,
    };

    original.save_to(&path).expect("save_to");
    assert!(path.exists(), "save_to should create the file");

    let reloaded = ChatConfig::load_from(&path).expect("load_from");
    assert_eq!(reloaded.default_provider, "openai");
    assert_eq!(reloaded.default_model, "gpt-4o-mini");
    assert_eq!(
        reloaded.system_prompt.as_deref(),
        Some("You are a terse assistant.")
    );
    assert_eq!(reloaded.permission_mode, "always");
    assert_eq!(reloaded.max_tokens, 8192);
    assert!((reloaded.temperature - 0.2).abs() < f32::EPSILON);
}

#[test]
fn parse_slash_command_known() {
    assert_eq!(parse_slash_command("/help"), Some(SlashCommand::Help));
    assert_eq!(parse_slash_command("/clear"), Some(SlashCommand::Clear));
    assert_eq!(parse_slash_command("/exit"), Some(SlashCommand::Exit));
    assert_eq!(parse_slash_command("/quit"), Some(SlashCommand::Exit));
    assert_eq!(
        parse_slash_command("/model claude-sonnet-4-20250514"),
        Some(SlashCommand::Model("claude-sonnet-4-20250514".into())),
    );
    // Leading/trailing whitespace is tolerated.
    assert_eq!(parse_slash_command("   /help  "), Some(SlashCommand::Help));
    // Non-slash input is not a slash command at all.
    assert_eq!(parse_slash_command("hello world"), None);
}

#[test]
fn parse_slash_command_unknown() {
    match parse_slash_command("/nope-not-a-real-command") {
        Some(SlashCommand::Unknown(name)) => {
            assert_eq!(name, "nope-not-a-real-command");
        }
        other => panic!("expected Unknown variant, got {other:?}"),
    }
}
