//! Integration tests for `brainclaw::onboard`.

use std::collections::HashMap;

use brainclaw::BrainClawConfig;
use brainclaw::onboard::{self, NonInteractiveEnv};

#[test]
fn non_interactive_writes_parseable_config() {
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("brainclaw.toml");

    let env = NonInteractiveEnv::from_map(
        [("ANTHROPIC_API_KEY".to_string(), "sk-test".to_string())]
            .into_iter()
            .collect(),
    );

    let outcome = onboard::run_non_interactive(&target, &env, false)
        .expect("non-interactive run should succeed");

    assert_eq!(outcome.config_path, target);
    assert_eq!(outcome.admin_token.len(), 64); // 32 bytes hex

    // File exists and round-trips through the real config parser.
    let parsed = BrainClawConfig::load(&target).expect("load written config");
    assert_eq!(parsed.provider.default_provider, "anthropic");
    assert_eq!(
        parsed.provider.api_key_env.as_deref(),
        Some("ANTHROPIC_API_KEY")
    );
    assert!(
        parsed.provider.api_key.is_none(),
        "api_key must NOT be written to disk during non-interactive run"
    );
    assert_eq!(
        parsed.security.admin_token.as_deref(),
        Some(outcome.admin_token.as_str()),
        "admin token in the file must match the one printed to the user"
    );
}

#[test]
fn non_interactive_refuses_to_overwrite() {
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("brainclaw.toml");
    std::fs::write(&target, "# pre-existing").unwrap();

    let env = NonInteractiveEnv::from_map(HashMap::new());
    let res = onboard::run_non_interactive(&target, &env, /*force=*/ false);
    assert!(
        res.is_err(),
        "expected overwrite error, got Ok({:?})",
        res.ok().map(|o| o.config_path)
    );
    assert!(
        res.unwrap_err().to_string().contains("already exists"),
        "error message should mention the existing file"
    );
}

#[test]
fn non_interactive_force_overwrites() {
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("brainclaw.toml");
    std::fs::write(&target, "# placeholder").unwrap();
    let env = NonInteractiveEnv::from_map(HashMap::new());
    let outcome =
        onboard::run_non_interactive(&target, &env, true).expect("force should overwrite");
    assert_eq!(outcome.config_path, target);

    let parsed = BrainClawConfig::load(&target).unwrap();
    assert!(parsed.security.admin_token.is_some());
}

#[test]
fn picks_openai_when_only_openai_key_present() {
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("brainclaw.toml");
    let env = NonInteractiveEnv::from_map(
        [("OPENAI_API_KEY".to_string(), "sk-openai".to_string())]
            .into_iter()
            .collect(),
    );
    let _outcome = onboard::run_non_interactive(&target, &env, false).unwrap();
    let parsed = BrainClawConfig::load(&target).unwrap();
    assert_eq!(parsed.provider.default_provider, "openai");
    assert_eq!(
        parsed.provider.api_key_env.as_deref(),
        Some("OPENAI_API_KEY")
    );
}

#[test]
fn admin_tokens_differ_across_runs() {
    let t1 = onboard::generate_admin_token();
    let t2 = onboard::generate_admin_token();
    assert_ne!(t1, t2, "two generated tokens should not collide");
    assert_eq!(t1.len(), 64);
    assert_eq!(t2.len(), 64);
}
