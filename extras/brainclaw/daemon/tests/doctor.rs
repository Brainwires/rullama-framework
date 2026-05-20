//! Integration tests for `brainclaw::doctor`.
//!
//! These do NOT hit the network. The provider-auth check is exercised in
//! its "no API key" fail path only — a network-backed happy path would
//! be flaky in CI.

use brainclaw::BrainClawConfig;
use brainclaw::doctor::{self, Status};

fn count(results: &[doctor::CheckResult], s: Status) -> usize {
    results.iter().filter(|r| r.status == s).count()
}

fn find<'a>(results: &'a [doctor::CheckResult], name: &str) -> Option<&'a doctor::CheckResult> {
    results.iter().find(|r| r.name == name)
}

/// Build a config that won't accidentally depend on the ambient
/// filesystem: point storage dirs at a tempdir and disable sandbox so
/// we don't talk to Docker.
fn minimal_config(tmp: &std::path::Path) -> BrainClawConfig {
    let mut cfg = BrainClawConfig::default();
    cfg.sandbox.enabled = false;
    cfg.memory.enabled = true;
    cfg.memory.storage_dir = tmp.join("memory").to_string_lossy().into_owned();
    cfg.pairing.store_path = Some(tmp.join("pairing.json").to_string_lossy().into_owned());
    cfg.skills.enabled = false;
    cfg.skills.directories.clear();
    cfg.skills.registry_url = None;
    // Pick a high random port that's extremely unlikely to be in use.
    cfg.gateway.port = 58637;
    cfg
}

#[tokio::test]
async fn fail_when_provider_key_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = minimal_config(tmp.path());
    cfg.provider.default_provider = "anthropic".into();
    cfg.provider.api_key = None;
    cfg.provider.api_key_env = Some("__BRAINCLAW_DOCTOR_TEST_UNSET_VAR__".into());
    // Ensure the ambient ANTHROPIC_API_KEY isn't set — skip the test
    // when it IS set, since we can't safely remove env vars in parallel
    // tests without data races.
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        eprintln!("skipping: ANTHROPIC_API_KEY is set in the environment");
        return;
    }

    let results = doctor::run_doctor_with_config(&cfg).await;
    let provider_auth = find(&results, "provider-auth").expect("provider-auth result");
    assert_eq!(
        provider_auth.status,
        Status::Fail,
        "expected Fail when no key is resolvable, got {:?}: {}",
        provider_auth.status,
        provider_auth.detail
    );
}

#[tokio::test]
async fn channel_check_skips_when_no_channels_listed() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = minimal_config(tmp.path());
    // security.allowed_channel_types defaults to empty.

    let results = doctor::run_doctor_with_config(&cfg).await;
    // Exactly one "channel-credentials" skip row is produced.
    let channel_rows: Vec<_> = results
        .iter()
        .filter(|r| r.name == "channel-credentials")
        .collect();
    assert_eq!(channel_rows.len(), 1);
    assert_eq!(channel_rows[0].status, Status::Skip);
}

#[tokio::test]
async fn memory_dir_under_tempdir_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = minimal_config(tmp.path());
    // The storage dir doesn't exist yet; the parent (the tempdir) does
    // and is writable, so memory should Pass.
    let results = doctor::run_doctor_with_config(&cfg).await;
    let mem = find(&results, "memory").expect("memory row");
    assert_eq!(
        mem.status,
        Status::Pass,
        "expected Pass for a tempdir-backed memory path, got {:?}: {}",
        mem.status,
        mem.detail
    );
}

#[tokio::test]
async fn sandbox_disabled_yields_skip() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = minimal_config(tmp.path());
    let results = doctor::run_doctor_with_config(&cfg).await;
    let sb = find(&results, "sandbox").expect("sandbox skip row");
    assert_eq!(sb.status, Status::Skip);
}

#[tokio::test]
async fn overall_shape_makes_sense() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = minimal_config(tmp.path());
    let results = doctor::run_doctor_with_config(&cfg).await;

    // At least one of each of Pass / Fail / Skip will almost certainly
    // show up; we just assert the counts sum to the total and that the
    // vector isn't empty.
    let total = results.len();
    assert!(total > 0, "doctor produced no results");
    let pass = count(&results, Status::Pass);
    let warn = count(&results, Status::Warn);
    let fail = count(&results, Status::Fail);
    let skip = count(&results, Status::Skip);
    assert_eq!(pass + warn + fail + skip, total);
}
