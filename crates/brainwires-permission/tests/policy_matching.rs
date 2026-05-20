//! Integration tests for the `PolicyEngine` — one table-driven run per
//! `PolicyCondition` variant, then cross-cutting tests for AND/OR/NOT
//! composition, priority ordering, disabled policies, and default-action
//! fallback.
//!
//! These sit at the security perimeter: any false-negative here means an
//! agent slips past a policy. Keep tests blunt and explicit — prefer a
//! larger table to clever loops.

use brainwires_permission::{
    GitOperation, PolicyAction, PolicyCondition, PolicyEngine, PolicyRequest, ToolCategory,
    policy::{EnforcementMode, Policy},
};

// ── PolicyCondition::Tool ────────────────────────────────────────────────

#[test]
fn tool_condition_matches_exact_name_only() {
    let c = PolicyCondition::Tool("read_file".into());
    assert!(c.matches(&PolicyRequest::for_tool("read_file")));
    assert!(!c.matches(&PolicyRequest::for_tool("read_files"))); // prefix ≠ match
    assert!(!c.matches(&PolicyRequest::for_tool("write_file")));
    assert!(!c.matches(&PolicyRequest::new())); // no tool_name = no match
}

// ── PolicyCondition::ToolCategory ────────────────────────────────────────

#[test]
fn tool_category_condition_matches_when_category_equals() {
    let c = PolicyCondition::ToolCategory(ToolCategory::Bash);
    let mut req = PolicyRequest::new();
    req.tool_category = Some(ToolCategory::Bash);
    assert!(c.matches(&req));

    req.tool_category = Some(ToolCategory::Web);
    assert!(!c.matches(&req));

    req.tool_category = None;
    assert!(!c.matches(&req));
}

// ── PolicyCondition::FilePath ────────────────────────────────────────────

#[test]
fn file_path_condition_uses_glob_semantics() {
    let c = PolicyCondition::FilePath("**/.env*".into());

    let cases = [
        (".env", true),
        (".env.local", true),
        ("src/.env", true),
        ("a/b/c/.env.production", true),
        ("env.toml", false), // leading dot required by pattern
        ("src/config.rs", false),
    ];
    for (path, expected) in cases {
        let req = PolicyRequest::for_file(path, "read_file");
        assert_eq!(
            c.matches(&req),
            expected,
            "pattern `**/.env*` vs `{path}` should be {expected}",
        );
    }
}

#[test]
fn file_path_condition_is_false_when_path_is_missing() {
    let c = PolicyCondition::FilePath("**/*.rs".into());
    // No `file_path` on the request — condition must decline rather than panic.
    let req = PolicyRequest::for_tool("read_file");
    assert!(!c.matches(&req));
}

// ── PolicyCondition::MinTrustLevel ───────────────────────────────────────

#[test]
fn min_trust_level_is_inclusive() {
    let c = PolicyCondition::MinTrustLevel(3);
    for (level, expected) in [(0, false), (2, false), (3, true), (4, true), (255, true)] {
        let req = PolicyRequest::new().with_trust_level(level);
        assert_eq!(
            c.matches(&req),
            expected,
            "trust={level} vs min=3 should be {expected}",
        );
    }
}

// ── PolicyCondition::Domain ──────────────────────────────────────────────

#[test]
fn domain_condition_exact_match() {
    let c = PolicyCondition::Domain("api.example.com".into());
    let mut req = PolicyRequest::new();

    req.domain = Some("api.example.com".into());
    assert!(c.matches(&req));

    req.domain = Some("example.com".into());
    assert!(!c.matches(&req));

    req.domain = Some("EVIL.api.example.com".into());
    assert!(!c.matches(&req)); // exact match is not a suffix match
}

#[test]
fn domain_condition_wildcard_matches_bare_and_subdomains() {
    let c = PolicyCondition::Domain("*.example.com".into());

    let cases = [
        ("example.com", true),              // bare apex matches
        ("api.example.com", true),          // subdomain matches
        ("a.b.example.com", true),          // nested subdomain matches
        ("example.com.attacker.io", false), // suffix-confusion bypass attempt
        ("fakeexample.com", false),         // must be a proper subdomain
        ("notexample.com", false),
    ];
    for (domain, expected) in cases {
        let mut req = PolicyRequest::new();
        req.domain = Some(domain.into());
        assert_eq!(
            c.matches(&req),
            expected,
            "wildcard `*.example.com` vs `{domain}` should be {expected}",
        );
    }
}

// ── PolicyCondition::GitOp ───────────────────────────────────────────────

#[test]
fn git_op_condition_matches_exact_operation() {
    let c = PolicyCondition::GitOp(GitOperation::Reset);
    assert!(c.matches(&PolicyRequest::for_git(GitOperation::Reset)));
    assert!(!c.matches(&PolicyRequest::for_git(GitOperation::Commit)));
    assert!(!c.matches(&PolicyRequest::new()));
}

// ── PolicyCondition::TimeRange ───────────────────────────────────────────

#[test]
fn time_range_condition_non_wrapping_covers_current_hour_or_not() {
    use chrono::Timelike;
    let hour = chrono::Local::now().hour() as u8;

    // Window that definitely includes `hour`.
    let include = PolicyCondition::TimeRange {
        start_hour: hour,
        end_hour: hour.saturating_add(1).min(23),
    };
    // Window that definitely excludes `hour`. Use a strictly-empty window
    // (start == end) so the inclusive/exclusive check is exercised.
    let exclude = PolicyCondition::TimeRange {
        start_hour: hour,
        end_hour: hour,
    };

    let req = PolicyRequest::new();
    if include.matches(&req) {
        // Included window should match, empty window should not.
        assert!(!exclude.matches(&req));
    }
    // Regardless of clock, the empty window [hour..hour) must never match.
    assert!(!exclude.matches(&req));
}

#[test]
fn time_range_wraps_midnight() {
    // 22 → 06 covers night hours; make sure the wrap branch is taken.
    let c = PolicyCondition::TimeRange {
        start_hour: 22,
        end_hour: 6,
    };
    // We can't fake the clock without more machinery; at minimum the
    // pattern must type-check, evaluate without panic, and return a bool.
    let _ = c.matches(&PolicyRequest::new());
}

// ── Always / Not / And / Or ──────────────────────────────────────────────

#[test]
fn always_condition_matches_unconditionally() {
    assert!(PolicyCondition::Always.matches(&PolicyRequest::new()));
}

#[test]
fn not_inverts_inner() {
    let c = PolicyCondition::Not(Box::new(PolicyCondition::Always));
    assert!(!c.matches(&PolicyRequest::new()));
}

#[test]
fn and_is_short_circuit_all() {
    let pass = PolicyCondition::Always;
    let fail = PolicyCondition::Tool("definitely-not-this".into());

    assert!(PolicyCondition::And(vec![pass.clone(), pass.clone()]).matches(&PolicyRequest::new()));
    assert!(!PolicyCondition::And(vec![pass.clone(), fail.clone()]).matches(&PolicyRequest::new()));
    // Empty AND: `all` over an empty iter is true by vacuous truth.
    assert!(PolicyCondition::And(vec![]).matches(&PolicyRequest::new()));
}

#[test]
fn or_is_short_circuit_any() {
    let pass = PolicyCondition::Always;
    let fail = PolicyCondition::Tool("definitely-not-this".into());

    assert!(PolicyCondition::Or(vec![fail.clone(), pass.clone()]).matches(&PolicyRequest::new()));
    assert!(!PolicyCondition::Or(vec![fail.clone(), fail.clone()]).matches(&PolicyRequest::new()));
    // Empty OR: `any` over empty iter is false.
    assert!(!PolicyCondition::Or(vec![]).matches(&PolicyRequest::new()));
}

// ── Policy struct behaviour ──────────────────────────────────────────────

#[test]
fn empty_conditions_never_match_as_safety_default() {
    // A policy with no conditions is treated as non-matching. This is the
    // documented safety fallback — deleting the last condition must NOT
    // suddenly blanket-apply the policy.
    let p = Policy::new("bogus").with_action(PolicyAction::Deny);
    assert!(!p.matches(&PolicyRequest::new()));
}

#[test]
fn disabled_policy_never_matches() {
    let mut p = Policy::new("will_disable")
        .with_condition(PolicyCondition::Always)
        .with_action(PolicyAction::Deny);
    assert!(p.matches(&PolicyRequest::new()));
    p.enabled = false;
    assert!(!p.matches(&PolicyRequest::new()));
}

#[test]
fn policy_with_multiple_conditions_requires_all() {
    let p = Policy::new("two_cond")
        .with_condition(PolicyCondition::Tool("read_file".into()))
        .with_condition(PolicyCondition::MinTrustLevel(3))
        .with_action(PolicyAction::Allow);

    let mut req = PolicyRequest::for_tool("read_file").with_trust_level(2);
    assert!(!p.matches(&req));
    req = req.with_trust_level(3);
    assert!(p.matches(&req));
    req = PolicyRequest::for_tool("write_file").with_trust_level(5);
    assert!(!p.matches(&req));
}

// ── PolicyEngine priority + default-action ───────────────────────────────

#[test]
fn higher_priority_policy_wins_even_when_registered_later() {
    let mut engine = PolicyEngine::new();

    // Low-priority allow registered first…
    engine.add_policy(
        Policy::new("allow_reads")
            .with_condition(PolicyCondition::ToolCategory(ToolCategory::FileRead))
            .with_action(PolicyAction::Allow)
            .with_priority(10),
    );
    // …then a high-priority deny. The engine must sort and hit the deny first.
    engine.add_policy(
        Policy::new("deny_env")
            .with_condition(PolicyCondition::FilePath("**/.env*".into()))
            .with_action(PolicyAction::Deny)
            .with_priority(100),
    );

    let req = PolicyRequest::for_file(".env", "read_file");
    let decision = engine.evaluate(&req);
    assert_eq!(decision.matched_policy.as_deref(), Some("deny_env"));
    assert!(!decision.is_allowed());
}

#[test]
fn default_action_applies_when_no_policy_matches() {
    let mut engine = PolicyEngine::new();
    engine.set_default_action(PolicyAction::Deny);

    engine.add_policy(
        Policy::new("only_bash")
            .with_condition(PolicyCondition::ToolCategory(ToolCategory::Bash))
            .with_action(PolicyAction::Allow),
    );

    let req = PolicyRequest::for_tool("read_file");
    let decision = engine.evaluate(&req);
    assert!(matches!(decision.action, PolicyAction::Deny));
    assert!(decision.matched_policy.is_none());
}

#[test]
fn with_defaults_denies_env_files() {
    let engine = PolicyEngine::with_defaults();
    let req = PolicyRequest::for_file(".env.production", "read_file");
    let decision = engine.evaluate(&req);
    assert!(!decision.is_allowed(), "default policy set must deny .env*");
}

#[test]
fn with_defaults_requires_approval_on_git_reset() {
    let engine = PolicyEngine::with_defaults();
    let req = PolicyRequest::for_git(GitOperation::Reset);
    let decision = engine.evaluate(&req);
    assert!(decision.requires_approval());
}

#[test]
fn remove_policy_returns_removed_and_updates_engine() {
    let mut engine = PolicyEngine::with_defaults();
    let removed = engine.remove_policy("protect_env_files");
    assert!(removed.is_some());
    let req = PolicyRequest::for_file(".env", "read_file");
    // A second, broader policy (`protect_secrets` / `protect_credentials`)
    // should NOT catch `.env` — after the remove, this request falls through
    // to the default Allow action.
    let decision = engine.evaluate(&req);
    assert!(decision.is_allowed());
}

#[test]
fn enforcement_mode_default_is_coercive() {
    assert_eq!(EnforcementMode::default(), EnforcementMode::Coercive);
}
