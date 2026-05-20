//! Property-based tests for `PolicyCondition::Domain` wildcard matching.
//!
//! Domain policies are a load-bearing security surface: a bypass here lets
//! outbound traffic reach an attacker-controlled host. The matching rule
//! (see `policy.rs`) is:
//!
//!   pattern `*.example.com` matches:
//!     - any host ending in `.example.com`
//!     - the bare apex `example.com`
//!   everything else must be rejected.
//!
//! These properties sweep randomized inputs to catch prefix-confusion and
//! suffix-confusion bypass attempts that a hand-written table might miss.

use brainwires_permission::{PolicyCondition, PolicyRequest};
use proptest::prelude::*;

fn matches_wildcard(pattern: &str, domain: &str) -> bool {
    let c = PolicyCondition::Domain(pattern.into());
    let mut req = PolicyRequest::new();
    req.domain = Some(domain.into());
    c.matches(&req)
}

// ── Properties ───────────────────────────────────────────────────────────

proptest! {
    /// A wildcard `*.apex` must always match any `prefix.apex` sub-host and
    /// the bare `apex`.
    #[test]
    fn wildcard_matches_apex_and_any_subdomain(
        apex in "[a-z][a-z0-9]{1,10}\\.[a-z]{2,6}",
        subdomain in "[a-z][a-z0-9]{1,20}",
    ) {
        let pattern = format!("*.{apex}");
        prop_assert!(matches_wildcard(&pattern, &apex), "apex `{apex}` must match `{pattern}`");
        let sub = format!("{subdomain}.{apex}");
        prop_assert!(
            matches_wildcard(&pattern, &sub),
            "subdomain `{sub}` must match `{pattern}`",
        );
    }

    /// Classic suffix-confusion bypass: `example.com.attacker.io` must NOT
    /// match `*.example.com`. Real-world bug: naive `ends_with("example.com")`
    /// would let the attacker host through.
    #[test]
    fn wildcard_rejects_suffix_confusion(
        apex in "[a-z][a-z0-9]{2,10}\\.[a-z]{2,6}",
        attacker in "[a-z][a-z0-9]{2,15}\\.[a-z]{2,6}",
    ) {
        prop_assume!(attacker != apex);
        let pattern = format!("*.{apex}");
        let evil = format!("{apex}.{attacker}");
        prop_assert!(
            !matches_wildcard(&pattern, &evil),
            "suffix-confusion `{evil}` must NOT match `{pattern}`",
        );
    }

    /// Prefix-confusion bypass: `fakeexample.com` must NOT match
    /// `*.example.com`. The dot separator is load-bearing.
    #[test]
    fn wildcard_rejects_prefix_confusion(
        apex in "[a-z][a-z0-9]{2,10}\\.[a-z]{2,6}",
        prefix in "[a-z][a-z0-9]{1,5}",
    ) {
        prop_assume!(!prefix.is_empty());
        let pattern = format!("*.{apex}");
        let evil = format!("{prefix}{apex}"); // no dot separator
        prop_assert!(
            !matches_wildcard(&pattern, &evil),
            "prefix-confusion `{evil}` must NOT match `{pattern}`",
        );
    }

    /// Exact (non-wildcard) domain patterns must not do any loose matching.
    #[test]
    fn exact_domain_requires_exact_equality(
        a in "[a-z][a-z0-9]{2,10}\\.[a-z]{2,6}",
        b in "[a-z][a-z0-9]{2,10}\\.[a-z]{2,6}",
    ) {
        prop_assert!(matches_wildcard(&a, &a));
        if a != b {
            prop_assert!(!matches_wildcard(&a, &b));
        }
    }

    /// Empty domain on the request must never match any pattern.
    #[test]
    fn absent_domain_never_matches(pattern in "[a-z*.]{1,20}") {
        let c = PolicyCondition::Domain(pattern);
        let req = PolicyRequest::new(); // domain = None
        prop_assert!(!c.matches(&req));
    }
}
