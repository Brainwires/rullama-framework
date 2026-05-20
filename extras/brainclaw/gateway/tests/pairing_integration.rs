//! Integration tests for the DM pairing policy.
//!
//! Exercises the [`PairingHandler`] end-to-end against a real
//! [`PairingStore`], verifying that unknown peers get a pairing code
//! issued, that admin approval via `approve_by_code` lets them through
//! on the next message, and that `Open`-mode peers not in the allowlist
//! are rejected.

use std::sync::Arc;

use brainwires_gateway::pairing::{PairingHandler, PairingOutcome, PairingPolicy, PairingStore};
use tempfile::tempdir;

#[tokio::test]
async fn unknown_peer_is_gated_then_approved() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pairing.json");
    let store = Arc::new(PairingStore::load(&path).unwrap());

    let handler = PairingHandler::with_fixed_policy(
        Arc::clone(&store),
        PairingPolicy::Pairing {
            code_ttl_secs: 900,
            persist_approvals: true,
        },
    );

    // First contact: peer is unknown, so a code is issued.
    let first = handler
        .check("discord", "alice", "Alice", "hi bot")
        .await
        .unwrap();
    let code = match first {
        PairingOutcome::PendingCodeIssued { code, .. } => code,
        other => panic!("expected PendingCodeIssued, got {other:?}"),
    };

    // Operator approves the code out-of-band.
    let approved = store.approve_by_code(&code).await.unwrap();
    assert_eq!(approved, Some(("discord".to_string(), "alice".to_string())));

    // Second contact: peer is approved; message is allowed through.
    let second = handler
        .check("discord", "alice", "Alice", "hi again")
        .await
        .unwrap();
    assert_eq!(second, PairingOutcome::Allow);
}

#[tokio::test]
async fn open_policy_rejects_peer_not_in_allowlist() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pairing.json");
    let store = Arc::new(PairingStore::load(&path).unwrap());

    let handler = PairingHandler::with_fixed_policy(
        store,
        PairingPolicy::Open {
            allow_from: vec!["discord:alice".to_string()],
        },
    );

    match handler
        .check("discord", "bob", "Bob", "hello")
        .await
        .unwrap()
    {
        PairingOutcome::Reject(_) => {}
        other => panic!("expected Reject, got {other:?}"),
    }

    assert_eq!(
        handler
            .check("discord", "alice", "Alice", "hello")
            .await
            .unwrap(),
        PairingOutcome::Allow,
    );
}
