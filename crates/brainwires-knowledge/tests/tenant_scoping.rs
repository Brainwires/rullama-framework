//! Integration tests for per-owner (tenant) scoping of thoughts.
//!
//! These tests verify that:
//!   - Writes with `owner_id = Some(x)` store that owner on the thought.
//!   - Reads with `owner_id = Some(x)` only surface thoughts owned by `x`.
//!   - Reads with `owner_id = None` preserve pre-tenant single-tenant behavior
//!     (every row regardless of owner is visible).
//!   - Capturing with `owner_id = None` and listing with `owner_id = Some(x)`
//!     does NOT leak the unscoped thought into the scoped view.
//!
//! These tests use the default LanceDB backend via `BrainClient::with_paths`.

#![cfg(feature = "knowledge")]

use brainwires_knowledge::knowledge::brain_client::BrainClient;
use brainwires_knowledge::knowledge::types::{
    CaptureThoughtRequest, ListRecentRequest, SearchMemoryRequest,
};
use tempfile::TempDir;

async fn setup() -> (TempDir, BrainClient) {
    let temp = TempDir::new().expect("failed to create tempdir");
    let lance_path = temp.path().join("brain.lance");
    let pks_path = temp.path().join("pks.db");
    let bks_path = temp.path().join("bks.db");

    let client = BrainClient::with_paths(
        lance_path.to_str().unwrap(),
        pks_path.to_str().unwrap(),
        bks_path.to_str().unwrap(),
    )
    .await
    .expect("failed to construct BrainClient");

    (temp, client)
}

fn capture(content: &str, owner: Option<&str>) -> CaptureThoughtRequest {
    CaptureThoughtRequest {
        content: content.to_string(),
        category: None,
        tags: None,
        importance: None,
        source: None,
        owner_id: owner.map(|s| s.to_string()),
    }
}

#[tokio::test]
async fn list_recent_is_scoped_by_owner() {
    let (_tmp, mut client) = setup().await;

    let alice_id = client
        .capture_thought(capture("alice loves rust programming", Some("alice")))
        .await
        .unwrap()
        .id;
    let bob_id = client
        .capture_thought(capture("bob prefers go programming", Some("bob")))
        .await
        .unwrap()
        .id;

    // Alice's view
    let alice_list = client
        .list_recent(ListRecentRequest {
            limit: 100,
            category: None,
            since: None,
            owner_id: Some("alice".into()),
        })
        .await
        .unwrap();
    let alice_ids: Vec<_> = alice_list.thoughts.iter().map(|t| t.id.clone()).collect();
    assert!(
        alice_ids.contains(&alice_id),
        "alice should see her thought"
    );
    assert!(
        !alice_ids.contains(&bob_id),
        "alice must NOT see bob's thought"
    );

    // Bob's view
    let bob_list = client
        .list_recent(ListRecentRequest {
            limit: 100,
            category: None,
            since: None,
            owner_id: Some("bob".into()),
        })
        .await
        .unwrap();
    let bob_ids: Vec<_> = bob_list.thoughts.iter().map(|t| t.id.clone()).collect();
    assert!(bob_ids.contains(&bob_id), "bob should see his thought");
    assert!(
        !bob_ids.contains(&alice_id),
        "bob must NOT see alice's thought"
    );
}

#[tokio::test]
async fn get_thought_with_wrong_owner_returns_none() {
    let (_tmp, mut client) = setup().await;

    let alice_id = client
        .capture_thought(capture("alice's secret note", Some("alice")))
        .await
        .unwrap()
        .id;

    // Alice can fetch her own thought.
    let got = client.get_thought(&alice_id, Some("alice")).await.unwrap();
    assert!(got.is_some(), "alice should see her own thought");

    // Bob cannot.
    let denied = client.get_thought(&alice_id, Some("bob")).await.unwrap();
    assert!(
        denied.is_none(),
        "bob must not be able to fetch alice's thought"
    );

    // Unscoped reads still work (backward compat).
    let unscoped = client.get_thought(&alice_id, None).await.unwrap();
    assert!(unscoped.is_some(), "unscoped read must still find thought");
}

#[tokio::test]
async fn search_memory_does_not_leak_across_owners() {
    let (_tmp, mut client) = setup().await;

    client
        .capture_thought(capture(
            "alice enjoys rust programming language",
            Some("alice"),
        ))
        .await
        .unwrap();
    let bob_resp = client
        .capture_thought(capture(
            "bob also loves rust programming language",
            Some("bob"),
        ))
        .await
        .unwrap();

    let alice_results = client
        .search_memory(SearchMemoryRequest {
            query: "rust programming".into(),
            limit: 10,
            min_score: 0.0,
            category: None,
            sources: Some(vec!["thoughts".into()]),
            owner_id: Some("alice".into()),
        })
        .await
        .unwrap();

    for r in &alice_results.results {
        assert_ne!(
            r.thought_id.as_deref(),
            Some(bob_resp.id.as_str()),
            "alice's search must never surface bob's thought"
        );
    }
    assert!(
        !alice_results.results.is_empty(),
        "alice should still see her own match"
    );
}

#[tokio::test]
async fn delete_with_wrong_owner_is_noop() {
    let (_tmp, mut client) = setup().await;

    let alice_id = client
        .capture_thought(capture("keep this around", Some("alice")))
        .await
        .unwrap()
        .id;

    // Bob attempts to delete alice's thought — must fail silently.
    let del = client.delete_thought(&alice_id, Some("bob")).await.unwrap();
    assert!(!del.deleted, "cross-owner delete must not report success");

    // Alice can still see her thought.
    let still_there = client.get_thought(&alice_id, Some("alice")).await.unwrap();
    assert!(
        still_there.is_some(),
        "thought must survive a cross-owner delete attempt"
    );

    // Alice can actually delete it.
    let real_del = client
        .delete_thought(&alice_id, Some("alice"))
        .await
        .unwrap();
    assert!(real_del.deleted, "owner-scoped delete must succeed");

    let gone = client.get_thought(&alice_id, Some("alice")).await.unwrap();
    assert!(gone.is_none(), "thought should be gone after real delete");
}

#[tokio::test]
async fn unscoped_capture_is_invisible_to_scoped_readers() {
    let (_tmp, mut client) = setup().await;

    let unscoped_id = client
        .capture_thought(capture("legacy single-tenant thought", None))
        .await
        .unwrap()
        .id;

    // Unscoped reader sees it (backward compat).
    let unscoped_list = client
        .list_recent(ListRecentRequest {
            limit: 100,
            category: None,
            since: None,
            owner_id: None,
        })
        .await
        .unwrap();
    let unscoped_ids: Vec<_> = unscoped_list
        .thoughts
        .iter()
        .map(|t| t.id.clone())
        .collect();
    assert!(
        unscoped_ids.contains(&unscoped_id),
        "unscoped reader must see unscoped thoughts"
    );

    // Alice (scoped) does NOT see it.
    let alice_list = client
        .list_recent(ListRecentRequest {
            limit: 100,
            category: None,
            since: None,
            owner_id: Some("alice".into()),
        })
        .await
        .unwrap();
    let alice_ids: Vec<_> = alice_list.thoughts.iter().map(|t| t.id.clone()).collect();
    assert!(
        !alice_ids.contains(&unscoped_id),
        "scoped reader must NOT see unscoped thoughts"
    );
}

#[tokio::test]
async fn update_thought_respects_owner_scope() {
    let (_tmp, mut client) = setup().await;

    let alice_id = client
        .capture_thought(capture("original alice content", Some("alice")))
        .await
        .unwrap()
        .id;

    // Bob cannot update alice's thought.
    let bob_update = client
        .update_thought(&alice_id, "bob was here".into(), Some("bob".into()))
        .await
        .unwrap();
    assert!(!bob_update, "cross-owner update_thought must be a no-op");

    let after_bob = client
        .get_thought(&alice_id, Some("alice"))
        .await
        .unwrap()
        .expect("alice's thought must still exist");
    assert_eq!(after_bob.content, "original alice content");

    // Alice can update her own thought.
    let alice_update = client
        .update_thought(&alice_id, "updated content".into(), Some("alice".into()))
        .await
        .unwrap();
    assert!(alice_update, "owner-scoped update_thought must succeed");

    let after_alice = client
        .get_thought(&alice_id, Some("alice"))
        .await
        .unwrap()
        .expect("alice's thought must still exist after her update");
    assert_eq!(after_alice.content, "updated content");
}
