//! End-to-end coverage of the SQLite-backed `SessionStore` exercised through
//! the public crate API (no access to private internals).

#![cfg(feature = "sqlite")]

use brainwires_stores::{ListOptions, Message, SessionId, SessionStore, SqliteSessionStore};

fn open_tmp_store() -> (SqliteSessionStore, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = SqliteSessionStore::open(tmp.path().join("sessions.db")).expect("open store");
    (store, tmp)
}

#[tokio::test]
async fn save_and_load_roundtrip_preserves_messages() {
    let (store, _tmp) = open_tmp_store();
    let id = SessionId::new("user-42");
    let messages = vec![
        Message::user("what's the weather?"),
        Message::assistant("sunny and 72"),
        Message::user("thanks"),
    ];

    store.save(&id, &messages).await.expect("save");
    let loaded = store
        .load(&id)
        .await
        .expect("load")
        .expect("session must exist after save");

    assert_eq!(loaded.len(), messages.len());
    assert_eq!(loaded[0].text(), Some("what's the weather?"));
    assert_eq!(loaded[1].text(), Some("sunny and 72"));
    assert_eq!(loaded[2].text(), Some("thanks"));
}

#[tokio::test]
async fn save_overwrites_existing_session() {
    let (store, _tmp) = open_tmp_store();
    let id = SessionId::new("conv-1");

    store.save(&id, &[Message::user("first")]).await.unwrap();
    store
        .save(&id, &[Message::user("first"), Message::assistant("second")])
        .await
        .unwrap();

    let loaded = store.load(&id).await.unwrap().expect("session present");
    assert_eq!(loaded.len(), 2, "save must overwrite, not append");
}

#[tokio::test]
async fn delete_removes_session_and_load_returns_none() {
    let (store, _tmp) = open_tmp_store();
    let id = SessionId::new("ephemeral");
    store.save(&id, &[Message::user("hello")]).await.unwrap();
    assert!(store.load(&id).await.unwrap().is_some());

    store.delete(&id).await.unwrap();
    assert!(store.load(&id).await.unwrap().is_none());
}

#[tokio::test]
async fn delete_unknown_id_is_no_op() {
    let (store, _tmp) = open_tmp_store();
    store
        .delete(&SessionId::new("never-existed"))
        .await
        .expect("delete on unknown id must succeed silently");
}

#[tokio::test]
async fn list_enumerates_all_sessions() {
    let (store, _tmp) = open_tmp_store();
    for id in ["alpha", "beta", "gamma"] {
        store
            .save(&SessionId::new(id), &[Message::user(id)])
            .await
            .unwrap();
    }
    let listed = store.list().await.unwrap();
    assert_eq!(listed.len(), 3);
    let ids: std::collections::HashSet<_> =
        listed.iter().map(|r| r.id.as_str().to_string()).collect();
    assert_eq!(
        ids,
        ["alpha", "beta", "gamma"]
            .iter()
            .map(|s| s.to_string())
            .collect::<std::collections::HashSet<_>>()
    );
}

#[tokio::test]
async fn list_record_message_count_matches_save() {
    let (store, _tmp) = open_tmp_store();
    store
        .save(
            &SessionId::new("counted"),
            &[
                Message::user("a"),
                Message::user("b"),
                Message::user("c"),
                Message::user("d"),
            ],
        )
        .await
        .unwrap();
    let listed = store.list().await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].message_count, 4);
}

#[tokio::test]
async fn list_paginated_offset_and_limit_apply() {
    let (store, _tmp) = open_tmp_store();
    // Save in a known order; SQLite stamps updated_at on save, so insertion
    // order == ascending updated_at order.
    for id in ["one", "two", "three", "four", "five"] {
        store
            .save(&SessionId::new(id), &[Message::user(id)])
            .await
            .unwrap();
        // Brief gap so updated_at timestamps don't collide at second resolution.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    let page = store
        .list_paginated(ListOptions::new(1, Some(2)))
        .await
        .unwrap();
    assert_eq!(page.len(), 2, "limit=2 must cap result size");

    let all = store.list().await.unwrap();
    assert_eq!(all.len(), 5);
    assert_eq!(page[0].id.as_str(), all[1].id.as_str());
    assert_eq!(page[1].id.as_str(), all[2].id.as_str());

    let beyond = store
        .list_paginated(ListOptions::new(100, Some(10)))
        .await
        .unwrap();
    assert!(beyond.is_empty(), "offset past end yields empty page");

    let no_limit = store
        .list_paginated(ListOptions::new(2, None))
        .await
        .unwrap();
    assert_eq!(no_limit.len(), 3, "limit=None returns rest of the table");
}

#[tokio::test]
async fn reopen_recovers_persisted_state() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sessions.db");
    let id = SessionId::new("durable");

    {
        let store = SqliteSessionStore::open(&path).unwrap();
        store
            .save(&id, &[Message::user("survives restart")])
            .await
            .unwrap();
    }

    let store = SqliteSessionStore::open(&path).unwrap();
    let loaded = store
        .load(&id)
        .await
        .unwrap()
        .expect("session must persist across reopen");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].text(), Some("survives restart"));
}
