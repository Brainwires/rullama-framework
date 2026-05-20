//! Round-trip integration tests for the seven Lance-backed stores in
//! `brainwires-stores`. Exercises the CRUD path through the public API
//! against a real `LanceDatabase` in a tempdir — no mocks of the backend.
//!
//! ## Test categorisation
//!
//! Tests for `TaskStore`, `AgentStateStore`, and `PlanStore` run by
//! default — none of them call `embeddings.embed()` on the happy CRUD
//! path (PlanStore's `save()` short-circuits when `plan.embedding` is
//! already `Some(_)`, which these tests exploit).
//!
//! Tests for `ImageStore`, `MessageStore`, `SummaryStore`, `FactStore`,
//! and `MentalModelStore` are marked `#[ignore = "downloads fastembed
//! model"]` because their `add()` / `store()` methods unconditionally
//! call `self.embeddings.embed(...)` which loads (and on a clean system,
//! downloads) the all-MiniLM-L6-v2 ONNX model. Run with
//! `cargo test -p brainwires-stores --features
//! "session,task,plan,conversation,memory,image,lock" --test
//! lance_roundtrip -- --include-ignored` once the cache is warm.

#![cfg(all(
    feature = "plan",
    feature = "task",
    feature = "image",
    feature = "memory",
))]

use std::sync::Arc;

use brainwires_core::plan::{PlanMetadata, PlanStatus};
use brainwires_core::task::{Task, TaskPriority, TaskStatus};
use brainwires_storage::databases::lance::LanceDatabase;
use brainwires_storage::databases::traits::StorageBackend;
use brainwires_storage::embeddings::CachedEmbeddingProvider;
use brainwires_storage::image_types::ImageFormat;
use brainwires_stores::{
    AgentStateMetadata, AgentStateStore, FactStore, FactType, ImageStore, KeyFact, MentalModel,
    MentalModelStore, MessageMetadata, MessageStore, MessageSummary, ModelType, PlanStore,
    SummaryStore, TaskStore,
};

// ── Test helpers ──────────────────────────────────────────────────────────────

async fn open_db(tag: &str) -> (Arc<LanceDatabase>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join(format!("{tag}.lance"));
    let db = Arc::new(
        LanceDatabase::new(path.to_str().expect("utf-8 tempdir path"))
            .await
            .expect("open LanceDatabase"),
    );
    (db, tmp)
}

fn embeddings() -> Arc<CachedEmbeddingProvider> {
    // Cheap — only allocates a struct + LRU. The ONNX model is lazily
    // initialised on first `.embed()` call (OnceLock).
    Arc::new(CachedEmbeddingProvider::new().expect("CachedEmbeddingProvider"))
}

// ── PlanStore ─────────────────────────────────────────────────────────────────
//
// PlanStore::save() only embeds when `plan.embedding.is_none()`, so by
// pre-filling a zero-vector of the right dim we keep these tests in the
// default suite without hitting the ONNX model.

fn make_plan_with_zero_embedding(conversation_id: &str, task: &str) -> PlanMetadata {
    let mut plan = PlanMetadata::new(
        conversation_id.to_string(),
        task.to_string(),
        format!("# Plan for {task}\n\n1. Investigate\n2. Implement\n3. Verify"),
    );
    plan.embedding = Some(vec![0.0_f32; 384]); // bypass embeddings.embed() in save()
    plan
}

#[tokio::test]
async fn plan_store_save_and_get_roundtrip() {
    let (db, _tmp) = open_db("plan_save_get").await;
    let store = PlanStore::new(db, embeddings());
    store.ensure_table().await.expect("ensure_table");

    let plan = make_plan_with_zero_embedding("conv-A", "Investigate Q3 regression");
    let plan_id = plan.plan_id.clone();
    store.save(&plan).await.expect("save");

    let loaded = store
        .get(&plan_id)
        .await
        .expect("get")
        .expect("plan present after save");

    assert_eq!(loaded.plan_id, plan_id);
    assert_eq!(loaded.conversation_id, "conv-A");
    assert_eq!(loaded.task_description, "Investigate Q3 regression");
    assert!(loaded.plan_content.contains("Investigate"));
    assert_eq!(loaded.status, PlanStatus::Draft);
    assert!(!loaded.executed);
}

#[tokio::test]
async fn plan_store_get_by_conversation_isolates() {
    let (db, _tmp) = open_db("plan_by_conv").await;
    let store = PlanStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    for task in ["task one", "task two", "task three"] {
        store
            .save(&make_plan_with_zero_embedding("conv-X", task))
            .await
            .unwrap();
    }
    store
        .save(&make_plan_with_zero_embedding("conv-Y", "outsider"))
        .await
        .unwrap();

    let in_x = store.get_by_conversation("conv-X").await.unwrap();
    assert_eq!(in_x.len(), 3, "conv-X must yield exactly its three plans");
    let in_y = store.get_by_conversation("conv-Y").await.unwrap();
    assert_eq!(in_y.len(), 1);
    assert_eq!(in_y[0].task_description, "outsider");
}

#[tokio::test]
async fn plan_store_delete_removes_record() {
    let (db, _tmp) = open_db("plan_delete").await;
    let store = PlanStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    let plan = make_plan_with_zero_embedding("conv-del", "ephemeral");
    let plan_id = plan.plan_id.clone();
    store.save(&plan).await.unwrap();
    assert!(store.get(&plan_id).await.unwrap().is_some());

    store.delete(&plan_id).await.unwrap();
    assert!(
        store.get(&plan_id).await.unwrap().is_none(),
        "plan must be gone after delete"
    );
}

#[tokio::test]
async fn plan_store_hierarchy_parent_and_children() {
    let (db, _tmp) = open_db("plan_hierarchy").await;
    let store = PlanStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    // Parent
    let mut parent = make_plan_with_zero_embedding("conv-tree", "root task");
    let parent_id = parent.plan_id.clone();
    parent.child_plan_ids = vec!["child-A".into(), "child-B".into()];
    store.save(&parent).await.unwrap();

    // Two children linked via parent_plan_id
    for child_name in ["child-A", "child-B"] {
        let mut child = make_plan_with_zero_embedding("conv-tree", child_name);
        child.plan_id = child_name.to_string();
        child.parent_plan_id = Some(parent_id.clone());
        child.depth = 1;
        store.save(&child).await.unwrap();
    }

    let kids = store.get_children(&parent_id).await.unwrap();
    assert_eq!(kids.len(), 2, "must surface both children");
    let names: std::collections::HashSet<_> = kids.iter().map(|p| p.plan_id.clone()).collect();
    assert!(names.contains("child-A"));
    assert!(names.contains("child-B"));

    let tree = store.get_hierarchy(&parent_id).await.unwrap();
    assert_eq!(tree.len(), 3, "hierarchy is parent + 2 children");
    assert_eq!(tree[0].plan_id, parent_id, "first element is the root");
}

// ── TaskStore ─────────────────────────────────────────────────────────────────
//
// TaskStore takes no embeddings — runs full CRUD by default.

fn make_task(id: &str, description: &str) -> Task {
    let mut t = Task::new(id.to_string(), description.to_string());
    t.priority = TaskPriority::default();
    t.status = TaskStatus::Pending;
    t
}

#[tokio::test]
async fn task_store_save_and_get_roundtrip() {
    let (db, _tmp) = open_db("task_save_get").await;
    let store = TaskStore::new(db);
    store.ensure_table().await.unwrap();

    let task = make_task("task-001", "Investigate the panic");
    store.save(&task, "conv-task-1").await.unwrap();

    let loaded = store
        .get(&task.id)
        .await
        .unwrap()
        .expect("task present after save");
    assert_eq!(loaded.id, "task-001");
    assert_eq!(loaded.description, "Investigate the panic");
    assert!(matches!(loaded.status, TaskStatus::Pending));
}

#[tokio::test]
async fn task_store_get_by_conversation_isolates() {
    let (db, _tmp) = open_db("task_by_conv").await;
    let store = TaskStore::new(db);
    store.ensure_table().await.unwrap();

    store
        .save(&make_task("t-a", "alpha"), "conv-1")
        .await
        .unwrap();
    store
        .save(&make_task("t-b", "beta"), "conv-1")
        .await
        .unwrap();
    store
        .save(&make_task("t-c", "gamma"), "conv-2")
        .await
        .unwrap();

    let in_1 = store.get_by_conversation("conv-1").await.unwrap();
    assert_eq!(in_1.len(), 2);
    let in_2 = store.get_by_conversation("conv-2").await.unwrap();
    assert_eq!(in_2.len(), 1);
    assert_eq!(in_2[0].description, "gamma");
}

#[tokio::test]
async fn task_store_delete_removes_record() {
    let (db, _tmp) = open_db("task_delete").await;
    let store = TaskStore::new(db);
    store.ensure_table().await.unwrap();

    let task = make_task("t-rm", "soon to vanish");
    store.save(&task, "conv-rm").await.unwrap();
    assert!(store.get("t-rm").await.unwrap().is_some());

    store.delete("t-rm").await.unwrap();
    assert!(store.get("t-rm").await.unwrap().is_none());
}

// ── AgentStateStore ──────────────────────────────────────────────────────────
//
// Lives in the same file as TaskStore. Also embedding-free.

fn make_agent_state(agent_id: &str, task_id: &str, conv_id: &str) -> AgentStateMetadata {
    let now = chrono::Utc::now().timestamp();
    AgentStateMetadata {
        agent_id: agent_id.to_string(),
        task_id: task_id.to_string(),
        conversation_id: conv_id.to_string(),
        status: "running".to_string(),
        iteration: 3,
        context_json: r#"{"step":"verify"}"#.to_string(),
        created_at: now,
        updated_at: now,
    }
}

#[tokio::test]
async fn agent_state_store_save_and_get_roundtrip() {
    let (db, _tmp) = open_db("agent_save_get").await;
    let store = AgentStateStore::new(db);
    store.ensure_table().await.unwrap();

    let state = make_agent_state("agent-1", "task-1", "conv-1");
    store.save(&state).await.unwrap();

    let loaded = store
        .get("agent-1")
        .await
        .unwrap()
        .expect("agent state present after save");
    assert_eq!(loaded.agent_id, "agent-1");
    assert_eq!(loaded.task_id, "task-1");
    assert_eq!(loaded.iteration, 3);
    assert_eq!(loaded.status, "running");
}

#[tokio::test]
async fn agent_state_store_delete_removes_record() {
    let (db, _tmp) = open_db("agent_delete").await;
    let store = AgentStateStore::new(db);
    store.ensure_table().await.unwrap();

    store
        .save(&make_agent_state("agent-rm", "task-rm", "conv-rm"))
        .await
        .unwrap();
    assert!(store.get("agent-rm").await.unwrap().is_some());

    store.delete("agent-rm").await.unwrap();
    assert!(store.get("agent-rm").await.unwrap().is_none());
}

// ── ImageStore ────────────────────────────────────────────────────────────────
//
// ImageStore.store() unconditionally embeds, so all of these are
// `#[ignore]`'d. The hash helper is pure and tested without ignore.

#[test]
fn image_store_compute_hash_is_deterministic_and_unique() {
    // ImageStore is generic over its backend with no type-inference hint here,
    // so we explicitly pick the default LanceDatabase to satisfy the bound.
    type DefaultImageStore = ImageStore<LanceDatabase>;
    let h1 = DefaultImageStore::compute_hash(b"alpha-bytes");
    let h2 = DefaultImageStore::compute_hash(b"alpha-bytes");
    let h3 = DefaultImageStore::compute_hash(b"different-bytes");
    assert_eq!(h1, h2, "same input yields same hash");
    assert_ne!(h1, h3, "different input yields different hash");
    assert_eq!(h1.len(), 64, "SHA-256 hex is 64 chars");
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn image_store_store_from_bytes_dedupes_by_hash() {
    let (db, _tmp) = open_db("image_dedupe").await;
    let store = ImageStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    let bytes = b"\x89PNG\r\n\x1a\nfake-png-body";
    let first = store
        .store_from_bytes(bytes, "a cat".into(), "conv-img".into(), ImageFormat::Png)
        .await
        .unwrap();
    let second = store
        .store_from_bytes(bytes, "a cat".into(), "conv-img".into(), ImageFormat::Png)
        .await
        .unwrap();

    assert_eq!(
        first.image_id, second.image_id,
        "duplicate bytes must return the existing image_id"
    );
    let by_hash = store
        .get_by_hash(&first.file_hash)
        .await
        .unwrap()
        .expect("hash lookup hits");
    assert_eq!(by_hash.image_id, first.image_id);
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn image_store_delete_removes_record() {
    let (db, _tmp) = open_db("image_delete").await;
    let store = ImageStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    let meta = store
        .store_from_bytes(
            b"some-bytes",
            "analysis".into(),
            "conv-x".into(),
            ImageFormat::Jpeg,
        )
        .await
        .unwrap();
    assert!(store.get(&meta.image_id).await.unwrap().is_some());

    let deleted = store.delete(&meta.image_id).await.unwrap();
    assert!(deleted);
    assert!(store.get(&meta.image_id).await.unwrap().is_none());
}

// ── MessageStore ─────────────────────────────────────────────────────────────

fn make_message(message_id: &str, conv_id: &str, role: &str, content: &str) -> MessageMetadata {
    MessageMetadata {
        message_id: message_id.to_string(),
        conversation_id: conv_id.to_string(),
        role: role.to_string(),
        content: content.to_string(),
        token_count: Some(content.split_whitespace().count() as i32),
        model_id: None,
        images: None,
        created_at: chrono::Utc::now().timestamp(),
        expires_at: None,
    }
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn message_store_add_and_get_roundtrip() {
    let (db, _tmp) = open_db("msg_add_get").await;
    let store = MessageStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    let msg = make_message("m-1", "conv-msg", "user", "hello world");
    store.add(msg.clone()).await.unwrap();

    let loaded = store.get("m-1").await.unwrap().expect("present after add");
    assert_eq!(loaded.message_id, "m-1");
    assert_eq!(loaded.content, "hello world");
    assert_eq!(loaded.role, "user");
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn message_store_get_by_conversation_isolates() {
    let (db, _tmp) = open_db("msg_by_conv").await;
    let store = MessageStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    store
        .add_batch(vec![
            make_message("m-a", "conv-A", "user", "first"),
            make_message("m-b", "conv-A", "assistant", "second"),
            make_message("m-c", "conv-B", "user", "isolated"),
        ])
        .await
        .unwrap();

    let in_a = store.get_by_conversation("conv-A").await.unwrap();
    assert_eq!(in_a.len(), 2);
    let in_b = store.get_by_conversation("conv-B").await.unwrap();
    assert_eq!(in_b.len(), 1);
    assert_eq!(in_b[0].content, "isolated");
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn message_store_delete_removes_record() {
    let (db, _tmp) = open_db("msg_delete").await;
    let store = MessageStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    store
        .add(make_message("m-rm", "conv-rm", "user", "byebye"))
        .await
        .unwrap();
    assert!(store.get("m-rm").await.unwrap().is_some());

    store.delete("m-rm").await.unwrap();
    assert!(store.get("m-rm").await.unwrap().is_none());
}

// ── SummaryStore ─────────────────────────────────────────────────────────────

fn make_summary(summary_id: &str, conv_id: &str, summary: &str) -> MessageSummary {
    MessageSummary {
        summary_id: summary_id.to_string(),
        original_message_id: format!("orig-of-{summary_id}"),
        conversation_id: conv_id.to_string(),
        role: "assistant".to_string(),
        summary: summary.to_string(),
        key_entities: vec!["entity-1".into(), "entity-2".into()],
        created_at: chrono::Utc::now().timestamp(),
    }
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn summary_store_add_and_get_roundtrip() {
    let (db, _tmp) = open_db("sum_add_get").await;
    let store = SummaryStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    store
        .add(make_summary("s-1", "conv-s", "compressed"))
        .await
        .unwrap();

    let loaded = store.get("s-1").await.unwrap().expect("present after add");
    assert_eq!(loaded.summary_id, "s-1");
    assert_eq!(loaded.summary, "compressed");
    assert_eq!(loaded.key_entities.len(), 2);
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn summary_store_add_batch_and_count() {
    let (db, _tmp) = open_db("sum_batch").await;
    let store = SummaryStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    store
        .add_batch(vec![
            make_summary("s-1", "conv-batch", "alpha"),
            make_summary("s-2", "conv-batch", "beta"),
            make_summary("s-3", "conv-batch", "gamma"),
        ])
        .await
        .unwrap();

    assert_eq!(store.count().await.unwrap(), 3);
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn summary_store_delete_removes_record() {
    let (db, _tmp) = open_db("sum_delete").await;
    let store = SummaryStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    store
        .add(make_summary("s-x", "conv-del", "to remove"))
        .await
        .unwrap();
    assert!(store.get("s-x").await.unwrap().is_some());

    store.delete("s-x").await.unwrap();
    assert!(store.get("s-x").await.unwrap().is_none());
}

// ── FactStore ────────────────────────────────────────────────────────────────

fn make_fact(fact_id: &str, conv_id: &str, fact: &str, fact_type: FactType) -> KeyFact {
    KeyFact {
        fact_id: fact_id.to_string(),
        original_message_ids: vec!["orig-1".into()],
        conversation_id: conv_id.to_string(),
        fact: fact.to_string(),
        fact_type,
        created_at: chrono::Utc::now().timestamp(),
    }
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn fact_store_add_and_get_roundtrip() {
    let (db, _tmp) = open_db("fact_add_get").await;
    let store = FactStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    store
        .add(make_fact(
            "f-1",
            "conv-fact",
            "User prefers Rust over Python",
            FactType::Decision,
        ))
        .await
        .unwrap();

    let loaded = store.get("f-1").await.unwrap().expect("present after add");
    assert_eq!(loaded.fact_id, "f-1");
    assert_eq!(loaded.fact, "User prefers Rust over Python");
    assert_eq!(loaded.fact_type, FactType::Decision);
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn fact_store_add_batch_and_count() {
    let (db, _tmp) = open_db("fact_batch").await;
    let store = FactStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    store
        .add_batch(vec![
            make_fact("f-a", "conv-b", "fact alpha", FactType::Definition),
            make_fact("f-b", "conv-b", "fact beta", FactType::Requirement),
        ])
        .await
        .unwrap();

    assert_eq!(store.count().await.unwrap(), 2);
}

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn fact_store_delete_removes_record() {
    let (db, _tmp) = open_db("fact_delete").await;
    let store = FactStore::new(db, embeddings());
    store.ensure_table().await.unwrap();

    store
        .add(make_fact(
            "f-rm",
            "conv-rm",
            "ephemeral fact",
            FactType::Other,
        ))
        .await
        .unwrap();
    assert!(store.get("f-rm").await.unwrap().is_some());

    store.delete("f-rm").await.unwrap();
    assert!(store.get("f-rm").await.unwrap().is_none());
}

// ── MentalModelStore ─────────────────────────────────────────────────────────
//
// Takes `Arc<dyn StorageBackend>`, not generic `Arc<B>`, so we coerce
// explicitly.

#[tokio::test]
#[ignore = "downloads fastembed model on first run"]
async fn mental_model_store_add_and_count() {
    let (db, _tmp) = open_db("mental_add").await;
    let backend: Arc<dyn StorageBackend> = db;
    let store = MentalModelStore::new(backend, embeddings());
    store.ensure_table().await.unwrap();

    let model = MentalModel::new(
        "User typically debugs by printf rather than gdb".to_string(),
        ModelType::Behavioral,
        "conv-mental".to_string(),
        vec!["fact-1".into(), "fact-2".into()],
    );
    let model_id = model.model_id.clone();
    store.add(model).await.unwrap();

    assert_eq!(store.count().await.unwrap(), 1);
    // No `get()` on this store; verify delete by id leaves count at zero.
    store.delete(&model_id).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
}
