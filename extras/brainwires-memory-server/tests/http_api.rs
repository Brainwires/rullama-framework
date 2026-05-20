//! HTTP integration tests for the Mem0-compatible memory server.
//!
//! Each test spins up the Axum app in-process on an ephemeral port with a
//! fresh [`TempDir`] so tests do not share state.

use std::net::SocketAddr;

use brainwires_memory_server::{AppState, build_app, build_client};
use reqwest::StatusCode;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

struct TestServer {
    base: String,
    _tmp: TempDir,
    handle: JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

async fn spawn_server() -> TestServer {
    let tmp = TempDir::new().expect("tempdir");
    let client = build_client(tmp.path())
        .await
        .expect("failed to build BrainClient");
    let app = build_app(AppState::new(client));

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind ephemeral port");
    let addr: SocketAddr = listener.local_addr().expect("local_addr");
    let base = format!("http://{addr}");

    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    TestServer {
        base,
        _tmp: tmp,
        handle,
    }
}

fn http() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("reqwest client")
}

async fn post_memory(base: &str, body: Value) -> reqwest::Response {
    http()
        .post(format!("{base}/v1/memories"))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/memories")
}

async fn created_id(resp: reqwest::Response) -> String {
    assert!(
        resp.status().is_success(),
        "expected success, got {}",
        resp.status()
    );
    let v: Value = resp.json().await.expect("json body");
    v["results"][0]["id"]
        .as_str()
        .expect("id in first result")
        .to_string()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok() {
    let srv = spawn_server().await;
    let resp = http()
        .get(format!("{}/health", srv.base))
        .send()
        .await
        .expect("GET /health");
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["status"], "ok");
}

#[tokio::test]
async fn add_memory_returns_id_per_user() {
    let srv = spawn_server().await;

    let alice = post_memory(
        &srv.base,
        json!({ "memory": "alice likes rust", "user_id": "alice" }),
    )
    .await;
    let alice_id = created_id(alice).await;

    let bob = post_memory(
        &srv.base,
        json!({ "memory": "bob likes go", "user_id": "bob" }),
    )
    .await;
    let bob_id = created_id(bob).await;

    assert_ne!(alice_id, bob_id);
}

#[tokio::test]
async fn list_memories_is_tenant_scoped() {
    let srv = spawn_server().await;

    let alice_id = created_id(
        post_memory(
            &srv.base,
            json!({ "memory": "alice fact", "user_id": "alice" }),
        )
        .await,
    )
    .await;
    let bob_id =
        created_id(post_memory(&srv.base, json!({ "memory": "bob fact", "user_id": "bob" })).await)
            .await;

    let alice_list: Value = http()
        .get(format!("{}/v1/memories?user_id=alice", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let alice_ids: Vec<String> = alice_list["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert!(alice_ids.contains(&alice_id), "alice should see her memory");
    assert!(
        !alice_ids.contains(&bob_id),
        "alice must not see bob's memory"
    );

    let bob_list: Value = http()
        .get(format!("{}/v1/memories?user_id=bob", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let bob_ids: Vec<String> = bob_list["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert!(bob_ids.contains(&bob_id), "bob should see his memory");
    assert!(
        !bob_ids.contains(&alice_id),
        "bob must not see alice's memory"
    );
}

#[tokio::test]
async fn get_memory_cross_tenant_returns_404() {
    let srv = spawn_server().await;

    let alice_id = created_id(
        post_memory(
            &srv.base,
            json!({ "memory": "alice secret note", "user_id": "alice" }),
        )
        .await,
    )
    .await;

    // Bob cannot fetch Alice's memory.
    let bob_resp = http()
        .get(format!("{}/v1/memories/{}?user_id=bob", srv.base, alice_id))
        .send()
        .await
        .unwrap();
    assert_eq!(bob_resp.status(), StatusCode::NOT_FOUND);

    // Alice can.
    let alice_resp = http()
        .get(format!(
            "{}/v1/memories/{}?user_id=alice",
            srv.base, alice_id
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(alice_resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn search_is_tenant_scoped() {
    let srv = spawn_server().await;

    let _alice_id = created_id(
        post_memory(
            &srv.base,
            json!({ "memory": "alice enjoys rust programming", "user_id": "alice" }),
        )
        .await,
    )
    .await;
    let _bob_id = created_id(
        post_memory(
            &srv.base,
            json!({ "memory": "bob likes go", "user_id": "bob" }),
        )
        .await,
    )
    .await;

    // Bob searching for "rust" must not leak alice's memory.
    let bob_search: Value = http()
        .post(format!("{}/v1/memories/search", srv.base))
        .json(&json!({ "query": "rust programming", "user_id": "bob", "limit": 10 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let bob_results = bob_search["results"].as_array().unwrap();
    for r in bob_results {
        let content = r["memory"].as_str().unwrap_or("");
        assert!(
            !content.contains("alice"),
            "bob's search must not surface alice's memory: {content}"
        );
    }

    // Alice searching for "rust" should find her memory.
    let alice_search: Value = http()
        .post(format!("{}/v1/memories/search", srv.base))
        .json(&json!({ "query": "rust programming", "user_id": "alice", "limit": 10 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let alice_results = alice_search["results"].as_array().unwrap();
    assert!(
        !alice_results.is_empty(),
        "alice should see her rust-related memory"
    );
    for r in alice_results {
        let content = r["memory"].as_str().unwrap_or("");
        assert!(
            !content.contains("bob"),
            "alice's search must not surface bob's memory: {content}"
        );
    }
}

#[tokio::test]
async fn patch_memory_updates_content() {
    let srv = spawn_server().await;

    let alice_id = created_id(
        post_memory(
            &srv.base,
            json!({ "memory": "original content", "user_id": "alice" }),
        )
        .await,
    )
    .await;

    let patch = http()
        .patch(format!("{}/v1/memories/{}", srv.base, alice_id))
        .json(&json!({ "memory": "updated content", "user_id": "alice" }))
        .send()
        .await
        .unwrap();
    assert_eq!(patch.status(), StatusCode::OK);

    let fetched: Value = http()
        .get(format!(
            "{}/v1/memories/{}?user_id=alice",
            srv.base, alice_id
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(fetched["memory"].as_str().unwrap(), "updated content");
}

#[tokio::test]
async fn delete_memory_is_tenant_scoped() {
    let srv = spawn_server().await;

    let alice_id = created_id(
        post_memory(
            &srv.base,
            json!({ "memory": "to be deleted eventually", "user_id": "alice" }),
        )
        .await,
    )
    .await;

    // Bob can't delete alice's memory.
    let bob_delete = http()
        .delete(format!("{}/v1/memories/{}?user_id=bob", srv.base, alice_id))
        .send()
        .await
        .unwrap();
    assert_eq!(bob_delete.status(), StatusCode::NOT_FOUND);

    // Alice's memory still exists.
    let still = http()
        .get(format!(
            "{}/v1/memories/{}?user_id=alice",
            srv.base, alice_id
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(still.status(), StatusCode::OK);

    // Alice can delete it.
    let alice_delete = http()
        .delete(format!(
            "{}/v1/memories/{}?user_id=alice",
            srv.base, alice_id
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(alice_delete.status(), StatusCode::OK);

    // Now it's gone.
    let gone = http()
        .get(format!(
            "{}/v1/memories/{}?user_id=alice",
            srv.base, alice_id
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(gone.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn add_without_user_id_is_400() {
    let srv = spawn_server().await;
    let resp = http()
        .post(format!("{}/v1/memories", srv.base))
        .json(&json!({ "memory": "no owner" }))
        .send()
        .await
        .unwrap();
    // Either 400 (explicit) or 422 (serde rejects missing user_id) is acceptable
    // for "bad request"; the key invariant is we do NOT create an untracked row.
    assert!(
        resp.status() == StatusCode::BAD_REQUEST
            || resp.status() == StatusCode::UNPROCESSABLE_ENTITY,
        "expected 400 or 422, got {}",
        resp.status()
    );
}
