//! # brainwires-memory-server
//!
//! A Mem0-compatible memory REST API server for Brainwires agents.
//!
//! Storage is delegated to [`brainwires_knowledge::knowledge::brain_client::BrainClient`]
//! (LanceDB-backed thoughts with per-owner tenant scoping). Every request
//! must carry a `user_id`, which maps onto the knowledge layer's `owner_id`
//! for hard tenant isolation.
//!
//! ## API surface
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | `POST` | `/v1/memories` | Add one or more memories |
//! | `GET` | `/v1/memories` | List memories for a user |
//! | `GET` | `/v1/memories/{id}` | Get a single memory |
//! | `PATCH` | `/v1/memories/{id}` | Update memory content |
//! | `DELETE` | `/v1/memories/{id}` | Delete a memory |
//! | `DELETE` | `/v1/memories?user_id=…` | Delete all memories for a user |
//! | `POST` | `/v1/memories/search` | Semantic search |
//! | `GET` | `/health` | Health check |

pub mod routes;
pub mod types;

use std::sync::Arc;

use anyhow::Result;
use axum::{Router, routing};
use brainwires_knowledge::knowledge::brain_client::BrainClient;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Shared application state injected into every route handler.
#[derive(Clone)]
pub struct AppState {
    /// The underlying `BrainClient` from `brainwires-knowledge`.
    ///
    /// Wrapped in [`tokio::sync::Mutex`] because `BrainClient::capture_thought`
    /// and other mutating methods require `&mut self`.
    pub client: Arc<Mutex<BrainClient>>,
}

impl AppState {
    /// Construct an [`AppState`] from an owned [`BrainClient`].
    pub fn new(client: BrainClient) -> Self {
        Self {
            client: Arc::new(Mutex::new(client)),
        }
    }
}

/// Build a fresh [`BrainClient`] rooted at `storage_dir`.
///
/// Creates:
/// - LanceDB at `{storage_dir}/brain.lance`
/// - PKS at    `{storage_dir}/pks.db`
/// - BKS at    `{storage_dir}/bks.db`
pub async fn build_client(storage_dir: &std::path::Path) -> Result<BrainClient> {
    std::fs::create_dir_all(storage_dir)?;
    let lance = storage_dir.join("brain.lance");
    let pks = storage_dir.join("pks.db");
    let bks = storage_dir.join("bks.db");

    let client = BrainClient::with_paths(
        lance.to_str().expect("lance path must be valid UTF-8"),
        pks.to_str().expect("pks path must be valid UTF-8"),
        bks.to_str().expect("bks path must be valid UTF-8"),
    )
    .await?;
    Ok(client)
}

/// Build the Axum application router using the supplied state.
pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/health", routing::get(routes::health))
        .route(
            "/v1/memories",
            routing::post(routes::add_memory)
                .get(routes::list_memories)
                .delete(routes::delete_all_memories),
        )
        .route(
            "/v1/memories/search",
            routing::post(routes::search_memories),
        )
        .route(
            "/v1/memories/{id}",
            routing::get(routes::get_memory)
                .patch(routes::update_memory)
                .delete(routes::delete_memory),
        )
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
