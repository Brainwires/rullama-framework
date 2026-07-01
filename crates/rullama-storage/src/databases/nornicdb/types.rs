//! NornicDB public configuration types and enums.

// ── Public configuration types ──────────────────────────────────────────

/// Configuration for connecting to NornicDB.
#[derive(Debug, Clone)]
pub struct NornicConfig {
    /// Base URL of the NornicDB server (e.g. `http://localhost:7474`).
    pub url: String,
    /// Neo4j database name.
    pub database: String,
    /// Optional username for authentication.
    pub username: Option<String>,
    /// Optional password for authentication.
    pub password: Option<String>,
    /// Label applied to code-chunk nodes.
    pub node_label: String,
    /// Name of the vector index used for similarity search.
    pub index_name: String,
    /// Which wire protocol to use.
    pub transport: TransportKind,
}

impl Default for NornicConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:7474".to_string(),
            database: "neo4j".to_string(),
            username: None,
            password: None,
            node_label: "CodeChunk".to_string(),
            index_name: "code_embedding_index".to_string(),
            transport: TransportKind::Rest,
        }
    }
}

/// Which transport protocol to use when talking to NornicDB.
#[derive(Debug, Clone, Default)]
pub enum TransportKind {
    /// REST / HTTP on port 7474 (default, always available).
    #[default]
    Rest,
    /// Neo4j Bolt binary protocol on a custom port (typically 7687).
    Bolt {
        /// Bolt port number (default: 7687).
        port: u16,
    },
    /// Qdrant-compatible gRPC on a custom port (typically 6334).
    Grpc {
        /// gRPC port number (default: 6334).
        port: u16,
    },
}

/// Cognitive memory tier for NornicDB's decay system.
///
/// Each tier has a different half-life controlling how quickly memories
/// decay in relevance searches:
///
/// | Tier        | Half-life | Typical use           |
/// |-------------|-----------|-----------------------|
/// | Episodic    | 7 days    | Chat context          |
/// | Semantic    | 69 days   | Extracted facts       |
/// | Procedural  | 693 days  | Long-lived patterns   |
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CognitiveMemoryTier {
    /// 7-day half-life (chat context).
    Episodic,
    /// 69-day half-life (facts).
    Semantic,
    /// 693-day half-life (patterns).
    Procedural,
}

impl CognitiveMemoryTier {
    /// Return the Neo4j node label for this tier.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Episodic => "Episodic",
            Self::Semantic => "Semantic",
            Self::Procedural => "Procedural",
        }
    }
}

impl std::fmt::Display for CognitiveMemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_label())
    }
}
