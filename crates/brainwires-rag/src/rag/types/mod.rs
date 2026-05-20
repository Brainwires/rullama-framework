// Re-export shared types from core (SearchResult, ChunkMetadata used by both storage and cognition)
pub use brainwires_core::{ChunkMetadata, SearchResult};

mod code_relations;
mod ensemble;
mod incremental;
mod index;
mod query;
mod search;
mod statistics;

#[cfg(test)]
mod tests;

// --- index ---
pub use index::default_max_file_size;
pub use index::{IndexRequest, IndexResponse, IndexingMode};

// --- query ---
pub use query::{QueryRequest, QueryResponse};
pub use query::{default_hybrid, default_limit, default_min_score};

// --- statistics ---
pub use statistics::{
    ClearRequest, ClearResponse, LanguageStats, StatisticsRequest, StatisticsResponse,
};

// --- incremental ---
pub use incremental::{IncrementalUpdateRequest, IncrementalUpdateResponse};

// --- search ---
pub use search::{
    AdvancedSearchRequest, GitSearchResult, SearchGitHistoryRequest, SearchGitHistoryResponse,
};
pub use search::{default_git_path, default_max_commits};

// --- code relations ---
pub use code_relations::{FindDefinitionRequest, FindReferencesRequest, GetCallGraphRequest};

pub use code_relations::{FindDefinitionResponse, FindReferencesResponse, GetCallGraphResponse};

// --- ensemble ---
pub use ensemble::{EnsembleRequest, EnsembleResponse, SearchStrategy};
