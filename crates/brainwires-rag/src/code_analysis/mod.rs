//! Code relationships module for definition/reference tracking and call graphs.
//!
//! This module provides capabilities for understanding code relationships:
//! - Find where symbols are defined
//! - Find all references to a symbol
//! - Build call graphs for functions/methods
//!
//! Uses RepoMap (tree-sitter AST-based) for all languages.
//!
//! ## Usage
//!
//! ```ignore
//! use brainwires_rag::code_analysis::{HybridRelationsProvider, RelationsProvider};
//!
//! let provider = HybridRelationsProvider::new()?;
//! let definitions = provider.extract_definitions(&file_info)?;
//! let references = provider.extract_references(&file_info, &symbol_index)?;
//! ```

pub mod repomap;
pub mod storage;
pub mod types;

use anyhow::Result;

pub use types::{
    CallEdge, CallGraphNode, Definition, DefinitionResult, PrecisionLevel, Reference,
    ReferenceKind, ReferenceResult, SymbolId, SymbolInfo, SymbolKind, Visibility,
};

use crate::rag::indexer::FileInfo;
use std::collections::HashMap;

/// Trait for extracting code relationships from source files.
///
/// Implementors of this trait can extract symbol definitions and references
/// from source code files.
pub trait RelationsProvider: Send + Sync {
    /// Extract definitions from a file.
    ///
    /// Returns a list of all symbol definitions (functions, classes, etc.)
    /// found in the given file.
    fn extract_definitions(&self, file_info: &FileInfo) -> Result<Vec<Definition>>;

    /// Extract references from a file.
    ///
    /// `symbol_index` maps symbol names to their definitions, used for
    /// resolving which symbol a reference points to.
    fn extract_references(
        &self,
        file_info: &FileInfo,
        symbol_index: &HashMap<String, Vec<Definition>>,
    ) -> Result<Vec<Reference>>;

    /// Check if this provider supports the given language.
    fn supports_language(&self, language: &str) -> bool;

    /// Get the precision level of this provider for the given language.
    fn precision_level(&self, language: &str) -> PrecisionLevel;
}

/// Hybrid provider that selects the best available provider per language.
///
/// Currently delegates to RepoMap (tree-sitter AST-based) for all languages.
pub struct HybridRelationsProvider {
    repomap: repomap::RepoMapProvider,
}

impl HybridRelationsProvider {
    /// Create a new hybrid relations provider.
    pub fn new() -> Result<Self> {
        Ok(Self {
            repomap: repomap::RepoMapProvider::new(),
        })
    }
}

impl RelationsProvider for HybridRelationsProvider {
    fn extract_definitions(&self, file_info: &FileInfo) -> Result<Vec<Definition>> {
        self.repomap.extract_definitions(file_info)
    }

    fn extract_references(
        &self,
        file_info: &FileInfo,
        symbol_index: &HashMap<String, Vec<Definition>>,
    ) -> Result<Vec<Reference>> {
        self.repomap.extract_references(file_info, symbol_index)
    }

    fn supports_language(&self, language: &str) -> bool {
        self.repomap.supports_language(language)
    }

    fn precision_level(&self, language: &str) -> PrecisionLevel {
        self.repomap.precision_level(language)
    }
}

/// Configuration for relations extraction
#[derive(Debug, Clone)]
pub struct RelationsConfig {
    /// Whether relations extraction is enabled
    pub enabled: bool,
    /// Maximum call graph traversal depth
    pub max_call_depth: usize,
}

impl Default for RelationsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_call_depth: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_provider_creation() {
        let provider = HybridRelationsProvider::new().unwrap();
        assert!(provider.supports_language("Rust"));
        assert!(provider.supports_language("Python"));
        assert!(provider.supports_language("Unknown"));
    }

    #[test]
    fn test_precision_level() {
        let provider = HybridRelationsProvider::new().unwrap();
        assert_eq!(provider.precision_level("Rust"), PrecisionLevel::Medium);
        assert_eq!(provider.precision_level("Python"), PrecisionLevel::Medium);
    }

    #[test]
    fn test_relations_config_default() {
        let config = RelationsConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_call_depth, 3);
    }
}
