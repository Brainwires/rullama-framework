//! Code indexing, file walking, and chunking strategies
//!
//! Provides functionality to walk directories, detect languages, parse AST,
//! and chunk code files into semantically meaningful units for embedding.

mod ast_parser;
mod chunker;
mod file_info;
mod file_walker;
mod language;
#[cfg(feature = "pdf-extract-feature")]
mod pdf_extractor;

pub use ast_parser::AstParser;
pub use chunker::{ChunkStrategy, Chunker, CodeChunker};
pub use file_info::FileInfo;
pub use file_walker::FileWalker;
pub use language::detect_language;
#[cfg(feature = "pdf-extract-feature")]
pub use pdf_extractor::extract_pdf_to_markdown;

use brainwires_core::ChunkMetadata;

/// Represents a code chunk ready for embedding
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// The actual source code content of this chunk
    pub content: String,
    /// Metadata about this chunk (file path, line numbers, language, etc.)
    pub metadata: ChunkMetadata,
}
