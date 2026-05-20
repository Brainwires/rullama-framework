# brainwires-rag

[![Crates.io](https://img.shields.io/crates/v/brainwires-rag.svg)](https://crates.io/crates/brainwires-rag)
[![Documentation](https://docs.rs/brainwires-rag/badge.svg)](https://docs.rs/brainwires-rag)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

Codebase indexing and hybrid retrieval (vector + BM25) for Brainwires
agents. Includes AST-aware chunking via `tree-sitter` (12 languages),
Git history search, and reranking via spectral diversity / cross-encoder.

## What lives here

- `rag::client::RagClient` — hybrid retrieval over a LanceDB-backed index.
- `rag::indexer` — file walking, AST-based chunking (Rust, Python,
  JavaScript, TypeScript, Go, Java, Swift, C, C++, C#, Ruby, PHP).
- `rag::embedding::FastEmbedManager` — local FastEmbed embeddings
  (`all-MiniLM-L6-v2` by default).
- `rag::git` — Git-aware indexing + commit history search.
- `rag::documents` — PDF / DOCX / Markdown / plaintext ingestion (PDF
  behind the `pdf-extract-feature` feature).

Two RAG-internal subdomains travel inside this crate (no public API
extracted because nothing outside the RAG client uses them):

- `spectral` — log-determinant diversity reranking + cross-encoder reranker.
- `code_analysis` — AST symbol / definition / reference graphs.

## Features

| Feature | Notes |
|---|---|
| (default) | RagClient + indexer + embeddings + git + spectral + code_analysis. Tree-sitter language parsers always-on. |
| `documents` | DOCX / Markdown ingestion (zip-based) |
| `pdf-extract-feature` | PDF text extraction via `pdf-extract` |
| `qdrant-backend` | Forward `qdrant-backend` to `brainwires-storage` |

## Heavy deps

This crate pulls `lancedb`, `tantivy`, `git2`, `tree-sitter` + 12 grammar
crates, `rmcp`, `rayon`, and friends. Consumers that only want
`brainwires-knowledge` (BKS / PKS / brain client) or
`brainwires-prompting` should depend on those crates directly to avoid
the RAG dep weight.

## Usage

```toml
[dependencies]
brainwires-rag = "0.11"
```

```rust,ignore
use brainwires_rag::{RagClient, IndexRequest, QueryRequest};

let client = RagClient::new(/* config */).await?;
client.index(IndexRequest { /* ... */ }).await?;
let results = client.query(QueryRequest { /* ... */ }).await?;
```

## See also

- [`brainwires-knowledge`](https://crates.io/crates/brainwires-knowledge) —
  knowledge graphs / BKS / PKS / brain client (sibling).
- [`brainwires-prompting`](https://crates.io/crates/brainwires-prompting) —
  adaptive prompting techniques (sibling).
- [`brainwires`](https://crates.io/crates/brainwires) — umbrella facade
  with `rag` feature.

## License

MIT OR Apache-2.0
