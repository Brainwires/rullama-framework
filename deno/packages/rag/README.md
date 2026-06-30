# @rullama/rag

RAG client interface + code-analysis surface for the rullama.

Extracted from `@rullama/knowledge` in v0.11.0 to mirror Rust's
`rullama-rag` crate. Code analysis (symbol extraction, repo maps, call
graphs, reference tracking) lives alongside the RAG client because both share
embedding pipelines and storage in Rust; in Deno they're co-located here under
the `./code-analysis` sub-path export.

The actual indexing service is Rust-side (`rullama-rag` over LanceDB + ONNX +
tantivy). The Deno package ships only the client interface.
