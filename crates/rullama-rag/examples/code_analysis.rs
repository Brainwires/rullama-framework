//! Code Analysis — Definitions, References, and Call Graphs
//!
//! Demonstrates how to use the code analysis module to extract symbol
//! definitions, find references, and build call graph structures from
//! source code using AST-based (tree-sitter) parsing.
//!
//! Run:
//! ```sh
//! cargo run -p rullama-knowledge --example code_analysis --features code-analysis,rag
//! ```

use std::collections::HashMap;

use rullama_rag::code_analysis::types::{
    CallEdge, CallGraphNode, DefinitionResult, PrecisionLevel, SymbolId,
};
use rullama_rag::code_analysis::{
    Definition, HybridRelationsProvider, RelationsProvider, SymbolKind,
};
use rullama_rag::rag::indexer::FileInfo;

fn main() -> anyhow::Result<()> {
    println!("=== Brainwires Code Analysis Example ===\n");

    // ── Step 1: Create the hybrid relations provider ─────────────────
    println!("--- Step 1: Initialize Relations Provider ---\n");

    let provider = HybridRelationsProvider::new()?;

    // Check language support and precision levels
    let languages = ["Rust", "Python", "JavaScript", "TypeScript", "Go", "Java"];
    println!("{:<14} {:<10} Precision", "Language", "Supported");
    println!("{:-<44}", "");
    for lang in &languages {
        let supported = provider.supports_language(lang);
        let precision = provider.precision_level(lang);
        println!(
            "{:<14} {:<10} {}",
            lang,
            if supported { "yes" } else { "no" },
            precision.description(),
        );
    }
    println!();

    // ── Step 2: Extract definitions from a Rust source snippet ───────
    println!("--- Step 2: Extract Definitions ---\n");

    let rust_source = r#"
use std::collections::HashMap;

/// Configuration for the application.
pub struct Config {
    pub host: String,
    pub port: u16,
    pub debug: bool,
}

impl Config {
    /// Create a new Config with defaults.
    pub fn new() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8080,
            debug: false,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.port == 0 {
            return Err("port must be non-zero".into());
        }
        Ok(())
    }
}

/// Process an incoming request using the given config.
pub fn process_request(config: &Config, data: &str) -> String {
    format!("Processed on {}:{}: {}", config.host, config.port, data)
}

/// Start the main server loop.
fn main() {
    let config = Config::new();
    config.validate().expect("invalid config");
    let result = process_request(&config, "hello world");
    println!("{}", result);
}
"#;

    let file_info = FileInfo {
        path: std::path::PathBuf::from("/demo/src/main.rs"),
        relative_path: "src/main.rs".to_string(),
        root_path: "/demo".to_string(),
        project: Some("demo".to_string()),
        extension: Some("rs".to_string()),
        language: Some("Rust".to_string()),
        content: rust_source.to_string(),
        hash: "demo-hash-001".to_string(),
    };

    let definitions = provider.extract_definitions(&file_info)?;

    println!("Found {} definitions in src/main.rs:\n", definitions.len());
    println!(
        "{:<6} {:<22} {:<12} {:<10} {:<10} Signature",
        "Line", "Name", "Kind", "Visibility", "Has Docs"
    );
    println!("{:-<90}", "");

    for def in &definitions {
        let sig_preview = if def.signature.len() > 30 {
            format!("{}...", &def.signature[..30])
        } else {
            def.signature.clone()
        };

        println!(
            "{:<6} {:<22} {:<12} {:<10} {:<10} {}",
            def.symbol_id.start_line,
            def.symbol_id.name,
            def.symbol_id.kind.display_name(),
            format!("{:?}", def.visibility),
            if def.doc_comment.is_some() {
                "yes"
            } else {
                "no"
            },
            sig_preview
        );
    }
    println!();

    // ── Step 3: Build a symbol index and find references ─────────────
    println!("--- Step 3: Find References ---\n");

    // Build a symbol index from the definitions we extracted
    let mut symbol_index: HashMap<String, Vec<Definition>> = HashMap::new();
    for def in &definitions {
        symbol_index
            .entry(def.symbol_id.name.clone())
            .or_default()
            .push(def.clone());
    }

    let references = provider.extract_references(&file_info, &symbol_index)?;

    println!("Found {} references:\n", references.len());
    println!("{:<6} {:<10} {:<16} Target", "Line", "Col", "Kind");
    println!("{:-<60}", "");

    for reference in &references {
        println!(
            "{:<6} {:<10} {:<16} {}",
            reference.start_line,
            reference.start_col,
            format!("{:?}", reference.reference_kind),
            reference.target_symbol_id,
        );
    }
    println!();

    // ── Step 4: Demonstrate SymbolId and DefinitionResult types ──────
    println!("--- Step 4: Symbol Identification ---\n");

    // Create symbol IDs for cross-referencing
    let sym_config = SymbolId::new("src/main.rs", "Config", SymbolKind::Struct, 5, 0);
    let sym_new = SymbolId::new("src/main.rs", "new", SymbolKind::Method, 13, 4);
    let sym_process = SymbolId::new(
        "src/main.rs",
        "process_request",
        SymbolKind::Function,
        33,
        0,
    );

    println!("Symbol IDs (for storage/lookup):");
    println!("  Config:          {}", sym_config.to_storage_id());
    println!("  Config::new:     {}", sym_new.to_storage_id());
    println!("  process_request: {}", sym_process.to_storage_id());
    println!();

    // Convert a Definition to a DefinitionResult (the MCP-facing type)
    if let Some(first_def) = definitions.first() {
        let result = DefinitionResult::from(first_def);
        println!("DefinitionResult for '{}':", result.name);
        println!("  File:      {}", result.file_path);
        println!("  Kind:      {:?}", result.kind);
        println!("  Lines:     {}-{}", result.start_line, result.end_line);
        println!("  Signature: {}", result.signature);
        if let Some(ref doc) = result.doc_comment {
            println!("  Docs:      {}", doc);
        }
    }
    println!();

    // ── Step 5: Demonstrate CallEdge and CallGraphNode types ─────────
    println!("--- Step 5: Call Graph Structures ---\n");

    // Build a small call graph manually to demonstrate the data model
    let call_edges = vec![
        CallEdge {
            caller_id: "src/main.rs:main:39:0".to_string(),
            callee_id: "src/main.rs:new:13:4".to_string(),
            call_site_file: "src/main.rs".to_string(),
            call_site_line: 40,
            call_site_col: 18,
        },
        CallEdge {
            caller_id: "src/main.rs:main:39:0".to_string(),
            callee_id: "src/main.rs:validate:23:4".to_string(),
            call_site_file: "src/main.rs".to_string(),
            call_site_line: 41,
            call_site_col: 11,
        },
        CallEdge {
            caller_id: "src/main.rs:main:39:0".to_string(),
            callee_id: "src/main.rs:process_request:33:0".to_string(),
            call_site_file: "src/main.rs".to_string(),
            call_site_line: 42,
            call_site_col: 18,
        },
    ];

    println!("Call edges from main():");
    for edge in &call_edges {
        println!(
            "  main (line {}) → {} at {}:{}",
            edge.call_site_line,
            edge.callee_id.split(':').nth(1).unwrap_or("?"),
            edge.call_site_file,
            edge.call_site_line
        );
    }
    println!();

    // Build a CallGraphNode tree
    let call_graph = CallGraphNode {
        name: "main".to_string(),
        kind: SymbolKind::Function,
        file_path: "src/main.rs".to_string(),
        line: 39,
        children: vec![
            CallGraphNode {
                name: "Config::new".to_string(),
                kind: SymbolKind::Method,
                file_path: "src/main.rs".to_string(),
                line: 13,
                children: vec![],
            },
            CallGraphNode {
                name: "Config::validate".to_string(),
                kind: SymbolKind::Method,
                file_path: "src/main.rs".to_string(),
                line: 23,
                children: vec![],
            },
            CallGraphNode {
                name: "process_request".to_string(),
                kind: SymbolKind::Function,
                file_path: "src/main.rs".to_string(),
                line: 33,
                children: vec![],
            },
        ],
    };

    println!("Call graph tree:");
    print_call_tree(&call_graph, 0);
    println!();

    // ── Step 6: Precision levels and configuration ───────────────────
    println!("--- Step 6: Configuration ---\n");

    let config = rullama_rag::code_analysis::RelationsConfig::default();
    println!("Default RelationsConfig:");
    println!("  enabled:        {}", config.enabled);
    println!("  max_call_depth: {}", config.max_call_depth);
    println!();

    println!("Precision levels:");
    for level in [PrecisionLevel::Medium, PrecisionLevel::Low] {
        println!("  {:?} — {}", level, level.description());
    }

    println!("\n=== Done ===");
    Ok(())
}

/// Recursively print a call graph tree with indentation.
fn print_call_tree(node: &CallGraphNode, depth: usize) {
    let indent = "  ".repeat(depth);
    println!(
        "{}{} ({}) — {}:{}",
        indent,
        node.name,
        node.kind.display_name(),
        node.file_path,
        node.line
    );
    for child in &node.children {
        print_call_tree(child, depth + 1);
    }
}
