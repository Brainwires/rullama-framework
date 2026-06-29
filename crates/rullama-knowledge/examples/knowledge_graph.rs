//! Knowledge Graph — Entity Store, Relationships, and Semantic Search
//!
//! Demonstrates how to use the knowledge module to build an in-memory
//! entity graph, capture thoughts via `BrainClient`, and search the
//! knowledge base semantically.
//!
//! Run:
//! ```sh
//! cargo run -p rullama-knowledge --example knowledge_graph --features knowledge
//! ```

use rullama_knowledge::knowledge::entity::{
    EntityStore, EntityType, ExtractionResult, Relationship,
};
use rullama_knowledge::knowledge::relationship_graph::RelationshipGraph;
use rullama_knowledge::knowledge::thought::{Thought, ThoughtCategory};

fn main() {
    println!("=== Brainwires Knowledge Graph Example ===\n");

    // ── Step 1: Build an EntityStore from extracted entities ──────────
    println!("--- Step 1: Populate the Entity Store ---\n");

    let mut store = EntityStore::new();

    // Simulate extraction results from three conversation messages
    let extraction_1 = ExtractionResult {
        entities: vec![
            ("main.rs".into(), EntityType::File),
            ("Config".into(), EntityType::Type),
            ("process_request".into(), EntityType::Function),
        ],
        relationships: vec![
            Relationship::Contains {
                container: "main.rs".into(),
                contained: "process_request".into(),
            },
            Relationship::References {
                from: "process_request".into(),
                to: "Config".into(),
            },
        ],
    };

    let extraction_2 = ExtractionResult {
        entities: vec![
            ("server.rs".into(), EntityType::File),
            ("handle_connection".into(), EntityType::Function),
            ("Config".into(), EntityType::Type),
        ],
        relationships: vec![
            Relationship::Contains {
                container: "server.rs".into(),
                contained: "handle_connection".into(),
            },
            Relationship::DependsOn {
                dependent: "handle_connection".into(),
                dependency: "Config".into(),
            },
            Relationship::CoOccurs {
                entity_a: "handle_connection".into(),
                entity_b: "process_request".into(),
                message_id: "msg-2".into(),
            },
        ],
    };

    let extraction_3 = ExtractionResult {
        entities: vec![
            ("DatabaseError".into(), EntityType::Error),
            ("process_request".into(), EntityType::Function),
        ],
        relationships: vec![Relationship::Modifies {
            modifier: "process_request".into(),
            modified: "DatabaseError".into(),
            change_type: "handles".into(),
        }],
    };

    store.add_extraction(extraction_1, "msg-1", 1000);
    store.add_extraction(extraction_2, "msg-2", 2000);
    store.add_extraction(extraction_3, "msg-3", 3000);

    // Print entity stats
    let stats = store.stats();
    println!("Total entities: {}", stats.total_entities);
    println!("Total relationships: {}", stats.total_relationships);
    println!("Entities by type:");
    for (entity_type, count) in &stats.entities_by_type {
        println!("  {}: {}", entity_type, count);
    }
    println!();

    // ── Step 2: Query the EntityStore ─────────────────────────────────
    println!("--- Step 2: Query the Entity Store ---\n");

    // Find the most-mentioned entities
    let top = store.get_top_entities(3);
    println!("Top 3 entities by mention count:");
    for entity in &top {
        println!(
            "  {} ({:?}) — {} mentions",
            entity.name, entity.entity_type, entity.mention_count
        );
    }
    println!();

    // Find all functions
    let functions = store.get_by_type(&EntityType::Function);
    println!(
        "Functions: {:?}",
        functions.iter().map(|e| &e.name).collect::<Vec<_>>()
    );

    // Find entities related to "process_request"
    let related = store.get_related("process_request");
    println!("Entities related to 'process_request': {:?}", related);
    println!();

    // ── Step 3: Build a RelationshipGraph and traverse it ────────────
    println!("--- Step 3: Build a Relationship Graph ---\n");

    let mut graph = RelationshipGraph::from_entity_store(&store);

    // Manually add edges from the relationships
    for rel in store.all_relationships() {
        graph.add_relationship(rel);
    }

    let graph_stats = graph.stats();
    println!(
        "Graph: {} nodes, {} edges",
        graph_stats.node_count, graph_stats.edge_count
    );
    println!();

    // Find neighbors of a node
    if let Some(node) = graph.get_node("Config") {
        println!(
            "Node 'Config': type={:?}, importance={:.2}",
            node.entity_type, node.importance
        );
        let neighbors = graph.get_neighbors("Config");
        println!(
            "  Neighbors: {:?}",
            neighbors.iter().map(|n| &n.entity_name).collect::<Vec<_>>()
        );
    }

    // Find shortest path between two entities
    if let Some(path) = graph.find_path("main.rs", "DatabaseError") {
        println!("Path from main.rs → DatabaseError: {}", path.join(" → "));
    } else {
        println!("No path from main.rs → DatabaseError");
    }
    println!();

    // ── Step 4: Impact analysis ──────────────────────────────────────
    println!("--- Step 4: Impact Analysis ---\n");

    let impacts = graph.get_impact_set("Config", 3);
    println!("Entities impacted by changes to 'Config':");
    for impact in &impacts {
        println!(
            "  {} ({:?}) — distance={}, impact={:.3}",
            impact.name, impact.entity_type, impact.distance, impact.impact_score
        );
    }
    println!();

    // ── Step 5: Entity context retrieval ──────────────────────────────
    println!("--- Step 5: Entity Context ---\n");

    let context = graph.get_entity_context("process_request", 2);
    println!("Context for 'process_request':");
    println!("  Related message IDs: {:?}", context.message_ids);
    for rel in &context.related_entities {
        println!(
            "  → {} ({:?}, {:?}) relevance={:.3}, distance={}",
            rel.name, rel.entity_type, rel.relationship, rel.relevance, rel.distance
        );
    }
    println!();

    // ── Step 6: Contradiction detection ──────────────────────────────
    println!("--- Step 6: Contradiction Detection ---\n");

    // Add a conflicting definition to trigger contradiction detection
    let conflicting = ExtractionResult {
        entities: vec![],
        relationships: vec![Relationship::Modifies {
            modifier: "process_request".into(),
            modified: "DatabaseError".into(),
            change_type: "ignores".into(), // conflicts with "handles"
        }],
    };
    store.add_extraction(conflicting, "msg-4", 4000);

    let contradictions = store.pending_contradictions();
    if contradictions.is_empty() {
        println!("No contradictions detected.");
    } else {
        println!("Contradictions detected:");
        for c in contradictions {
            println!(
                "  {:?} on '{}': existing='{}', new='{}'",
                c.kind, c.subject, c.existing_context, c.new_context
            );
        }
    }
    println!();

    // ── Step 7: Demonstrate Thought creation (without BrainClient) ───
    println!("--- Step 7: Thought Construction ---\n");

    let thought = Thought::new("Decided to use PostgreSQL for the auth service".into())
        .with_category(ThoughtCategory::Decision)
        .with_tags(vec![
            "database".into(),
            "auth".into(),
            "architecture".into(),
        ])
        .with_importance(0.9);

    println!("Thought: {}", thought.content);
    println!("  ID:         {}", thought.id);
    println!("  Category:   {}", thought.category);
    println!("  Tags:       {:?}", thought.tags);
    println!("  Importance: {:.1}", thought.importance);
    println!("  Created:    {}", thought.created_at);
    println!();

    // List all categories
    println!("Available thought categories:");
    for cat in ThoughtCategory::ALL {
        println!("  - {}", cat.as_str());
    }

    println!("\n=== Done ===");
}
