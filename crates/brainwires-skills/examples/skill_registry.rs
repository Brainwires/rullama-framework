//! Skill Registry — SKILL.md parsing, discovery, and routing
//!
//! Demonstrates how to create SKILL.md files, discover them with SkillRegistry,
//! match user queries with SkillRouter, and load full skill instructions.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use brainwires_skills::{SkillRegistry, SkillRouter, SkillSource};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Setup — create temp dir with three SKILL.md files
    println!("=== 1. Setup: Creating SKILL.md files ===\n");

    let temp_dir = std::env::temp_dir().join("brainwires-skills-example");
    std::fs::create_dir_all(&temp_dir)?;

    let skills = [
        (
            "review-pr",
            "Reviews pull requests for code quality and best practices",
            "# PR Review Instructions\n\n\
             1. Check for code style violations\n\
             2. Look for security issues\n\
             3. Verify test coverage\n\
             4. Suggest improvements",
        ),
        (
            "deploy",
            "Deploys applications to staging or production environments",
            "# Deploy Instructions\n\n\
             1. Verify build passes\n\
             2. Run database migrations\n\
             3. Deploy to target environment\n\
             4. Run smoke tests",
        ),
        (
            "test-gen",
            "Generates unit tests for functions and modules",
            "# Test Generation Instructions\n\n\
             1. Analyze function signatures\n\
             2. Identify edge cases\n\
             3. Generate test stubs\n\
             4. Add assertions for expected behavior",
        ),
    ];

    for (name, description, instructions) in &skills {
        let content =
            format!("---\nname: {name}\ndescription: {description}\n---\n\n{instructions}\n");
        let path = temp_dir.join(format!("{name}.md"));
        std::fs::write(&path, &content)?;
        println!("  Created: {}", path.display());
    }

    // 2. Discovery — create SkillRegistry and discover from temp dir
    println!("\n=== 2. Discovery: Scanning for skills ===\n");

    let mut registry = SkillRegistry::new();
    let discovery_paths: Vec<(PathBuf, SkillSource)> =
        vec![(temp_dir.clone(), SkillSource::Personal)];
    registry.discover_from(&discovery_paths)?;

    println!("  Discovered {} skills", registry.len());

    // 3. Listing — show all discovered skills with metadata
    println!("\n=== 3. Listing: All discovered skills ===\n");

    for name in registry.list_skills() {
        let meta = registry.get_metadata(name).unwrap();
        println!("  /{} — {}", meta.name, meta.description);
        println!("    Source: {}", meta.source);
    }

    // 4. Routing — match user queries against skills
    println!("\n=== 4. Routing: Matching queries to skills ===\n");

    let shared_registry = Arc::new(RwLock::new(registry));
    let router = SkillRouter::new(Arc::clone(&shared_registry));

    let queries = [
        "review my pull request for quality issues",
        "deploy the app to production",
        "generate tests for this module",
        "completely unrelated cooking recipe",
    ];

    for query in &queries {
        let matches = router.match_skills(query).await;
        println!("  Query: \"{query}\"");
        if matches.is_empty() {
            println!("    No matches found");
        } else {
            for m in &matches {
                println!(
                    "    -> {} (confidence: {:.2}, source: {})",
                    m.skill_name, m.confidence, m.source
                );
            }
        }
        println!();
    }

    // 5. Format suggestions the way the CLI would show them
    println!("=== 5. Formatted suggestions ===\n");

    let matches = router.match_skills("review code quality").await;
    match router.format_suggestions(&matches) {
        Some(suggestion) => println!("  {suggestion}"),
        None => println!("  No suggestions"),
    }

    // 6. Load full skill — lazy-load instructions from disk
    println!("\n=== 6. Loading full skill content ===\n");

    let mut registry = shared_registry.write().await;
    let skill = registry.get_skill("review-pr")?;

    println!("  Skill: {}", skill.name());
    println!("  Description: {}", skill.description());
    println!("  Execution mode: {}", skill.execution_mode);
    println!("  Instructions:\n");
    for line in skill.instructions.lines() {
        println!("    {line}");
    }

    // Cleanup
    std::fs::remove_dir_all(&temp_dir)?;

    println!("\nDone.");
    Ok(())
}
