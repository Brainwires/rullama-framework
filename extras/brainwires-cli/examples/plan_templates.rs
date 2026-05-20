//! Example: TemplateStore — reusable plan templates with variable substitution
//!
//! Demonstrates creating a `TemplateStore`, registering templates that contain
//! `{{variable}}` placeholders, searching/listing templates, and instantiating
//! them with concrete values.
//!
//! Run: cargo run -p brainwires-cli --example plan_templates

use std::collections::HashMap;

use anyhow::Result;
use brainwires_stores::{PlanTemplate, TemplateStore};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create a template store in a temporary directory
    let tmp_dir = tempfile::tempdir()?;
    let store = TemplateStore::new(tmp_dir.path())?;
    println!("TemplateStore created at {:?}\n", tmp_dir.path());

    // 2. Register a feature-implementation template
    let feature_template = PlanTemplate::new(
        "Feature Implementation".into(),
        "Step-by-step plan for implementing a new feature".into(),
        "\
# Feature: {{feature_name}}

## 1. Design
- Define the public API for {{component}}
- Write interface types in `src/{{module}}/types.rs`

## 2. Implementation
- Implement {{feature_name}} logic in `src/{{module}}/mod.rs`
- Add error handling for {{component}} edge cases

## 3. Testing
- Unit tests for {{component}} in `tests/{{module}}_test.rs`
- Integration test covering the {{feature_name}} happy path

## 4. Documentation
- Add rustdoc comments to all public items in {{module}}
"
        .into(),
    )
    .with_category("feature".into())
    .with_tags(vec!["rust".into(), "implementation".into()]);

    store.save(&feature_template)?;
    println!("Saved template: {}", feature_template.name);
    println!("  Variables: {:?}", feature_template.variables);
    println!("  Category:  {:?}", feature_template.category);
    println!();

    // 3. Register a bugfix template
    let bugfix_template = PlanTemplate::new(
        "Bug Fix Workflow".into(),
        "Systematic approach to diagnosing and fixing bugs".into(),
        "\
# Bug Fix: {{bug_title}}

## 1. Reproduce
- Reproduce {{bug_title}} using the steps from {{issue_tracker}} issue
- Capture failing test output

## 2. Root Cause
- Trace execution path in {{affected_module}}
- Identify the root cause

## 3. Fix
- Apply fix in `src/{{affected_module}}/`
- Add regression test for {{bug_title}}

## 4. Verify
- Run full test suite: `cargo test -p {{crate_name}}`
- Confirm the {{issue_tracker}} issue is resolved
"
        .into(),
    )
    .with_category("bugfix".into())
    .with_tags(vec!["debugging".into(), "testing".into()]);

    store.save(&bugfix_template)?;
    println!("Saved template: {}", bugfix_template.name);
    println!("  Variables: {:?}", bugfix_template.variables);
    println!();

    // 4. List all templates (sorted by usage count, then name)
    let all = store.list()?;
    println!("All templates ({}):", all.len());
    for t in &all {
        println!(
            "  [{}] {} — {} (used {} times)",
            t.template_id.chars().take(8).collect::<String>(),
            t.name,
            t.description,
            t.usage_count,
        );
    }
    println!();

    // 5. Search templates by keyword
    let results = store.search("bug")?;
    println!("Search for \"bug\": {} result(s)", results.len());
    for t in &results {
        println!("  {} — {}", t.name, t.description);
    }
    println!();

    // 6. List templates by category
    let features = store.list_by_category("feature")?;
    println!("Category \"feature\": {} template(s)", features.len());
    println!();

    // 7. Instantiate the feature template with concrete values
    let mut substitutions = HashMap::new();
    substitutions.insert("feature_name".into(), "message encryption".into());
    substitutions.insert("component".into(), "EncryptionService".into());
    substitutions.insert("module".into(), "encryption".into());

    let instantiated = feature_template.instantiate(&substitutions);
    println!("Instantiated plan:\n{}", instantiated);

    // 8. Mark a template as used and verify the count updates
    store.mark_used(&feature_template.template_id)?;
    store.mark_used(&feature_template.template_id)?;
    let updated = store.get(&feature_template.template_id)?;
    if let Some(t) = updated {
        println!("Template \"{}\" usage count: {}", t.name, t.usage_count);
    }

    println!("\nDone.");
    Ok(())
}
