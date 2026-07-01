//! Print every entry in the feature-inventory manifest. Used by
//! `cargo xtask test-harness coverage` and as a quick sanity check during
//! manifest authoring.

use std::process::ExitCode;

fn main() -> ExitCode {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| rullama_test_harness::MANIFEST_PATH.to_string());

    let manifest = match rullama_test_harness::manifest::load(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to load manifest at {path}: {e:#}");
            return ExitCode::from(1);
        }
    };

    println!(
        "schema_version={}  entries={}",
        manifest.schema_version,
        manifest.entries.len()
    );
    for f in &manifest.entries {
        println!(
            "  [{}] {} ({}) — {} required_case(s)",
            f.crate_name.as_deref().unwrap_or("?"),
            f.feature_id,
            f.section,
            f.required_cases.len()
        );
    }
    ExitCode::SUCCESS
}
