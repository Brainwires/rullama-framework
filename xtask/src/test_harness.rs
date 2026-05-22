//! `cargo xtask test-harness <subcommand>` — orchestration for the
//! Brainwires test harness.
//!
//! Subcommands:
//! - `coverage` — diff FEATURES.md against the feature-inventory manifest,
//!   fail with a copy-pasteable TOML stub on missing entries.
//!
//! Planned (later steps):
//! - `run [--tier=a|b|c|all]` — run the full harness.
//! - `deny-grep` — dependency-surface forbidden-pattern grep (Step 7).
//! - `baselines` — record / diff regression baselines (Step 12).

use std::path::PathBuf;
use std::process::ExitCode;

use serde::Deserialize;

/// FEATURES.md path relative to the workspace root.
const FEATURES_MD: &str = "FEATURES.md";

/// Feature-inventory manifest path, relative to the workspace root.
const MANIFEST_TOML: &str = "crates/brainwires-test-harness/tests/feature_inventory.toml";

/// Sections we never expect to cover (matches the awk generator in the
/// harness `Step 3` notes).
const EXCLUDED_TOP_SECTIONS: &[&str] = &["Table of Contents", "Extras & Standalone Binaries"];

pub fn dispatch(args: &[String]) -> ExitCode {
    match args.first().map(|s| s.as_str()) {
        Some("coverage") => coverage(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("deny-grep") => deny_grep::run(&args[1..]),
        Some("--help" | "-h") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("Unknown test-harness subcommand: {other}");
            print_help();
            ExitCode::FAILURE
        }
    }
}

mod deny_grep;

fn print_help() {
    println!("Usage: cargo xtask test-harness <subcommand>");
    println!();
    println!("Subcommands:");
    println!("  coverage         Diff FEATURES.md against feature_inventory.toml");
    println!("  run [flags]      Run harness cases via the run-harness binary");
    println!("  deny-grep        Scan crates/ against .deny-grep.toml forbidden patterns");
    println!();
    println!("Run flags (forwarded to run-harness):");
    println!("  --tier=a|b|c|all   Restrict to one tier (default: all)");
    println!("  --trials=N         Trials per case (default: 1)");
    println!("  --filter=<s>       Only cases whose name contains <s>");
    println!("  --json             Emit JSON output");
}

fn run(args: &[String]) -> ExitCode {
    let workspace_root = match find_workspace_root() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("could not locate workspace root: {e}");
            return ExitCode::FAILURE;
        }
    };
    let status = std::process::Command::new("cargo")
        .current_dir(&workspace_root)
        .args([
            "run",
            "--quiet",
            "-p",
            "brainwires-test-harness",
            "--bin",
            "run-harness",
            "--",
        ])
        .args(args)
        .status();
    match status {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("failed to invoke cargo run: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── coverage ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ManifestStub {
    #[serde(default, rename = "feature")]
    entries: Vec<FeatureStub>,
}

#[derive(Debug, Deserialize)]
struct FeatureStub {
    section: String,
    feature_id: String,
    // Reserved for the dangling-Rust-path check (Step 9+) — wired through
    // brainwires-test-harness's `inventory` registry once Tier-A cases land.
    #[serde(default)]
    #[allow(dead_code)]
    required_cases: Vec<String>,
    #[serde(default)]
    coverage_via: Option<String>,
}

fn coverage(args: &[String]) -> ExitCode {
    let _crit_min: Option<usize> = parse_crit_min(args);

    let workspace_root = match find_workspace_root() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("could not locate workspace root: {e}");
            return ExitCode::FAILURE;
        }
    };

    let features_md = workspace_root.join(FEATURES_MD);
    let manifest_toml = workspace_root.join(MANIFEST_TOML);

    let raw_features = match std::fs::read_to_string(&features_md) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to read {}: {e}", features_md.display());
            return ExitCode::FAILURE;
        }
    };
    let raw_manifest = match std::fs::read_to_string(&manifest_toml) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to read {}: {e}", manifest_toml.display());
            return ExitCode::FAILURE;
        }
    };

    let manifest: ManifestStub = match toml::from_str(&raw_manifest) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to parse manifest: {e}");
            return ExitCode::FAILURE;
        }
    };

    let headings = extract_headings(&raw_features);
    let manifest_sections: std::collections::HashSet<&str> =
        manifest.entries.iter().map(|e| e.section.as_str()).collect();

    let mut missing: Vec<&Heading> = Vec::new();
    for h in &headings {
        if !manifest_sections.contains(h.text.as_str()) {
            missing.push(h);
        }
    }

    // Dangling coverage_via references (alias points to a non-existent feature_id).
    let feature_ids: std::collections::HashSet<&str> = manifest
        .entries
        .iter()
        .map(|e| e.feature_id.as_str())
        .collect();
    let mut dangling_alias: Vec<(&str, &str)> = Vec::new();
    for e in &manifest.entries {
        if let Some(target) = &e.coverage_via
            && !feature_ids.contains(target.as_str())
        {
            dangling_alias.push((e.feature_id.as_str(), target.as_str()));
        }
    }

    let direct: usize = manifest
        .entries
        .iter()
        .filter(|e| !e.required_cases.is_empty())
        .count();
    let aliased: usize = manifest
        .entries
        .iter()
        .filter(|e| e.required_cases.is_empty() && e.coverage_via.is_some())
        .count();
    println!(
        "Coverage check: {} FEATURES.md headings, {} manifest entries  ({} direct cases, {} aliased)",
        headings.len(),
        manifest.entries.len(),
        direct,
        aliased,
    );

    if missing.is_empty() && dangling_alias.is_empty() {
        println!("OK: every FEATURES.md heading has a manifest entry.");
        return ExitCode::SUCCESS;
    }

    if !missing.is_empty() {
        println!();
        println!("✗ {} heading(s) missing from manifest:", missing.len());
        for h in &missing {
            println!("  - FEATURES.md:{} \"{}\"", h.line, h.text);
        }
        println!();
        println!("Add the following stubs to {}:", manifest_toml.display());
        println!();
        for h in &missing {
            print_stub(h);
        }
    }

    if !dangling_alias.is_empty() {
        println!();
        println!(
            "✗ {} dangling coverage_via reference(s):",
            dangling_alias.len()
        );
        for (from, to) in &dangling_alias {
            println!(
                "  - feature_id=\"{from}\" coverage_via=\"{to}\" (no feature_id matches)"
            );
        }
    }

    ExitCode::FAILURE
}

#[derive(Debug)]
struct Heading {
    line: usize,
    text: String,
}

fn extract_headings(raw: &str) -> Vec<Heading> {
    let mut out = Vec::new();
    let mut skip_top = false;
    for (i, line) in raw.lines().enumerate() {
        // Lines are 1-indexed in the report.
        let line_no = i + 1;
        if let Some(rest) = line.strip_prefix("## ") {
            // Top-level — reset skip state, decide if this section is excluded.
            skip_top = EXCLUDED_TOP_SECTIONS.contains(&rest);
            if !skip_top {
                out.push(Heading {
                    line: line_no,
                    text: rest.to_string(),
                });
            }
            continue;
        }
        if skip_top {
            continue;
        }
        if let Some(rest) = line.strip_prefix("### ") {
            out.push(Heading {
                line: line_no,
                text: rest.to_string(),
            });
        } else if let Some(rest) = line.strip_prefix("#### ") {
            out.push(Heading {
                line: line_no,
                text: rest.to_string(),
            });
        }
    }
    out
}

fn print_stub(h: &Heading) {
    let fid: String = h
        .text
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .to_ascii_lowercase();
    // Collapse repeated underscores + trim.
    let mut prev_underscore = false;
    let mut compact = String::with_capacity(fid.len());
    for c in fid.chars() {
        if c == '_' {
            if !prev_underscore {
                compact.push('_');
            }
            prev_underscore = true;
        } else {
            compact.push(c);
            prev_underscore = false;
        }
    }
    let fid = compact.trim_matches('_').to_string();
    println!("[[feature]]");
    println!("section = \"{}\"", h.text.replace('"', "\\\""));
    println!("feature_id = \"{fid}\"");
    println!("crate_name = \"TODO\"");
    println!("description = \"TODO\"");
    println!("required_cases = []");
    println!();
}

fn parse_crit_min(args: &[String]) -> Option<usize> {
    for arg in args {
        if let Some(v) = arg.strip_prefix("--crit-min=") {
            return v.parse().ok();
        }
    }
    None
}

fn find_workspace_root() -> std::io::Result<PathBuf> {
    let mut cwd = std::env::current_dir()?;
    loop {
        let candidate = cwd.join("Cargo.toml");
        if candidate.exists() {
            let content = std::fs::read_to_string(&candidate)?;
            if content.contains("[workspace]") {
                return Ok(cwd);
            }
        }
        if !cwd.pop() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no workspace Cargo.toml found",
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_headings_skips_excluded_section() {
        let md = "\
## Real Feature
content
## Table of Contents
- foo
- bar
### Sub Under TOC
## Another Real Feature
### Sub Under Real Feature
";
        let h = extract_headings(md);
        let texts: Vec<&str> = h.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(
            texts,
            vec!["Real Feature", "Another Real Feature", "Sub Under Real Feature"]
        );
    }

    #[test]
    fn extract_headings_includes_h4() {
        let md = "## Top\n### Sub\n#### SubSub\n";
        let h = extract_headings(md);
        assert_eq!(h.len(), 3);
        assert_eq!(h[2].text, "SubSub");
    }
}
