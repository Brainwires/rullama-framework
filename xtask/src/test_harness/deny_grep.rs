//! `cargo xtask test-harness deny-grep` — static-source forbidden-pattern check.
//!
//! Reads `.deny-grep.toml` at the workspace root, scans each rule's `include`
//! glob set, and fails with file:line locations on any match outside the
//! `allow_files` list. Used to enforce invariants that aren't easy (or
//! efficient) to capture as runtime Tier-B cases — e.g. "no `format!` in SQL
//! paths", "no `insecure_skip_verify`".

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use regex::Regex;
use serde::Deserialize;

const RULES_FILE: &str = ".deny-grep.toml";

#[derive(Debug, Deserialize)]
struct Ruleset {
    #[serde(default, rename = "rule")]
    rules: Vec<Rule>,
}

#[derive(Debug, Deserialize)]
struct Rule {
    id: String,
    description: String,
    include: Vec<String>,
    forbid_regex: String,
    #[serde(default)]
    allow_files: Vec<String>,
}

pub fn run(_args: &[String]) -> ExitCode {
    let workspace_root = match super::find_workspace_root() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("could not locate workspace root: {e}");
            return ExitCode::FAILURE;
        }
    };

    let rules_path = workspace_root.join(RULES_FILE);
    let raw = match std::fs::read_to_string(&rules_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to read {}: {e}", rules_path.display());
            return ExitCode::FAILURE;
        }
    };

    let ruleset: Ruleset = match toml::from_str(&raw) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to parse {}: {e}", rules_path.display());
            return ExitCode::FAILURE;
        }
    };

    println!("deny-grep: {} rule(s)", ruleset.rules.len());

    let mut total_violations: usize = 0;

    for rule in &ruleset.rules {
        let re = match Regex::new(&rule.forbid_regex) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("rule {}: invalid regex: {e}", rule.id);
                return ExitCode::FAILURE;
            }
        };

        let allow_set: std::collections::HashSet<PathBuf> = rule
            .allow_files
            .iter()
            .map(|p| workspace_root.join(p))
            .collect();

        let mut files = Vec::new();
        for include in &rule.include {
            let pattern = workspace_root.join(include);
            match glob::glob(pattern.to_str().unwrap_or(include)) {
                Ok(paths) => {
                    for p in paths.flatten() {
                        if p.is_file() {
                            files.push(p);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("rule {}: bad include glob {include:?}: {e}", rule.id);
                    return ExitCode::FAILURE;
                }
            }
        }
        files.sort();
        files.dedup();

        let mut violations: Vec<(PathBuf, usize, String)> = Vec::new();
        for path in &files {
            if allow_set.contains(path) {
                continue;
            }
            let body = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue, // ignore unreadable files (binaries etc.)
            };
            for (i, line) in body.lines().enumerate() {
                if re.is_match(line) {
                    violations.push((path.clone(), i + 1, line.to_string()));
                }
            }
        }

        if violations.is_empty() {
            println!("  ✓ {} — {}", rule.id, rule.description);
        } else {
            println!(
                "  ✗ {} — {}  ({} violation(s))",
                rule.id,
                rule.description,
                violations.len()
            );
            for (path, line, content) in &violations {
                let rel = relative_to(path, &workspace_root);
                let snippet = content.trim();
                let trimmed = if snippet.len() > 100 {
                    format!("{}…", &snippet[..100])
                } else {
                    snippet.to_string()
                };
                println!("      {}:{}  {}", rel.display(), line, trimmed);
            }
            total_violations += violations.len();
        }
    }

    println!();
    if total_violations == 0 {
        println!(
            "deny-grep: OK ({} rule(s), no violations)",
            ruleset.rules.len()
        );
        ExitCode::SUCCESS
    } else {
        println!("deny-grep: {total_violations} violation(s)");
        ExitCode::FAILURE
    }
}

fn relative_to(path: &Path, root: &Path) -> PathBuf {
    path.strip_prefix(root)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|_| path.to_path_buf())
}
