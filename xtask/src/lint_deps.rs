//! `cargo xtask lint-deps` — enforce the framework / extras boundary.
//!
//! The rule (see `docs/adr/ADR-0004-framework-extras-boundary.md`):
//!
//! - `crates/X → crates/Y`  ✅
//! - `extras/X → crates/Y`  ✅
//! - `crates/X → extras/Y`  ❌ — the framework cannot depend on its consumers.
//! - `extras/X → extras/Y`  ❌ — extras are siblings, not a hierarchy.
//!
//! Detection:
//!
//! 1. Build a `package_name → tier` map by walking every `Cargo.toml` under
//!    `crates/` and `extras/`. `tier` is `Crates` or `Extras`.
//! 2. For each `Cargo.toml`, scan `[dependencies]`, `[dev-dependencies]`,
//!    and `[build-dependencies]`. For each dep:
//!    - If it has `path = "..."`, classify the path's tier directly.
//!    - Else if it has `workspace = true`, look up the workspace
//!      `[workspace.dependencies]` table and classify by that path.
//! 3. Report any forbidden arrow.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use toml_edit::{DocumentMut, Item, Value};
use walkdir::WalkDir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    Crates,
    Extras,
}

impl Tier {
    fn name(self) -> &'static str {
        match self {
            Tier::Crates => "crates",
            Tier::Extras => "extras",
        }
    }
}

#[derive(Debug)]
struct PackageInfo {
    tier: Tier,
    cargo_toml: PathBuf,
}

#[derive(Debug)]
struct Violation {
    consumer_pkg: String,
    consumer_tier: Tier,
    consumer_cargo_toml: PathBuf,
    dep_name: String,
    dep_tier: Tier,
    table: &'static str,
    rule: &'static str,
}

pub fn lint_deps(_args: &[String]) -> ExitCode {
    let workspace_root = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("xtask lint-deps: cannot read cwd: {e}");
            return ExitCode::FAILURE;
        }
    };

    let crates_dir = workspace_root.join("crates");
    let extras_dir = workspace_root.join("extras");
    let workspace_cargo = workspace_root.join("Cargo.toml");

    if !crates_dir.is_dir() || !extras_dir.is_dir() || !workspace_cargo.is_file() {
        eprintln!(
            "xtask lint-deps: expected to be run from the workspace root (saw {})",
            workspace_root.display()
        );
        return ExitCode::FAILURE;
    }

    // Step 1: Build package map.
    let mut packages: HashMap<String, PackageInfo> = HashMap::new();
    if let Err(e) = collect_packages(&crates_dir, Tier::Crates, &mut packages) {
        eprintln!("xtask lint-deps: failed to scan crates/: {e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = collect_packages(&extras_dir, Tier::Extras, &mut packages) {
        eprintln!("xtask lint-deps: failed to scan extras/: {e}");
        return ExitCode::FAILURE;
    }

    // Step 2: Parse workspace.dependencies for the workspace = true lookups.
    let workspace_deps = match load_workspace_deps(&workspace_cargo) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("xtask lint-deps: failed to parse workspace Cargo.toml: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Step 3: Walk every package Cargo.toml and check arrows.
    let mut violations: Vec<Violation> = Vec::new();
    for (pkg_name, info) in &packages {
        if let Err(e) = check_package(
            pkg_name,
            info,
            &packages,
            &workspace_deps,
            &workspace_root,
            &mut violations,
        ) {
            eprintln!(
                "xtask lint-deps: error checking {} ({}): {e}",
                pkg_name,
                info.cargo_toml.display()
            );
            return ExitCode::FAILURE;
        }
    }

    if violations.is_empty() {
        println!(
            "lint-deps: OK ({} packages: {} crates, {} extras)",
            packages.len(),
            packages.values().filter(|p| p.tier == Tier::Crates).count(),
            packages.values().filter(|p| p.tier == Tier::Extras).count(),
        );
        return ExitCode::SUCCESS;
    }

    eprintln!(
        "lint-deps: {} forbidden dependency arrow{} found:",
        violations.len(),
        if violations.len() == 1 { "" } else { "s" },
    );
    for v in &violations {
        eprintln!(
            "  {}/{} → {}/{}\n      in [{}] of {}\n      rule: {}",
            v.consumer_tier.name(),
            v.consumer_pkg,
            v.dep_tier.name(),
            v.dep_name,
            v.table,
            v.consumer_cargo_toml.display(),
            v.rule,
        );
    }
    eprintln!();
    eprintln!("See docs/adr/ADR-0004-framework-extras-boundary.md.");
    ExitCode::FAILURE
}

fn collect_packages(
    root: &Path,
    tier: Tier,
    out: &mut HashMap<String, PackageInfo>,
) -> Result<(), String> {
    for entry in WalkDir::new(root)
        .min_depth(1)
        .max_depth(3)
        .into_iter()
        .filter_map(Result::ok)
    {
        if entry.file_name() != "Cargo.toml" {
            continue;
        }
        let path = entry.path().to_path_buf();
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let doc: DocumentMut = text
            .parse()
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        let Some(pkg) = doc.get("package") else {
            // Workspace root or virtual manifest — skip.
            continue;
        };
        let Some(name) = pkg.get("name").and_then(|v| v.as_str().map(str::to_string)) else {
            continue;
        };
        out.insert(
            name,
            PackageInfo {
                tier,
                cargo_toml: path,
            },
        );
    }
    Ok(())
}

/// Returns a map of `package_name → resolved_path` for every entry in the
/// workspace's `[workspace.dependencies]` table that uses `path = "..."`.
fn load_workspace_deps(workspace_cargo: &Path) -> Result<HashMap<String, PathBuf>, String> {
    let text = fs::read_to_string(workspace_cargo)
        .map_err(|e| format!("read {}: {e}", workspace_cargo.display()))?;
    let doc: DocumentMut = text
        .parse()
        .map_err(|e| format!("parse {}: {e}", workspace_cargo.display()))?;
    let mut out = HashMap::new();
    let workspace = doc.get("workspace").and_then(|w| w.as_table_like());
    if let Some(workspace_table) = workspace
        && let Some(deps) = workspace_table
            .get("dependencies")
            .and_then(|d| d.as_table_like())
    {
        for (name, item) in deps.iter() {
            if let Some(path) = extract_path(item) {
                out.insert(name.to_string(), PathBuf::from(path));
            }
        }
    }
    Ok(out)
}

fn extract_path(item: &Item) -> Option<String> {
    match item {
        Item::Value(Value::InlineTable(t)) => {
            t.get("path").and_then(|v| v.as_str().map(String::from))
        }
        Item::Table(t) => t.get("path").and_then(|v| v.as_str().map(String::from)),
        _ => None,
    }
}

fn check_package(
    pkg_name: &str,
    info: &PackageInfo,
    packages: &HashMap<String, PackageInfo>,
    workspace_deps: &HashMap<String, PathBuf>,
    workspace_root: &Path,
    violations: &mut Vec<Violation>,
) -> Result<(), String> {
    let text = fs::read_to_string(&info.cargo_toml)
        .map_err(|e| format!("read {}: {e}", info.cargo_toml.display()))?;
    let doc: DocumentMut = text
        .parse()
        .map_err(|e| format!("parse {}: {e}", info.cargo_toml.display()))?;

    let pkg_dir = info.cargo_toml.parent().unwrap_or(workspace_root);

    for table_name in ["dependencies", "dev-dependencies", "build-dependencies"] {
        let Some(table) = doc.get(table_name).and_then(|t| t.as_table_like()) else {
            continue;
        };
        for (dep_name, item) in table.iter() {
            // Skip self-deps from workspace = true lookups (rare).
            if dep_name == pkg_name {
                continue;
            }
            let dep_tier = classify_dep(item, dep_name, pkg_dir, workspace_deps, packages);
            let Some(dep_tier) = dep_tier else { continue };

            let rule = match (info.tier, dep_tier) {
                (Tier::Crates, Tier::Extras) => Some(
                    "framework crate (crates/) cannot depend on an extras/ entry — \
                     extras consume the framework, not vice-versa",
                ),
                (Tier::Extras, Tier::Extras) => Some(
                    "extras are siblings of equal standing, not a hierarchy — \
                     promote the shared library into crates/ instead",
                ),
                _ => None,
            };
            if let Some(rule) = rule {
                violations.push(Violation {
                    consumer_pkg: pkg_name.to_string(),
                    consumer_tier: info.tier,
                    consumer_cargo_toml: info.cargo_toml.clone(),
                    dep_name: dep_name.to_string(),
                    dep_tier,
                    table: match table_name {
                        "dependencies" => "dependencies",
                        "dev-dependencies" => "dev-dependencies",
                        "build-dependencies" => "build-dependencies",
                        _ => "?",
                    },
                    rule,
                });
            }
        }
    }
    Ok(())
}

/// Classify a dep into a tier, or `None` if it points outside the workspace
/// (e.g., crates.io deps, git deps without a path override).
fn classify_dep(
    item: &Item,
    dep_name: &str,
    pkg_dir: &Path,
    workspace_deps: &HashMap<String, PathBuf>,
    packages: &HashMap<String, PackageInfo>,
) -> Option<Tier> {
    // Direct path = "..." dep.
    if let Some(rel) = extract_path(item) {
        let resolved = pkg_dir.join(&rel);
        return tier_of_path(&resolved);
    }

    // workspace = true — look up the workspace.dependencies entry.
    let is_workspace_dep = match item {
        Item::Value(Value::InlineTable(t)) => t
            .get("workspace")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        Item::Table(t) => t
            .get("workspace")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        _ => false,
    };
    if is_workspace_dep && let Some(ws_path) = workspace_deps.get(dep_name) {
        return tier_of_path(ws_path);
    }

    // Some `dep_name.workspace = true` shorthand may parse as just a bool —
    // handled above. Otherwise fall back to the package map by name (covers
    // bare `dep_name = "X"` with no path / workspace marker, which can't be a
    // workspace member anyway, so this just returns None).
    let _ = packages;
    None
}

fn tier_of_path(path: &Path) -> Option<Tier> {
    let canon = path.canonicalize().ok()?;
    let s = canon.to_string_lossy();
    if s.contains("/crates/") {
        Some(Tier::Crates)
    } else if s.contains("/extras/") {
        Some(Tier::Extras)
    } else {
        None
    }
}
