use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use walkdir::WalkDir;

#[derive(Debug, PartialEq, Eq)]
enum BumpMode {
    Full,
    Patch,
}

struct BumpArgs {
    version: String,
    crates: Option<Vec<String>>,
}

/// Determine if this is a patch bump (same major.minor) or full bump.
fn bump_mode(current: &str, new: &str) -> BumpMode {
    let cur_parts: Vec<&str> = current.split('.').collect();
    let new_parts: Vec<&str> = new.split('.').collect();
    if cur_parts.len() >= 2
        && new_parts.len() >= 2
        && cur_parts[0] == new_parts[0]
        && cur_parts[1] == new_parts[1]
    {
        BumpMode::Patch
    } else {
        BumpMode::Full
    }
}

/// Parse bump-version arguments: `<VERSION> [--crates crate1,crate2,...]`
fn parse_bump_args(args: &[String]) -> Result<BumpArgs, String> {
    let version = args.first().ok_or("missing VERSION argument")?.clone();
    let mut crates = None;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--crates" {
            i += 1;
            let list = args.get(i).ok_or("--crates requires a value")?;
            crates = Some(list.split(',').map(|s| s.trim().to_string()).collect());
        }
        i += 1;
    }

    Ok(BumpArgs { version, crates })
}

/// Bump all version references across the workspace.
///
/// Dispatches to `bump_full()` or `bump_patch()` depending on whether the
/// major.minor changed.
pub fn bump_version(args: &[String]) -> ExitCode {
    let parsed = match parse_bump_args(args) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("Error: {msg}");
            eprintln!("Usage: cargo xtask bump-version <VERSION> [--crates crate1,crate2]");
            return ExitCode::FAILURE;
        }
    };

    let new_version = &parsed.version;

    // Validate semver format
    let parts: Vec<&str> = new_version.split('.').collect();
    if parts.len() != 3 || parts.iter().any(|p| p.parse::<u32>().is_err()) {
        eprintln!("Error: version must be semver (e.g. 0.4.1), got: {new_version}");
        return ExitCode::FAILURE;
    }

    let major_minor = format!("{}.{}", parts[0], parts[1]);
    let workspace_root = workspace_root();

    println!("Workspace root: {}", workspace_root.display());

    let current_version = read_workspace_version(&workspace_root);
    let mode = bump_mode(&current_version, new_version);

    if parsed.crates.is_some() && mode == BumpMode::Full {
        eprintln!("Error: --crates is only valid for patch bumps (same major.minor)");
        return ExitCode::FAILURE;
    }

    match mode {
        BumpMode::Patch => bump_patch(&workspace_root, new_version, &major_minor, parsed.crates),
        BumpMode::Full => {
            println!("Full version bump: {current_version} -> {new_version}");
            println!();
            bump_full(&workspace_root, new_version, &major_minor)
        }
    }
}

/// Perform a full version bump: update every version reference in the workspace.
///
/// Steps:
/// 0. Reset any per-crate version overrides from previous patch releases
/// 1. Update root Cargo.toml (workspace.package + workspace.dependencies)
/// 2. Update member Cargo.toml files with direct path deps
/// 3. Update excluded sub-workspace Cargo.toml files (e.g. extras/brainclaw)
/// 4. Update hardcoded versions in *.rs files
/// 5. Update hardcoded versions in *.ts files
/// 6. Update hardcoded versions in *.json files
/// 7. Update version examples in *.md files
/// 8. Stamp CHANGELOG.md
fn bump_full(root: &Path, new_version: &str, major_minor: &str) -> ExitCode {
    let mut changes = 0u32;

    let old_version = read_workspace_version(root);

    // Reset any per-crate version overrides from previous patch releases
    changes += reset_explicit_versions(root);

    // 1. Update root Cargo.toml (workspace.package + workspace.dependencies)
    changes += update_workspace_cargo_toml(root, new_version);

    // 2. Update member Cargo.toml files with direct path deps
    changes += update_member_cargo_tomls(root, new_version);

    // 3. Update excluded sub-workspace Cargo.toml files
    changes += update_excluded_workspace_cargo_tomls(root, &old_version, new_version);

    // 4. Update hardcoded versions in *.rs files
    changes += update_rs_files(root, new_version);

    // 5. Update hardcoded versions in *.ts files
    changes += update_ts_files(root, &old_version, new_version);

    // 6. Update hardcoded versions in *.json files
    changes += update_json_files(root, &old_version, new_version);

    // 7. Update version examples in *.md files
    changes += update_md_files(root, major_minor, &old_version, new_version);

    // 8. Stamp CHANGELOG.md
    changes += update_changelog(root, new_version);

    println!();
    if changes > 0 {
        println!("Done! Updated {changes} file(s).");
        println!();
        println!("Next steps:");
        println!("  1. Review changes: git diff");
        println!("  2. Run: cargo check --workspace");
        println!("  3. Commit the version bump");
    } else {
        println!("No files needed updating.");
    }

    ExitCode::SUCCESS
}

fn workspace_root() -> PathBuf {
    // xtask binary is at <root>/target/..., but we run via `cargo xtask`
    // which sets CWD to the workspace root. Use CARGO_MANIFEST_DIR of the
    // workspace (xtask's parent).
    let xtask_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    xtask_dir
        .parent()
        .expect("xtask should be inside workspace")
        .to_path_buf()
}

/// Update the root Cargo.toml:
/// - `[workspace.package].version`
/// - All `version = "..."` on internal brainwires-* deps in `[workspace.dependencies]`
fn update_workspace_cargo_toml(root: &Path, new_version: &str) -> u32 {
    let cargo_path = root.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_path).expect("Failed to read root Cargo.toml");

    let mut doc = content
        .parse::<toml_edit::DocumentMut>()
        .expect("Failed to parse root Cargo.toml");

    let mut changed = false;

    // Update [workspace.package].version
    if let Some(pkg) = doc.get_mut("workspace").and_then(|w| w.get_mut("package"))
        && let Some(v) = pkg.get_mut("version")
    {
        let old = v.as_str().unwrap_or("").to_string();
        if old != new_version {
            *v = toml_edit::value(new_version);
            println!("  [workspace.package].version: {old} -> {new_version}");
            changed = true;
        }
    }

    // Update [workspace.dependencies].brainwires-* version fields
    if let Some(deps) = doc
        .get_mut("workspace")
        .and_then(|w| w.get_mut("dependencies"))
        && let Some(table) = deps.as_table_like_mut()
    {
        for (key, value) in table.iter_mut() {
            if !key.starts_with("brainwires") {
                continue;
            }
            // Only update inline tables with a `path` key (internal crates)
            if let Some(tbl) = value.as_inline_table_mut()
                && tbl.contains_key("path")
                && let Some(v) = tbl.get_mut("version")
            {
                let old = v.as_str().unwrap_or("").to_string();
                if old != new_version {
                    *v = toml_edit::value(new_version)
                        .into_value()
                        .expect("string is a value");
                    println!("  [workspace.dependencies].{key}: {old} -> {new_version}");
                    changed = true;
                }
            }
        }
    }

    if changed {
        std::fs::write(&cargo_path, doc.to_string()).expect("Failed to write root Cargo.toml");
        println!("  Updated: {}", cargo_path.display());
        1
    } else {
        println!("  Root Cargo.toml: already at {new_version}");
        0
    }
}

/// Scan member Cargo.toml files for direct `path = "..."` deps on brainwires crates
/// that have a hardcoded `version` field (e.g. brainwires-wasm which can't use workspace
/// inheritance due to `default-features = false` override limitation).
fn update_member_cargo_tomls(root: &Path, new_version: &str) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) != Some("Cargo.toml") {
            continue;
        }
        // Skip the root Cargo.toml (already handled)
        if path == root.join("Cargo.toml") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut doc = match content.parse::<toml_edit::DocumentMut>() {
            Ok(d) => d,
            Err(_) => continue,
        };

        let mut changed = false;

        // Check [dependencies] and [dev-dependencies]
        for section in &["dependencies", "dev-dependencies", "build-dependencies"] {
            let Some(deps) = doc.get_mut(section) else {
                continue;
            };
            let Some(table) = deps.as_table_like_mut() else {
                continue;
            };

            for (key, value) in table.iter_mut() {
                if !key.starts_with("brainwires") {
                    continue;
                }
                if let Some(tbl) = value.as_inline_table_mut()
                    && tbl.contains_key("path")
                    && !tbl.contains_key("workspace")
                    && let Some(v) = tbl.get_mut("version")
                {
                    let old = v.as_str().unwrap_or("").to_string();
                    if old != new_version {
                        *v = toml_edit::value(new_version)
                            .into_value()
                            .expect("string is a value");
                        println!("  [{section}].{key}: {old} -> {new_version}");
                        changed = true;
                    }
                }
            }
        }

        if changed {
            std::fs::write(path, doc.to_string()).expect("Failed to write member Cargo.toml");
            println!("  Updated: {}", path.display());
            count += 1;
        }
    }

    if count == 0 {
        println!("  No member Cargo.toml files needed updating.");
    }
    count
}

/// Find and update hardcoded version strings in Rust source files.
/// Looks for patterns like `"version": "X.Y.Z"` and `"0.2.0"` in brainwires contexts.
fn update_rs_files(root: &Path, new_version: &str) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target"
                && name != ".git"
                && name != "node_modules"
                && name != "deprecated"
                && name != "xtask"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Replace "version": "X.Y.Z" patterns (JSON-style in Rust string literals).
        let new_content = replace_version_in_rs(&content, new_version);

        if new_content != content {
            std::fs::write(path, &new_content).expect("Failed to write .rs file");
            println!("  Updated: {}", path.display());
            count += 1;
        }
    }

    if count == 0 {
        println!("  No .rs files needed updating.");
    }
    count
}

/// Replace version strings in Rust source that match brainwires version patterns.
fn replace_version_in_rs(content: &str, new_version: &str) -> String {
    let mut result = content.to_string();

    // Pattern: "version": "X.Y.Z" (JSON in Rust strings)
    // We look for the specific pattern used in protocol.rs and similar
    let patterns = [
        // JSON-style only: "version": "X.Y.Z" (embedded JSON literals in Rust strings).
        // Do NOT match bare `version: "X.Y.Z"` struct fields — those are often arbitrary
        // test data (e.g. "0.1.0-test") and must not be treated as the framework version.
        (r#""version": ""#, '"'),
    ];

    for (prefix, terminator) in &patterns {
        let mut search_from = 0;
        loop {
            let Some(start) = result[search_from..].find(prefix) else {
                break;
            };
            let abs_start = search_from + start;
            let value_start = abs_start + prefix.len();
            let Some(end) = result[value_start..].find(*terminator) else {
                break;
            };
            let old_ver = &result[value_start..value_start + end];
            // Only replace if it looks like a brainwires version (0.x.y pattern)
            if old_ver.starts_with("0.") && old_ver.split('.').count() == 3 {
                let before = &result[..value_start];
                let after = &result[value_start + end..];
                result = format!("{before}{new_version}{after}");
                search_from = value_start + new_version.len();
            } else {
                search_from = value_start + end;
            }
        }
    }

    // Also handle doc-comment lines that contain markdown-style brainwires version
    // references (e.g. `//! brainwires = { version = "0.4", ... }`).
    // These use the same patterns as .md files but live inside .rs files.
    let new_mm = {
        let parts: Vec<&str> = new_version.split('.').collect();
        if parts.len() >= 2 {
            format!("{}.{}", parts[0], parts[1])
        } else {
            new_version.to_string()
        }
    };
    let mut doc_result = String::with_capacity(result.len());
    for line in result.lines() {
        let trimmed = line.trim_start();
        if (trimmed.starts_with("///") || trimmed.starts_with("//!"))
            && trimmed.contains("brainwires")
        {
            let leading_ws = &line[..line.len() - trimmed.len()];
            let marker = &trimmed[..3];
            let rest = &trimmed[3..];
            let updated_rest = replace_brainwires_version_in_line(rest, &new_mm);
            doc_result.push_str(leading_ws);
            doc_result.push_str(marker);
            doc_result.push_str(&updated_rest);
        } else {
            doc_result.push_str(line);
        }
        doc_result.push('\n');
    }
    // Preserve original trailing newline behavior
    if !result.ends_with('\n') && doc_result.ends_with('\n') {
        doc_result.pop();
    }

    doc_result
}

/// Update version references in Markdown files.
/// Replaces:
/// - `brainwires[-*] = { version = "X.Y"` and `brainwires[-*] = "X.Y"` patterns
/// - `"version": "OLD"` and `"cli_version": "OLD"` in code-block examples
/// - `vOLD_VERSION` bare version tags (e.g. `v0.6.0`)
fn update_md_files(
    root: &Path,
    new_major_minor: &str,
    old_version: &str,
    new_version: &str,
) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }

        // Skip CHANGELOG files — version references there are historical
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if filename.to_ascii_uppercase().contains("CHANGELOG") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Existing brainwires dep-style replacements
        let mut new_content = replace_version_in_md(&content, new_major_minor);

        // Also replace verbatim version strings in code-block examples:
        // "version": "OLD", "cli_version": "OLD", vOLD (bare tag), /OLD/ path segment
        for pattern in &[
            format!("\"version\": \"{old_version}\""),
            format!("\"cli_version\": \"{old_version}\""),
            format!("v{old_version}"),
            format!("/{old_version}"),
        ] {
            let replacement = pattern.replace(old_version, new_version);
            if new_content.contains(pattern.as_str()) {
                new_content = new_content.replace(pattern.as_str(), &replacement);
            }
        }

        if new_content != content {
            std::fs::write(path, &new_content).expect("Failed to write .md file");
            println!("  Updated: {}", path.display());
            count += 1;
        }
    }

    if count == 0 {
        println!("  No .md files needed updating.");
    }
    count
}

/// Update version strings in TypeScript source and test files.
/// Replaces `version: "OLD"` patterns used in example/test data.
fn update_ts_files(root: &Path, old_version: &str, new_version: &str) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ts") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let new_content = content
            .replace(
                &format!("version: \"{old_version}\""),
                &format!("version: \"{new_version}\""),
            )
            .replace(
                &format!("\"version\": \"{old_version}\""),
                &format!("\"version\": \"{new_version}\""),
            );

        if new_content != content {
            std::fs::write(path, &new_content).expect("Failed to write .ts file");
            println!("  Updated: {}", path.display());
            count += 1;
        }
    }

    if count == 0 {
        println!("  No .ts files needed updating.");
    }
    count
}

/// Update version strings in JSON files (e.g. user_agent strings in config examples).
fn update_json_files(root: &Path, old_version: &str, new_version: &str) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Only replace version strings that appear after a brainwires crate name,
        // e.g. "brainwires-cli/0.6.0" or `"version": "0.6.0"` in example configs.
        let new_content = content
            .replace(&format!("/{old_version}\""), &format!("/{new_version}\""))
            .replace(
                &format!("\"version\": \"{old_version}\""),
                &format!("\"version\": \"{new_version}\""),
            );

        if new_content != content {
            std::fs::write(path, &new_content).expect("Failed to write .json file");
            println!("  Updated: {}", path.display());
            count += 1;
        }
    }

    if count == 0 {
        println!("  No .json files needed updating.");
    }
    count
}

/// Update [workspace.package].version in excluded sub-workspace Cargo.toml files
/// (e.g. extras/brainclaw/Cargo.toml which has its own workspace, not inherited).
fn update_excluded_workspace_cargo_tomls(root: &Path, old_version: &str, new_version: &str) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .max_depth(3)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) != Some("Cargo.toml") {
            continue;
        }
        // Skip the root workspace Cargo.toml (already handled)
        if path == root.join("Cargo.toml") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut doc = match content.parse::<toml_edit::DocumentMut>() {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Only act on files that have a [workspace] table (sub-workspaces)
        // and whose [workspace.package].version is the old version.
        let Some(pkg) = doc.get_mut("workspace").and_then(|w| w.get_mut("package")) else {
            continue;
        };

        let Some(v) = pkg.get_mut("version") else {
            continue;
        };

        if v.as_str() != Some(old_version) {
            continue;
        }

        *v = toml_edit::value(new_version);
        std::fs::write(path, doc.to_string()).expect("Failed to write sub-workspace Cargo.toml");
        println!("  [workspace.package].version: {old_version} -> {new_version}");
        println!("  Updated: {}", path.display());
        count += 1;
    }

    if count == 0 {
        println!("  No excluded sub-workspace Cargo.toml files needed updating.");
    }
    count
}

/// Update CHANGELOG.md: rename `## [Unreleased]` to `## [X.Y.Z]` and insert
/// a fresh empty `## [Unreleased]` section above it.
///
/// Looks for the first line matching `## [Unreleased]` (case-insensitive on the
/// word "Unreleased"). If the section has content, it becomes the new release
/// section. A blank `## [Unreleased]` header is inserted above it.
fn update_changelog(root: &Path, new_version: &str) -> u32 {
    let changelog_path = root.join("CHANGELOG.md");
    let content = match std::fs::read_to_string(&changelog_path) {
        Ok(c) => c,
        Err(_) => {
            println!("  CHANGELOG.md: not found, skipping");
            return 0;
        }
    };

    // Find the `## [Unreleased]` line (case-insensitive match on "unreleased").
    let mut lines: Vec<&str> = content.lines().collect();
    let unreleased_idx = lines.iter().position(|line| {
        let trimmed = line.trim();
        trimmed.to_ascii_lowercase().starts_with("## [unreleased]")
    });

    let Some(idx) = unreleased_idx else {
        println!("  CHANGELOG.md: no ## [Unreleased] section found, skipping");
        return 0;
    };

    // Build the today's date string for the release heading.
    let today = {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Convert to YYYY-MM-DD without pulling in chrono.
        let days_since_epoch = now / 86400;
        let (y, m, d) = days_to_ymd(days_since_epoch);
        format!("{y:04}-{m:02}-{d:02}")
    };

    // Replace the existing Unreleased line with the versioned heading.
    let versioned_heading = format!("## [{new_version}] - {today}");

    // Insert a fresh Unreleased section above the old one.
    // Result: ## [Unreleased] / blank / ## [X.Y.Z] - YYYY-MM-DD / (original content)
    lines[idx] = &versioned_heading;
    let fresh_section = ["## [Unreleased]", ""];
    let mut new_lines: Vec<&str> = Vec::with_capacity(lines.len() + fresh_section.len());
    new_lines.extend_from_slice(&lines[..idx]);
    new_lines.extend_from_slice(&fresh_section);
    new_lines.extend_from_slice(&lines[idx..]);

    // Rebuild with trailing newline.
    let mut new_content = new_lines.join("\n");
    if content.ends_with('\n') {
        new_content.push('\n');
    }

    if new_content == content {
        println!("  CHANGELOG.md: already stamped for {new_version}");
        return 0;
    }

    std::fs::write(&changelog_path, &new_content).expect("Failed to write CHANGELOG.md");
    println!("  CHANGELOG.md: [Unreleased] -> [{new_version}] - {today}");
    println!("  Updated: {}", changelog_path.display());
    1
}

/// Convert days since Unix epoch to (year, month, day).
///
/// Simple civil date calculation — no leap-second precision needed for
/// changelog timestamps.
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from Howard Hinnant's `chrono`-compatible date conversion.
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Map a workspace-relative file path to its brainwires crate name.
/// Assumes paths like "crates/brainwires-core/..." or "extras/brainwires-proxy/..."
fn file_to_crate(path: &str) -> Option<String> {
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() >= 2 {
        let dir = parts[1];
        if dir.starts_with("brainwires") {
            return Some(dir.to_string());
        }
    }
    None
}

/// Auto-detect which crates have changed since the last version tag.
/// Returns None if detection fails (no tags, not a git repo, etc.)
fn detect_changed_crates(root: &Path, current_version: &str) -> Option<Vec<String>> {
    let tag = format!("v{current_version}");
    let output = std::process::Command::new("git")
        .args(["diff", "--name-only", &format!("{tag}..HEAD")])
        .current_dir(root)
        .output()
        .ok()?;

    if !output.status.success() {
        let output = std::process::Command::new("git")
            .args(["diff", "--name-only", &format!("{current_version}..HEAD")])
            .current_dir(root)
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        return Some(parse_git_diff_to_crates(&output.stdout));
    }

    Some(parse_git_diff_to_crates(&output.stdout))
}

fn parse_git_diff_to_crates(stdout: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(stdout);
    let mut crates: HashSet<String> = HashSet::new();
    for line in text.lines() {
        if let Some(name) = file_to_crate(line.trim()) {
            crates.insert(name);
        }
    }
    let mut sorted: Vec<String> = crates.into_iter().collect();
    sorted.sort();
    sorted
}

/// Replace `brainwires* = { version = "X.Y"` and `brainwires* = "X.Y"` in markdown.
fn replace_version_in_md(content: &str, new_major_minor: &str) -> String {
    let mut result = String::with_capacity(content.len());

    for line in content.lines() {
        let new_line = replace_brainwires_version_in_line(line, new_major_minor);
        result.push_str(&new_line);
        result.push('\n');
    }

    // Preserve original trailing newline behavior
    if !content.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

/// Replace version in a single markdown line for brainwires crate references.
fn replace_brainwires_version_in_line(line: &str, new_mm: &str) -> String {
    // Pattern 1: brainwires[-*] = { version = "X.Y", ... }
    // Pattern 2: brainwires[-*] = "X.Y"
    if !line.contains("brainwires") {
        return line.to_string();
    }

    let mut result = line.to_string();

    // Pattern 1: version = "X.Y" (inside inline table or toml)
    let version_eq = "version = \"";
    let mut search_from = 0;
    loop {
        let Some(ver_pos) = result[search_from..].find(version_eq) else {
            break;
        };
        let abs_pos = search_from + ver_pos;

        if !result[..abs_pos].contains("brainwires") {
            search_from = abs_pos + version_eq.len();
            continue;
        }

        let value_start = abs_pos + version_eq.len();
        let Some(end) = result[value_start..].find('"') else {
            break;
        };
        let old_ver = &result[value_start..value_start + end];

        if old_ver.starts_with("0.") && old_ver != new_mm {
            let before = &result[..value_start].to_string();
            let after = &result[value_start + end..].to_string();
            result = format!("{before}{new_mm}{after}");
            search_from = value_start + new_mm.len();
        } else {
            search_from = value_start + end;
        }
    }

    // Pattern 2: brainwires[-*] = "X.Y" (simple form, no inline table)
    // Match: `brainwires` optionally followed by `-word` segments, then ` = "X.Y"`
    // Skip lines already handled by Pattern 1 (contain `version = "`)
    if !result.contains("version = \"") {
        let eq_quote = "= \"";
        search_from = 0;
        loop {
            let Some(eq_pos) = result[search_from..].find(eq_quote) else {
                break;
            };
            let abs_eq = search_from + eq_pos;

            // Check that a brainwires identifier immediately precedes ` = "`
            let before_eq = result[..abs_eq].trim_end();
            if !before_eq.ends_with(|c: char| c.is_ascii_alphanumeric() || c == '-') {
                search_from = abs_eq + eq_quote.len();
                continue;
            }
            // Walk backwards to find the start of the identifier
            let ident_end = before_eq.len();
            let ident_start = before_eq
                .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
                .map(|i| i + 1)
                .unwrap_or(0);
            let ident = &before_eq[ident_start..ident_end];
            if !ident.starts_with("brainwires") {
                search_from = abs_eq + eq_quote.len();
                continue;
            }

            let value_start = abs_eq + eq_quote.len();
            let Some(end) = result[value_start..].find('"') else {
                break;
            };
            let old_ver = &result[value_start..value_start + end];

            if old_ver.starts_with("0.") && old_ver != new_mm {
                let before = result[..value_start].to_string();
                let after = result[value_start + end..].to_string();
                result = format!("{before}{new_mm}{after}");
                search_from = value_start + new_mm.len();
            } else {
                search_from = value_start + end;
            }
        }
    }

    result
}

/// Reset any crate with an explicit version back to `version.workspace = true`.
/// Called during full (minor/major) bumps to clean up after patch releases.
fn reset_explicit_versions(root: &Path) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) != Some("Cargo.toml") {
            continue;
        }
        if path == root.join("Cargo.toml") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let mut doc = match content.parse::<toml_edit::DocumentMut>() {
            Ok(d) => d,
            Err(_) => continue,
        };

        let crate_name = doc
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();

        if !crate_name.starts_with("brainwires") {
            continue;
        }

        let Some(pkg) = doc.get_mut("package") else {
            continue;
        };

        // Check if version is an explicit string (not workspace inherited)
        let is_explicit = pkg.get("version").map(|v| v.is_str()).unwrap_or(false);

        if !is_explicit {
            continue;
        }

        let old = pkg
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Replace with version.workspace = true using dotted-key table form
        // (matches the style used by all other workspace-inherited fields like
        // edition.workspace = true, license.workspace = true, etc.)
        let mut tbl = toml_edit::Table::new();
        tbl.set_dotted(true);
        tbl.insert("workspace", toml_edit::value(true));
        pkg.as_table_like_mut()
            .unwrap()
            .insert("version", toml_edit::Item::Table(tbl));

        println!("  {crate_name}: version = \"{old}\" -> version.workspace = true");
        std::fs::write(path, doc.to_string()).expect("write member Cargo.toml");
        count += 1;
    }

    count
}

/// Build a map of crate_name -> [dependency crate names] for internal brainwires crates.
/// Parses each member Cargo.toml for brainwires-* dependencies.
fn build_dep_graph(root: &Path) -> HashMap<String, Vec<String>> {
    let mut graph: HashMap<String, Vec<String>> = HashMap::new();

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) != Some("Cargo.toml") {
            continue;
        }
        if path == root.join("Cargo.toml") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let doc = match content.parse::<toml_edit::DocumentMut>() {
            Ok(d) => d,
            Err(_) => continue,
        };

        let Some(name) = doc
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
        else {
            continue;
        };

        if !name.starts_with("brainwires") {
            continue;
        }

        let mut deps = Vec::new();
        for section in &["dependencies", "dev-dependencies", "build-dependencies"] {
            let Some(dep_table) = doc.get(section).and_then(|d| d.as_table_like()) else {
                continue;
            };
            for (key, _) in dep_table.iter() {
                if key.starts_with("brainwires") && key != name {
                    deps.push(key.to_string());
                }
            }
        }

        graph.insert(name.to_string(), deps);
    }

    graph
}

/// Read the current version from [workspace.package].version
fn read_workspace_version(root: &Path) -> String {
    let cargo_path = root.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_path).expect("Failed to read root Cargo.toml");
    let doc = content
        .parse::<toml_edit::DocumentMut>()
        .expect("Failed to parse root Cargo.toml");
    doc.get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .expect("No [workspace.package].version found")
        .to_string()
}

/// Check if a workspace-relative path belongs to any of the affected crates.
fn is_in_affected_crate(rel_path: &str, affected: &HashSet<String>) -> bool {
    let parts: Vec<&str> = rel_path.split('/').collect();
    if parts.len() >= 2 {
        let dir = parts[1];
        if dir.starts_with("brainwires") {
            return affected.contains(dir);
        }
    }
    false
}

fn update_workspace_deps_selective(
    root: &Path,
    new_version: &str,
    affected: &HashSet<String>,
) -> u32 {
    let cargo_path = root.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_path).expect("read root Cargo.toml");
    let mut doc = content
        .parse::<toml_edit::DocumentMut>()
        .expect("parse root Cargo.toml");
    let mut changed = false;

    if let Some(deps) = doc
        .get_mut("workspace")
        .and_then(|w| w.get_mut("dependencies"))
        && let Some(table) = deps.as_table_like_mut()
    {
        for (key, value) in table.iter_mut() {
            if !affected.contains(key.get()) {
                continue;
            }
            if let Some(tbl) = value.as_inline_table_mut()
                && tbl.contains_key("path")
                && let Some(v) = tbl.get_mut("version")
            {
                let old = v.as_str().unwrap_or("").to_string();
                if old != new_version {
                    *v = toml_edit::value(new_version)
                        .into_value()
                        .expect("string is a value");
                    println!("  [workspace.dependencies].{key}: {old} -> {new_version}");
                    changed = true;
                }
            }
        }
    }

    if changed {
        std::fs::write(&cargo_path, doc.to_string()).expect("write root Cargo.toml");
        println!("  Updated: {}", cargo_path.display());
        1
    } else {
        0
    }
}

fn set_explicit_versions(root: &Path, new_version: &str, affected: &HashSet<String>) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) != Some("Cargo.toml") {
            continue;
        }
        if path == root.join("Cargo.toml") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let mut doc = match content.parse::<toml_edit::DocumentMut>() {
            Ok(d) => d,
            Err(_) => continue,
        };

        let crate_name = doc
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();

        if !affected.contains(&crate_name) {
            continue;
        }

        let Some(pkg) = doc.get_mut("package") else {
            continue;
        };

        // Check if version is inherited from workspace.
        // Dotted keys like `version.workspace = true` parse as a Table, not InlineTable,
        // so we use as_table_like() which handles both forms.
        let is_workspace = pkg
            .get("version")
            .and_then(|v| v.as_table_like())
            .map(|t| t.contains_key("workspace"))
            .unwrap_or(false);

        if is_workspace {
            // Replace `version.workspace = true` with `version = "X.Y.Z"`
            pkg.as_table_like_mut()
                .unwrap()
                .insert("version", toml_edit::value(new_version));
            println!("  {crate_name}: version.workspace = true -> version = \"{new_version}\"");
        } else {
            // Already has explicit version — update it
            if let Some(v) = pkg.get_mut("version") {
                let old = v.as_str().unwrap_or("").to_string();
                if old != new_version {
                    *v = toml_edit::value(new_version);
                    println!("  {crate_name}: version {old} -> {new_version}");
                }
            }
        }

        std::fs::write(path, doc.to_string()).expect("write member Cargo.toml");
        count += 1;
    }

    count
}

fn update_rs_files_selective(root: &Path, new_version: &str, affected: &HashSet<String>) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }

        let rel = path.strip_prefix(root).unwrap_or(path);
        let rel_str = rel.to_string_lossy();
        if !is_in_affected_crate(&rel_str, affected) {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let new_content = replace_version_in_rs(&content, new_version);
        if new_content != content {
            std::fs::write(path, &new_content).expect("write .rs file");
            println!("  Updated: {}", path.display());
            count += 1;
        }
    }

    count
}

fn update_md_files_selective(
    root: &Path,
    new_major_minor: &str,
    affected: &HashSet<String>,
) -> u32 {
    let mut count = 0u32;

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            name != "target" && name != ".git" && name != "node_modules" && name != "deprecated"
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if filename.to_ascii_uppercase().contains("CHANGELOG") {
            continue;
        }

        let rel = path.strip_prefix(root).unwrap_or(path);
        let rel_str = rel.to_string_lossy();
        if !is_in_affected_crate(&rel_str, affected) {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let new_content = replace_version_in_md(&content, new_major_minor);
        if new_content != content {
            std::fs::write(path, &new_content).expect("write .md file");
            println!("  Updated: {}", path.display());
            count += 1;
        }
    }

    count
}

/// Selective patch bump: only bump affected crates and their transitive dependents.
/// Workspace root version is intentionally NOT bumped — it stays at the current minor.
fn bump_patch(
    root: &Path,
    new_version: &str,
    major_minor: &str,
    explicit_crates: Option<Vec<String>>,
) -> ExitCode {
    let current_version = read_workspace_version(root);

    let direct = match explicit_crates {
        Some(crates) => crates,
        None => match detect_changed_crates(root, &current_version) {
            Some(crates) if !crates.is_empty() => crates,
            Some(_) => {
                println!("No crate changes detected since v{current_version}.");
                println!("Use --crates to specify crates manually.");
                return ExitCode::FAILURE;
            }
            None => {
                eprintln!("Could not detect changes (no git tag v{current_version}?).");
                eprintln!("Use --crates to specify crates manually.");
                return ExitCode::FAILURE;
            }
        },
    };

    let graph = build_dep_graph(root);
    let affected = cascade(&direct, &graph);

    let direct_set: HashSet<String> = direct.iter().cloned().collect();
    let mut cascaded: Vec<&String> = affected
        .iter()
        .filter(|c| !direct_set.contains(*c))
        .collect();
    cascaded.sort();
    let mut direct_sorted: Vec<&String> = direct.iter().collect();
    direct_sorted.sort();

    println!("Patch bump to {new_version}:");
    println!(
        "  Direct:  {}",
        direct_sorted
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !cascaded.is_empty() {
        println!(
            "  Cascade: {}",
            cascaded
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("  Total:   {} crate(s)", affected.len());
    println!();

    let mut changes = 0u32;

    changes += update_workspace_deps_selective(root, new_version, &affected);
    changes += set_explicit_versions(root, new_version, &affected);
    changes += update_rs_files_selective(root, new_version, &affected);
    changes += update_md_files_selective(root, major_minor, &affected);
    changes += update_changelog(root, new_version);

    println!();
    if changes > 0 {
        println!("Done! Updated {changes} file(s).");
        println!();
        println!("Next steps:");
        println!("  1. Review changes: git diff");
        println!("  2. Run: cargo check --workspace");
        println!("  3. Commit the version bump");
    } else {
        println!("No files needed updating.");
    }

    ExitCode::SUCCESS
}

/// Given a set of directly-affected crates, compute the full set including
/// all transitive dependents (crates that depend on any affected crate).
fn cascade(direct: &[String], graph: &HashMap<String, Vec<String>>) -> HashSet<String> {
    let mut affected: HashSet<String> = direct.iter().cloned().collect();
    let mut queue: VecDeque<String> = direct.iter().cloned().collect();

    while let Some(crate_name) = queue.pop_front() {
        for (dependent, deps) in graph {
            if deps.contains(&crate_name) && !affected.contains(dependent) {
                affected.insert(dependent.clone());
                queue.push_back(dependent.clone());
            }
        }
    }

    affected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rs_version_json_style() {
        // Only JSON-style "version": "X.Y.Z" is replaced.
        // Bare struct-field `version: "X.Y.Z"` is intentionally left alone
        // to avoid corrupting arbitrary test-data version strings.
        let json_line = r#"    "version": "0.1.0""#;
        let field_line = r#"    version: "0.1.0".into()"#;
        let input = format!("{json_line}\n{field_line}");
        let result = replace_version_in_rs(&input, "0.5.0");
        assert!(result.contains("0.5.0"), "should contain new version");
        assert!(
            !result.contains(r#""version": "0.1.0""#),
            "JSON-style line should be replaced"
        );
        assert!(
            result.contains(r#"version: "0.1.0""#),
            "bare struct-field line should be left unchanged"
        );
    }

    #[test]
    fn test_md_inline_table() {
        let input = r#"brainwires = { version = "0.1", features = ["agents"] }"#;
        let result = replace_brainwires_version_in_line(input, "0.5");
        assert_eq!(
            result,
            r#"brainwires = { version = "0.5", features = ["agents"] }"#
        );
    }

    #[test]
    fn test_md_leaves_non_brainwires_alone() {
        let input = r#"tokio = { version = "1.43", features = ["full"] }"#;
        let result = replace_brainwires_version_in_line(input, "0.5");
        assert_eq!(result, input);
    }

    #[test]
    fn test_md_hyphenated_crate() {
        let input = r#"brainwires-agent-network = { version = "0.1", features = ["mesh"] }"#;
        let result = replace_brainwires_version_in_line(input, "0.5");
        assert_eq!(
            result,
            r#"brainwires-agent-network = { version = "0.5", features = ["mesh"] }"#
        );
    }

    #[test]
    fn test_md_simple_form() {
        let input = r#"brainwires-storage = "0.3""#;
        let result = replace_brainwires_version_in_line(input, "0.5");
        assert_eq!(result, r#"brainwires-storage = "0.5""#);
    }

    #[test]
    fn test_md_simple_form_with_comment() {
        let input = r#"brainwires = "0.2"  # default features: tools + agents"#;
        let result = replace_brainwires_version_in_line(input, "0.5");
        assert_eq!(
            result,
            r#"brainwires = "0.5"  # default features: tools + agents"#
        );
    }

    #[test]
    fn test_md_simple_form_leaves_non_brainwires() {
        let input = r#"tokio = "1.43""#;
        let result = replace_brainwires_version_in_line(input, "0.5");
        assert_eq!(result, input);
    }

    #[test]
    fn test_md_simple_form_no_change_when_current() {
        let input = r#"brainwires = "0.5""#;
        let result = replace_brainwires_version_in_line(input, "0.5");
        assert_eq!(result, input);
    }

    #[test]
    fn test_days_to_ymd_epoch() {
        // 1970-01-01
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn test_days_to_ymd_known_date() {
        // 2026-03-14 is day 20526 since epoch
        let (y, m, d) = days_to_ymd(20526);
        assert_eq!((y, m, d), (2026, 3, 14));
    }

    #[test]
    fn test_changelog_update() {
        let tmpdir = std::env::temp_dir().join("xtask_changelog_test");
        let _ = std::fs::create_dir_all(&tmpdir);
        let changelog = tmpdir.join("CHANGELOG.md");
        std::fs::write(
            &changelog,
            "# Changelog\n\n## [Unreleased]\n\n### Added\n- Cool feature\n\n## [0.3.0] - 2025-12-01\n",
        )
        .unwrap();

        let count = update_changelog(&tmpdir, "0.4.0");
        assert_eq!(count, 1);

        let result = std::fs::read_to_string(&changelog).unwrap();
        // Should have a fresh Unreleased section
        assert!(result.contains("## [Unreleased]\n\n## [0.4.0]"));
        // The release date should be today
        assert!(result.contains("## [0.4.0] - "));
        // Original content should be preserved under the new version heading
        assert!(result.contains("### Added\n- Cool feature"));
        // Old release should still be there
        assert!(result.contains("## [0.3.0] - 2025-12-01"));

        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[test]
    fn test_detect_patch_bump() {
        assert_eq!(bump_mode("0.4.0", "0.4.1"), BumpMode::Patch);
        assert_eq!(bump_mode("0.4.0", "0.4.2"), BumpMode::Patch);
    }

    #[test]
    fn test_detect_minor_bump() {
        assert_eq!(bump_mode("0.4.0", "0.5.0"), BumpMode::Full);
        assert_eq!(bump_mode("0.4.1", "0.5.0"), BumpMode::Full);
    }

    #[test]
    fn test_detect_major_bump() {
        assert_eq!(bump_mode("0.4.0", "1.0.0"), BumpMode::Full);
    }

    #[test]
    fn test_parse_crates_flag() {
        let args = vec![
            "0.4.1".into(),
            "--crates".into(),
            "brainwires-core,brainwires-agent".into(),
        ];
        let parsed = parse_bump_args(&args).unwrap();
        assert_eq!(parsed.version, "0.4.1");
        assert_eq!(
            parsed.crates,
            Some(vec!["brainwires-core".into(), "brainwires-agent".into()])
        );
    }

    #[test]
    fn test_parse_no_crates_flag() {
        let args = vec!["0.5.0".into()];
        let parsed = parse_bump_args(&args).unwrap();
        assert_eq!(parsed.version, "0.5.0");
        assert_eq!(parsed.crates, None);
    }

    #[test]
    fn test_changelog_no_unreleased() {
        let tmpdir = std::env::temp_dir().join("xtask_changelog_test_none");
        let _ = std::fs::create_dir_all(&tmpdir);
        let changelog = tmpdir.join("CHANGELOG.md");
        std::fs::write(&changelog, "# Changelog\n\n## [0.3.0]\n").unwrap();

        let count = update_changelog(&tmpdir, "0.4.0");
        assert_eq!(count, 0, "should not modify if no [Unreleased] section");

        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[test]
    fn test_cascade_single_dep() {
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();
        graph.insert("brainwires-agent".into(), vec!["brainwires-core".into()]);
        graph.insert("brainwires-core".into(), vec![]);

        let affected = cascade(&["brainwires-core".to_string()], &graph);
        assert!(affected.contains("brainwires-core"));
        assert!(affected.contains("brainwires-agent"));
    }

    #[test]
    fn test_cascade_transitive() {
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();
        graph.insert("brainwires".into(), vec!["brainwires-agent".into()]);
        graph.insert("brainwires-agent".into(), vec!["brainwires-core".into()]);
        graph.insert("brainwires-core".into(), vec![]);

        let affected = cascade(&["brainwires-core".to_string()], &graph);
        assert!(affected.contains("brainwires-core"));
        assert!(affected.contains("brainwires-agent"));
        assert!(affected.contains("brainwires"));
    }

    #[test]
    fn test_cascade_no_deps() {
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();
        graph.insert("brainwires-core".into(), vec![]);
        graph.insert("brainwires-skills".into(), vec![]);

        let affected = cascade(&["brainwires-skills".to_string()], &graph);
        assert!(affected.contains("brainwires-skills"));
        assert!(!affected.contains("brainwires-core"));
    }

    #[test]
    fn test_reset_explicit_version() {
        let tmpdir = std::env::temp_dir().join("xtask_reset_test");
        let _ = std::fs::remove_dir_all(&tmpdir);
        std::fs::create_dir_all(tmpdir.join("crates/brainwires-test")).unwrap();

        // Create a root Cargo.toml (needed so reset skips it)
        std::fs::write(
            tmpdir.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/brainwires-test\"]\n",
        )
        .unwrap();

        // Create a member with explicit version
        std::fs::write(
            tmpdir.join("crates/brainwires-test/Cargo.toml"),
            "[package]\nname = \"brainwires-test\"\nversion = \"0.4.1\"\nedition = \"2024\"\n",
        )
        .unwrap();

        let count = reset_explicit_versions(&tmpdir);
        assert_eq!(count, 1);

        let result =
            std::fs::read_to_string(tmpdir.join("crates/brainwires-test/Cargo.toml")).unwrap();
        assert!(
            result.contains("version.workspace = true")
                || result.contains("version = { workspace = true }"),
            "should have workspace-inherited version, got:\n{result}"
        );
        assert!(
            !result.contains("\"0.4.1\""),
            "should not contain old explicit version"
        );

        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[test]
    fn test_is_in_affected_crate() {
        let affected: HashSet<String> =
            ["brainwires-core".into(), "brainwires-agent".into()].into();
        assert!(is_in_affected_crate(
            "crates/brainwires-core/src/lib.rs",
            &affected
        ));
        assert!(is_in_affected_crate(
            "crates/brainwires-agent/src/mod.rs",
            &affected
        ));
        assert!(!is_in_affected_crate(
            "crates/brainwires-storage/src/lib.rs",
            &affected
        ));
        assert!(!is_in_affected_crate("xtask/src/main.rs", &affected));
        assert!(!is_in_affected_crate("README.md", &affected));
    }

    #[test]
    fn test_file_to_crate_name() {
        assert_eq!(
            file_to_crate("crates/brainwires-core/src/lib.rs"),
            Some("brainwires-core".to_string())
        );
        assert_eq!(
            file_to_crate("crates/brainwires-agent/src/mod.rs"),
            Some("brainwires-agent".to_string())
        );
        assert_eq!(
            file_to_crate("extras/brainwires-proxy/src/main.rs"),
            Some("brainwires-proxy".to_string())
        );
        assert_eq!(file_to_crate("README.md"), None);
        assert_eq!(file_to_crate("xtask/src/main.rs"), None);
    }

    #[test]
    fn test_rs_doc_comment_version() {
        let input = concat!(
            "//! ```toml\n",
            "//! brainwires = { version = \"0.2\", features = [\"full\"] }\n",
            "//! ```\n",
        );
        let result = replace_version_in_rs(input, "0.5.0");
        assert!(
            result.contains("version = \"0.5\""),
            "should update doc comment version, got:\n{result}"
        );
        assert!(
            !result.contains("version = \"0.2\""),
            "should not contain old version"
        );
    }

    #[test]
    fn test_rs_doc_comment_leaves_non_brainwires() {
        let input = "/// tokio = { version = \"1.43\", features = [\"full\"] }\n";
        let result = replace_version_in_rs(input, "0.5.0");
        assert_eq!(
            result, input,
            "should not modify non-brainwires doc comments"
        );
    }
}
