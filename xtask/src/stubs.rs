use std::path::{Path, PathBuf};
use std::process::ExitCode;
use walkdir::WalkDir;

/// Patterns that indicate unfinished code.
///
/// Each entry is `(pattern, description, is_macro)`.
///   - `is_macro = true` means the pattern is a Rust macro invocation like `todo!()`
///     and the match is exact (word-boundary aware).
///   - `is_macro = false` means the pattern is a comment/string marker and we
///     match case-insensitively.
const PATTERNS: &[(&str, &str, bool)] = &[
    // ── Rust panic macros (hard blockers) ────────────────────────────
    ("todo!(", "todo!() macro — panics at runtime", true),
    (
        "unimplemented!(",
        "unimplemented!() macro — panics at runtime",
        true,
    ),
    // ── Comment markers (soft warnings) ──────────────────────────────
    ("FIXME", "FIXME marker", false),
    ("HACK", "HACK marker", false),
    ("XXX", "XXX marker", false),
    ("STUB", "STUB marker", false),
    ("STOPSHIP", "STOPSHIP marker", false),
    // ── Strings that suggest incomplete code ─────────────────────────
    (
        "not yet implemented",
        "\"not yet implemented\" string",
        false,
    ),
    ("not implemented", "\"not implemented\" string", false),
];

/// Lines containing these strings are excluded from results.
const ALLOW_LIST: &[&str] = &[
    "#[cfg(test)]",
    "#[test]",
    "mod tests",
    // Allow comments that document the pattern rather than use it
    // e.g. `// We removed the todo!() stubs`
    "check-stubs",
    "check_stubs",
    "PATTERNS",
];

/// Directories to skip entirely.
const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules", "test-results"];

/// Files to skip entirely (relative to workspace root).
const SKIP_FILES: &[&str] = &[
    "xtask/src/stubs.rs",
    // todo_scanner's purpose is scanning for TODO/FIXME markers — its code necessarily contains them
    "crates/rullama-autonomy/src/self_improve/strategies/todo_scanner.rs",
];

/// A single finding in the scan.
struct Finding {
    path: PathBuf,
    line_no: usize,
    line: String,
    _pattern: &'static str,
    description: &'static str,
    is_hard: bool,
}

/// Scan the workspace for unfinished code markers.
///
/// Returns `ExitCode::FAILURE` if any hard blockers (macros) are found,
/// `ExitCode::SUCCESS` otherwise (warnings are informational).
pub fn check_stubs(args: &[String]) -> ExitCode {
    let workspace_root = workspace_root();

    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let strict = args.iter().any(|a| a == "--strict" || a == "-s");
    let show_help = args.iter().any(|a| a == "--help" || a == "-h");

    if show_help {
        println!("Usage: cargo xtask check-stubs [OPTIONS]");
        println!();
        println!("Scan for unfinished code (todo!(), FIXME, HACK, etc.) in published source.");
        println!();
        println!("Options:");
        println!("  -v, --verbose   Show all scanned files");
        println!("  -s, --strict    Treat comment markers (FIXME, HACK, etc.) as errors too");
        println!("  -h, --help      Show this help");
        println!();
        println!("Hard blockers (always fail):");
        for &(pat, desc, is_macro) in PATTERNS {
            if is_macro {
                println!("  {pat:<30} {desc}");
            }
        }
        println!();
        println!("Soft warnings (fail only with --strict):");
        for &(pat, desc, is_macro) in PATTERNS {
            if !is_macro {
                println!("  {pat:<30} {desc}");
            }
        }
        return ExitCode::SUCCESS;
    }

    println!(
        "Scanning for unfinished code in: {}",
        workspace_root.display()
    );
    println!();

    let mut findings: Vec<Finding> = Vec::new();
    let mut files_scanned = 0u32;

    for entry in WalkDir::new(&workspace_root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            !SKIP_DIRS.contains(&name)
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }

        // Skip this tool's own source file — it necessarily contains all patterns.
        if let Ok(rel) = path.strip_prefix(&workspace_root) {
            let rel_str = rel.to_string_lossy();
            if SKIP_FILES.iter().any(|sf| rel_str.ends_with(sf)) {
                continue;
            }
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        files_scanned += 1;
        if verbose {
            println!(
                "  scanning: {}",
                path.strip_prefix(&workspace_root).unwrap_or(path).display()
            );
        }

        let mut in_test_block = false;

        for (line_no_0, line) in content.lines().enumerate() {
            let trimmed = line.trim();

            // Track test module/cfg(test) blocks — skip test code entirely.
            if trimmed.contains("#[cfg(test)]") || trimmed == "mod tests {" {
                in_test_block = true;
                continue;
            }

            if in_test_block {
                continue;
            }

            // Skip lines that match the allow list.
            if ALLOW_LIST.iter().any(|allow| line.contains(allow)) {
                continue;
            }

            for &(pattern, description, is_macro) in PATTERNS {
                let matched = if is_macro {
                    line.contains(pattern)
                } else {
                    line.to_ascii_uppercase()
                        .contains(&pattern.to_ascii_uppercase())
                };

                if !matched {
                    continue;
                }

                // For comment markers, only flag if they appear in a comment
                // or string literal context — not as part of an identifier.
                if !is_macro && !is_marker_in_comment_or_string(line, pattern) {
                    continue;
                }

                findings.push(Finding {
                    path: path.to_path_buf(),
                    line_no: line_no_0 + 1,
                    line: trimmed.to_string(),
                    _pattern: pattern,
                    description,
                    is_hard: is_macro,
                });

                // Only report first match per line.
                break;
            }
        }
    }

    println!("Scanned {files_scanned} .rs files");
    println!();

    if findings.is_empty() {
        println!("No unfinished code found. Clean!");
        return ExitCode::SUCCESS;
    }

    // Partition into hard blockers and soft warnings.
    let hard: Vec<&Finding> = findings.iter().filter(|f| f.is_hard).collect();
    let soft: Vec<&Finding> = findings.iter().filter(|f| !f.is_hard).collect();

    if !hard.is_empty() {
        println!("ERRORS — hard blockers ({} found):", hard.len());
        println!();
        for f in &hard {
            let rel = f.path.strip_prefix(&workspace_root).unwrap_or(&f.path);
            println!(
                "  {}:{}: {} [{}]",
                rel.display(),
                f.line_no,
                f.line,
                f.description,
            );
        }
        println!();
    }

    if !soft.is_empty() {
        let label = if strict { "ERRORS" } else { "WARNINGS" };
        println!("{label} — comment markers ({} found):", soft.len());
        println!();
        for f in &soft {
            let rel = f.path.strip_prefix(&workspace_root).unwrap_or(&f.path);
            println!(
                "  {}:{}: {} [{}]",
                rel.display(),
                f.line_no,
                f.line,
                f.description,
            );
        }
        println!();
    }

    // Summary
    let total_errors = if strict { findings.len() } else { hard.len() };

    if total_errors > 0 {
        println!(
            "FAILED: {} error(s), {} warning(s)",
            if strict { findings.len() } else { hard.len() },
            if strict { 0 } else { soft.len() },
        );
        ExitCode::FAILURE
    } else {
        println!(
            "PASSED with {} warning(s) (use --strict to treat as errors)",
            soft.len()
        );
        ExitCode::SUCCESS
    }
}

/// Check whether a marker appears inside a comment (`//`, `///`, `//!`, `/* */`)
/// or a string literal context on the given line, rather than as part of an
/// identifier like `MyHackProcessor`.
fn is_marker_in_comment_or_string(line: &str, marker: &str) -> bool {
    let upper_line = line.to_ascii_uppercase();
    let upper_marker = marker.to_ascii_uppercase();

    let Some(pos) = upper_line.find(&upper_marker) else {
        return false;
    };

    // Check if a line comment (`//`) precedes the marker.
    if let Some(comment_start) = line.find("//")
        && comment_start < pos
    {
        return true;
    }

    // Check if inside a block comment.
    if let Some(block_start) = line.find("/*")
        && block_start < pos
    {
        return true;
    }

    // Check if inside a string literal (very rough heuristic: odd number of
    // unescaped quotes before the position).
    let before = &line[..pos];
    let quote_count = before.chars().filter(|&c| c == '"').count();
    if quote_count % 2 == 1 {
        return true;
    }

    // Check word boundary: the marker should not be embedded in an identifier.
    // e.g. `MyHackProcessor` should NOT match, but `// HACK:` should.
    let before_char = line[..pos].chars().last();
    let after_pos = pos + marker.len();
    let after_char = line
        .get(after_pos..after_pos + 1)
        .and_then(|s| s.chars().next());

    let before_is_boundary = match before_char {
        None => true,
        Some(c) => !c.is_ascii_alphanumeric() && c != '_',
    };
    let after_is_boundary = match after_char {
        None => true,
        Some(c) => !c.is_ascii_alphanumeric() && c != '_',
    };

    before_is_boundary && after_is_boundary
}

fn workspace_root() -> PathBuf {
    let xtask_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    xtask_dir
        .parent()
        .expect("xtask should be inside workspace")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_in_comment() {
        assert!(is_marker_in_comment_or_string("// TODO: fix this", "TODO"));
        assert!(is_marker_in_comment_or_string("/// FIXME: broken", "FIXME"));
        assert!(is_marker_in_comment_or_string(
            "//! HACK: workaround",
            "HACK"
        ));
        assert!(is_marker_in_comment_or_string("/* XXX */", "XXX"));
    }

    #[test]
    fn test_marker_in_string() {
        assert!(is_marker_in_comment_or_string(
            r#"let msg = "not yet implemented";"#,
            "not yet implemented"
        ));
    }

    #[test]
    fn test_marker_in_identifier_rejected() {
        // Should NOT match — HACK is part of an identifier
        assert!(!is_marker_in_comment_or_string(
            "let hackProcessor = MyHackProcessor::new();",
            "HACK"
        ));
    }

    #[test]
    fn test_marker_standalone() {
        // HACK followed by colon — should match
        assert!(is_marker_in_comment_or_string(
            "// HACK: workaround",
            "HACK"
        ));
        // FIXME at end of comment
        assert!(is_marker_in_comment_or_string("// FIXME", "FIXME"));
    }

    #[test]
    fn test_marker_case_insensitive() {
        assert!(is_marker_in_comment_or_string("// fixme: thing", "FIXME"));
        assert!(is_marker_in_comment_or_string("// Fixme: thing", "FIXME"));
    }

    #[test]
    fn test_todo_macro_not_confused_with_comment() {
        // The macro pattern matching is separate, but the marker check
        // should still work for `// TODO` comments
        assert!(is_marker_in_comment_or_string(
            "// TODO: finish this",
            "TODO"
        ));
    }
}
