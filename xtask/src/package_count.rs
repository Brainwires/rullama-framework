use std::path::{Path, PathBuf};
use std::process::ExitCode;

use walkdir::WalkDir;

/// Count workspace members and update references in `.md` files.
///
/// Members under `crates/` count as "crates"; members under sdks/ servers/
/// integrations/ examples/ (the consumer tier) count as
/// "extras". The `xtask` member and `deprecated/` entries are excluded.
pub fn update_package_count(args: &[String]) -> ExitCode {
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let workspace_root = workspace_root();

    println!("Workspace root: {}", workspace_root.display());

    let cargo_path = workspace_root.join("Cargo.toml");
    let cargo_toml = std::fs::read_to_string(&cargo_path).expect("failed to read Cargo.toml");
    let doc: toml_edit::DocumentMut = cargo_toml
        .parse()
        .expect("failed to parse workspace Cargo.toml");

    let members = doc["workspace"]["members"]
        .as_array()
        .expect("workspace.members should be an array");

    let mut crate_count: usize = 0;
    let mut extra_count: usize = 0;

    for member in members.iter() {
        let path = member.as_str().expect("member should be a string");
        if path == "xtask" {
            continue;
        }
        // The consumer ("extras") tier was reorganized out of a single `extras/`
        // dir into sdks/ servers/ integrations/ examples/.
        if path.starts_with("crates/") {
            crate_count += 1;
        } else if path.starts_with("sdks/")
            || path.starts_with("servers/")
            || path.starts_with("integrations/")
            || path.starts_with("examples/")
        {
            extra_count += 1;
        }
    }

    println!("Counted {crate_count} crates, {extra_count} extras");

    let mut total_replacements = 0u32;

    // Walk all .md files in the workspace root and crate/extras directories
    for entry in WalkDir::new(&workspace_root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            // Skip build artifacts, git, node_modules, deprecated, target
            !matches!(
                name.as_ref(),
                "target" | ".git" | "node_modules" | "deprecated" | "test-results" | "adr"
            )
        })
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str());
        if ext != Some("md") {
            continue;
        }

        // Skip CHANGELOG.md — it's a historical record; counts in past
        // entries should not be rewritten.
        let file_name = entry.file_name().to_string_lossy();
        if file_name == "CHANGELOG.md" {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let updated = update_counts_in_content(&content, crate_count, extra_count);
        if updated != content {
            let rel = path.strip_prefix(&workspace_root).unwrap_or(path);
            let changes = count_line_diffs(&content, &updated);
            if dry_run {
                println!(
                    "  [dry-run] would update {} ({} lines)",
                    rel.display(),
                    changes
                );
            } else {
                std::fs::write(path, &updated).expect("failed to write file");
                println!("  updated {} ({} lines)", rel.display(), changes);
            }
            total_replacements += changes;
        }
    }

    if total_replacements == 0 {
        println!("All counts are already up to date.");
    } else if dry_run {
        println!("{total_replacements} line(s) would be updated (dry run).");
    } else {
        println!("{total_replacements} line(s) updated.");
    }

    ExitCode::SUCCESS
}

/// Update crate and extras counts in markdown content.
///
/// Replaces patterns like `N crates` and `N extras` with the actual counts,
/// skipping historical changelog lines (containing `→` or `->`).
fn update_counts_in_content(content: &str, crates: usize, extras: usize) -> String {
    let mut result = String::with_capacity(content.len());
    let mut in_code_block = false;

    for line in content.lines() {
        let trimmed = line.trim_start();

        // Track fenced code blocks
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            result.push_str(line);
            result.push('\n');
            continue;
        }

        // Skip code blocks
        if in_code_block {
            result.push_str(line);
            result.push('\n');
            continue;
        }

        // Skip historical changelog lines (arrows indicate past state transitions)
        if line.contains("→") || line.contains("->") {
            result.push_str(line);
            result.push('\n');
            continue;
        }

        let mut updated_line = replace_count(line, "crates", crates);
        updated_line = replace_count(&updated_line, "extras", extras);

        result.push_str(&updated_line);
        result.push('\n');
    }

    // Preserve whether the original file ended with a newline
    if !content.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

/// Replace `\d+ <word>` patterns in a line with the correct count.
///
/// Only replaces when the number is preceded by a word boundary (space,
/// punctuation, or start of line) and followed by the exact word.
fn replace_count(line: &str, word: &str, count: usize) -> String {
    let count_str = count.to_string();
    let mut result = String::with_capacity(line.len());
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Look for a digit
        if chars[i].is_ascii_digit() {
            // Check word boundary before the digit
            let at_boundary = i == 0 || !chars[i - 1].is_alphanumeric();
            if at_boundary {
                // Consume all digits
                let start = i;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                // Check for space(s) followed by the target word
                let after_digits = i;
                let mut j = i;
                while j < chars.len() && chars[j] == ' ' {
                    j += 1;
                }
                if j > after_digits {
                    // Check if the word follows
                    let word_chars: Vec<char> = word.chars().collect();
                    let mut matches = true;
                    for (k, wc) in word_chars.iter().enumerate() {
                        if j + k >= chars.len() || chars[j + k] != *wc {
                            matches = false;
                            break;
                        }
                    }
                    // Ensure word boundary after the word
                    if matches {
                        let end = j + word_chars.len();
                        let word_boundary_after =
                            end >= chars.len() || !chars[end].is_alphanumeric();
                        if word_boundary_after {
                            // Replace the number, keep the spacing and word
                            result.push_str(&count_str);
                            // Preserve original spacing
                            for c in &chars[after_digits..j] {
                                result.push(*c);
                            }
                            // Preserve the word
                            for c in &word_chars {
                                result.push(*c);
                            }
                            i = end;
                            continue;
                        }
                    }
                }
                // No match — emit the original digits
                for c in &chars[start..after_digits] {
                    result.push(*c);
                }
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    result
}

/// Count lines that differ between two strings.
fn count_line_diffs(a: &str, b: &str) -> u32 {
    a.lines().zip(b.lines()).filter(|(la, lb)| la != lb).count() as u32
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
    fn replaces_crate_count() {
        let input = "The framework has 19 crates and 7 extras.";
        let result = update_counts_in_content(input, 18, 6);
        assert_eq!(result, "The framework has 18 crates and 6 extras.");
    }

    #[test]
    fn skips_arrow_lines() {
        let input = "#### Crate Merges (23 → 19 crates)";
        let result = update_counts_in_content(input, 18, 6);
        assert_eq!(result, input);
    }

    #[test]
    fn skips_code_blocks() {
        let input = "```\n19 crates here\n```";
        let result = update_counts_in_content(input, 18, 6);
        assert_eq!(result, input);
    }

    #[test]
    fn no_change_when_correct() {
        let input = "We have 18 crates.";
        let result = update_counts_in_content(input, 18, 6);
        assert_eq!(result, input);
    }

    #[test]
    fn replace_count_basic() {
        assert_eq!(
            replace_count("7 layers, 19 crates, leaves first", "crates", 18),
            "7 layers, 18 crates, leaves first"
        );
    }

    #[test]
    fn replace_count_does_not_match_partial_word() {
        // "crates" should not match inside "subcrates"
        assert_eq!(replace_count("19 subcrates", "crates", 18), "19 subcrates");
    }
}
