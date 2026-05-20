//! Full-text search helpers
//!
//! Uses SQLite FTS5 virtual tables for efficient text search across
//! skill names and descriptions.

use anyhow::{Context, Result};
use brainwires_agent::skills::{SkillManifest, SkillPackage};
use rusqlite::Connection;

/// Ensure the FTS5 virtual table exists.
pub fn ensure_fts_table(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
            name,
            description,
            tags,
            content='skills',
            content_rowid='rowid'
        );
        ",
    )
    .context("Failed to create FTS5 table")?;
    Ok(())
}

/// Index a skill package into the FTS table.
pub fn index_skill(conn: &Connection, package: &SkillPackage) -> Result<()> {
    let tags_text = package.manifest.tags.join(" ");

    // FTS5 content-sync: we insert directly into the FTS table.
    // Use INSERT OR REPLACE semantics via delete+insert for idempotency.
    conn.execute(
        "INSERT INTO skills_fts(skills_fts, rowid, name, description, tags) VALUES('delete',
            (SELECT rowid FROM skills WHERE name = ?1 AND version = ?2),
            ?1, ?3, ?4)",
        rusqlite::params![
            package.manifest.name,
            package.manifest.version.to_string(),
            package.manifest.description,
            tags_text,
        ],
    )
    .ok(); // Ignore errors from delete of non-existent rows

    conn.execute(
        "INSERT INTO skills_fts(rowid, name, description, tags) VALUES(
            (SELECT rowid FROM skills WHERE name = ?1 AND version = ?2),
            ?1, ?3, ?4)",
        rusqlite::params![
            package.manifest.name,
            package.manifest.version.to_string(),
            package.manifest.description,
            tags_text,
        ],
    )
    .context("Failed to index skill into FTS")?;

    Ok(())
}

/// Search skills using FTS5, optionally filtering by tags.
///
/// Returns manifests for the latest version of each matching skill.
pub fn search_skills(
    conn: &Connection,
    query: &str,
    tags: Option<&[String]>,
    limit: u32,
) -> Result<Vec<SkillManifest>> {
    // Tokenize query for FTS5 — wrap each word with wildcards for prefix matching
    let fts_query = tokenize_query(query);

    let sql = if tags.is_some() {
        "SELECT DISTINCT s.manifest FROM skills_fts f
         JOIN skills s ON s.rowid = f.rowid
         JOIN tags t ON t.skill_name = s.name AND t.skill_version = s.version
         WHERE skills_fts MATCH ?1
         AND t.tag IN (SELECT value FROM json_each(?2))
         ORDER BY rank
         LIMIT ?3"
    } else {
        "SELECT DISTINCT s.manifest FROM skills_fts f
         JOIN skills s ON s.rowid = f.rowid
         WHERE skills_fts MATCH ?1
         ORDER BY rank
         LIMIT ?2"
    };

    let mut stmt = conn
        .prepare(sql)
        .context("Failed to prepare search query")?;

    let rows: Vec<String> = if let Some(tags) = tags {
        let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".to_string());
        let mapped = stmt.query_map(rusqlite::params![fts_query, tags_json, limit], |row| {
            row.get::<_, String>(0)
        })?;
        mapped.filter_map(|r| r.ok()).collect()
    } else {
        let mapped = stmt.query_map(rusqlite::params![fts_query, limit], |row| {
            row.get::<_, String>(0)
        })?;
        mapped.filter_map(|r| r.ok()).collect()
    };

    let mut results = Vec::new();
    for json in rows {
        if let Ok(manifest) = serde_json::from_str::<SkillManifest>(&json) {
            results.push(manifest);
        }
    }

    Ok(results)
}

/// Tokenize a user query into an FTS5 match expression.
///
/// Splits on whitespace, wraps each token with `*` for prefix matching,
/// and joins with OR for broad matching.
fn tokenize_query(query: &str) -> String {
    let tokens: Vec<String> = query
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .map(|t| {
            // Strip non-alphanumeric for safety, keep hyphens
            let clean: String = t
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            if clean.is_empty() {
                return String::new();
            }
            format!("\"{}\"*", clean)
        })
        .filter(|t| !t.is_empty())
        .collect();

    if tokens.is_empty() {
        "\"\"".to_string()
    } else {
        tokens.join(" OR ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_query() {
        assert_eq!(tokenize_query("review pr"), "\"review\"* OR \"pr\"*");
        assert_eq!(tokenize_query("lint-code"), "\"lint-code\"*");
        assert_eq!(tokenize_query("  "), "\"\"");
    }
}
