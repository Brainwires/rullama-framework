//! SQLite-backed skill storage
//!
//! Uses SQLite with FTS5 for full-text search over skill metadata.

use anyhow::{Context, Result};
use brainwires_agent::skills::{SkillManifest, SkillPackage};
use rusqlite::Connection;

use super::search;

/// SQLite-backed store for skill packages.
pub struct SkillStore {
    conn: Connection,
}

impl SkillStore {
    /// Open (or create) the database at `path` and initialize tables.
    pub fn open(path: &str) -> Result<Self> {
        let conn =
            Connection::open(path).with_context(|| format!("Failed to open database: {}", path))?;
        let store = Self { conn };
        store.init_db()?;
        Ok(store)
    }

    /// Open an in-memory database (useful for testing).
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.init_db()?;
        Ok(store)
    }

    /// Create tables if they don't exist.
    fn init_db(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS skills (
                name        TEXT NOT NULL,
                version     TEXT NOT NULL,
                description TEXT NOT NULL,
                author      TEXT NOT NULL,
                license     TEXT NOT NULL,
                manifest    TEXT NOT NULL,
                content     TEXT NOT NULL,
                checksum    TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                PRIMARY KEY (name, version)
            );

            CREATE TABLE IF NOT EXISTS tags (
                skill_name    TEXT NOT NULL,
                skill_version TEXT NOT NULL,
                tag           TEXT NOT NULL,
                PRIMARY KEY (skill_name, skill_version, tag),
                FOREIGN KEY (skill_name, skill_version) REFERENCES skills(name, version)
            );
            ",
        )?;

        search::ensure_fts_table(&self.conn)?;

        Ok(())
    }

    /// Insert a new skill version.
    pub fn insert_skill(&self, package: &SkillPackage) -> Result<()> {
        let manifest_json =
            serde_json::to_string(&package.manifest).context("Failed to serialize manifest")?;

        self.conn.execute(
            "INSERT OR REPLACE INTO skills (name, version, description, author, license, manifest, content, checksum, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                package.manifest.name,
                package.manifest.version.to_string(),
                package.manifest.description,
                package.manifest.author,
                package.manifest.license,
                manifest_json,
                package.skill_content,
                package.checksum,
                package.manifest.created_at.to_rfc3339(),
                package.manifest.updated_at.to_rfc3339(),
            ],
        )?;

        // Insert tags
        for tag in &package.manifest.tags {
            self.conn.execute(
                "INSERT OR IGNORE INTO tags (skill_name, skill_version, tag) VALUES (?1, ?2, ?3)",
                rusqlite::params![
                    package.manifest.name,
                    package.manifest.version.to_string(),
                    tag,
                ],
            )?;
        }

        // Update FTS index
        search::index_skill(&self.conn, package)?;

        Ok(())
    }

    /// Full-text search across skill names and descriptions.
    pub fn search(
        &self,
        query: &str,
        tags: Option<&[String]>,
        limit: u32,
    ) -> Result<Vec<SkillManifest>> {
        search::search_skills(&self.conn, query, tags, limit)
    }

    /// Get the manifest for a specific version.
    pub fn get_manifest(&self, name: &str, version: &str) -> Result<Option<SkillManifest>> {
        let mut stmt = self
            .conn
            .prepare("SELECT manifest FROM skills WHERE name = ?1 AND version = ?2")?;

        let result = stmt.query_row(rusqlite::params![name, version], |row| {
            let json: String = row.get(0)?;
            Ok(json)
        });

        match result {
            Ok(json) => {
                let manifest: SkillManifest = serde_json::from_str(&json)?;
                Ok(Some(manifest))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get the latest manifest for a skill (highest semver).
    pub fn get_latest_manifest(&self, name: &str) -> Result<Option<SkillManifest>> {
        let versions = self.list_versions(name)?;
        if let Some(latest) = versions.last() {
            self.get_manifest(name, &latest.to_string())
        } else {
            Ok(None)
        }
    }

    /// Get a full package for download.
    pub fn get_package(&self, name: &str, version: &str) -> Result<Option<SkillPackage>> {
        let mut stmt = self.conn.prepare(
            "SELECT manifest, content, checksum FROM skills WHERE name = ?1 AND version = ?2",
        )?;

        let result = stmt.query_row(rusqlite::params![name, version], |row| {
            let manifest_json: String = row.get(0)?;
            let content: String = row.get(1)?;
            let checksum: String = row.get(2)?;
            Ok((manifest_json, content, checksum))
        });

        match result {
            Ok((manifest_json, content, checksum)) => {
                let manifest: SkillManifest = serde_json::from_str(&manifest_json)?;
                Ok(Some(SkillPackage {
                    manifest,
                    skill_content: content,
                    checksum,
                    signature: None,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List all versions of a skill, sorted by semver ascending.
    pub fn list_versions(&self, name: &str) -> Result<Vec<semver::Version>> {
        let mut stmt = self
            .conn
            .prepare("SELECT version FROM skills WHERE name = ?1")?;

        let rows = stmt.query_map(rusqlite::params![name], |row| {
            let v: String = row.get(0)?;
            Ok(v)
        })?;

        let mut versions: Vec<semver::Version> = Vec::new();
        for row in rows {
            let v_str = row?;
            if let Ok(v) = semver::Version::parse(&v_str) {
                versions.push(v);
            }
        }

        versions.sort();
        Ok(versions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_package(name: &str, version: &str) -> SkillPackage {
        let manifest = SkillManifest {
            name: name.to_string(),
            version: semver::Version::parse(version).unwrap(),
            description: format!("Test skill {}", name),
            author: "Test".to_string(),
            license: "MIT".to_string(),
            tags: vec!["test".to_string()],
            dependencies: vec![],
            min_framework_version: None,
            repository: None,
            signing_key: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let content = format!("# {}\nInstructions", name);
        let checksum = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(content.as_bytes());
            format!("{:x}", h.finalize())
        };
        SkillPackage {
            manifest,
            skill_content: content,
            checksum,
            signature: None,
        }
    }

    #[test]
    fn test_insert_and_get_manifest() {
        let store = SkillStore::open_in_memory().unwrap();
        let pkg = sample_package("my-skill", "1.0.0");
        store.insert_skill(&pkg).unwrap();

        let m = store.get_manifest("my-skill", "1.0.0").unwrap().unwrap();
        assert_eq!(m.name, "my-skill");
        assert_eq!(m.version, semver::Version::new(1, 0, 0));
    }

    #[test]
    fn test_list_versions() {
        let store = SkillStore::open_in_memory().unwrap();
        store
            .insert_skill(&sample_package("my-skill", "0.1.0"))
            .unwrap();
        store
            .insert_skill(&sample_package("my-skill", "1.0.0"))
            .unwrap();
        store
            .insert_skill(&sample_package("my-skill", "0.5.0"))
            .unwrap();

        let versions = store.list_versions("my-skill").unwrap();
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0], semver::Version::new(0, 1, 0));
        assert_eq!(versions[2], semver::Version::new(1, 0, 0));
    }

    #[test]
    fn test_get_package() {
        let store = SkillStore::open_in_memory().unwrap();
        let pkg = sample_package("dl-skill", "2.0.0");
        store.insert_skill(&pkg).unwrap();

        let retrieved = store.get_package("dl-skill", "2.0.0").unwrap().unwrap();
        assert_eq!(retrieved.skill_content, pkg.skill_content);
        assert!(retrieved.verify_checksum());
    }

    #[test]
    fn test_get_latest_manifest() {
        let store = SkillStore::open_in_memory().unwrap();
        store
            .insert_skill(&sample_package("evolving", "0.1.0"))
            .unwrap();
        store
            .insert_skill(&sample_package("evolving", "0.2.0"))
            .unwrap();

        let latest = store.get_latest_manifest("evolving").unwrap().unwrap();
        assert_eq!(latest.version, semver::Version::new(0, 2, 0));
    }

    #[test]
    fn test_search() {
        let store = SkillStore::open_in_memory().unwrap();
        store
            .insert_skill(&sample_package("review-pr", "1.0.0"))
            .unwrap();
        store
            .insert_skill(&sample_package("lint-code", "1.0.0"))
            .unwrap();

        let results = store.search("review", None, 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|m| m.name == "review-pr"));
    }

    #[test]
    fn test_not_found() {
        let store = SkillStore::open_in_memory().unwrap();
        assert!(store.get_manifest("nope", "1.0.0").unwrap().is_none());
        assert!(store.get_package("nope", "1.0.0").unwrap().is_none());
        assert!(store.list_versions("nope").unwrap().is_empty());
    }
}
