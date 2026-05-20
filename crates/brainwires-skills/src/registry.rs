//! Skill Registry
//!
//! Central registry for managing Agent Skills with progressive disclosure.
//!
//! # Progressive Disclosure
//!
//! - At startup: Only metadata (name, description) is loaded
//! - On activation: Full content is loaded on-demand and cached
//!
//! # Discovery
//!
//! Use [`SkillRegistry::discover_from`] with explicit path + source pairs.
//! The CLI adapter provides platform-specific paths (personal + project dirs).

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::metadata::{Skill, SkillMetadata, SkillResources, SkillSource};
use super::parser;

/// Skill registry managing all available skills
pub struct SkillRegistry {
    /// Skills indexed by name (metadata only at startup)
    skills: HashMap<String, SkillMetadata>,
    /// Cache of fully loaded skills (loaded on-demand)
    loaded_cache: HashMap<String, Skill>,
    /// Paths used for the last discovery (stored for `reload`)
    discovery_paths: Vec<(PathBuf, SkillSource)>,
}

impl SkillRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
            loaded_cache: HashMap::new(),
            discovery_paths: Vec::new(),
        }
    }

    /// Discover skills from explicit path + source pairs.
    ///
    /// Clears existing skills, then loads metadata from each provided directory.
    /// Paths provided later in the slice override earlier ones for same-named skills
    /// (so project skills can override personal skills when passed last).
    pub fn discover_from(&mut self, paths: &[(PathBuf, SkillSource)]) -> Result<()> {
        self.skills.clear();
        self.loaded_cache.clear();
        self.discovery_paths = paths.to_vec();

        for (path, source) in paths {
            if path.exists() {
                self.load_from_directory(path, *source)?;
            }
        }

        tracing::info!("Discovered {} skills", self.skills.len());
        Ok(())
    }

    /// Reload skills using the same paths as the last `discover_from` call.
    pub fn reload(&mut self) -> Result<()> {
        tracing::info!("Reloading skills from disk");
        let paths = self.discovery_paths.clone();
        self.discover_from(&paths)
    }

    /// Load skill metadata from a directory
    fn load_from_directory(&mut self, dir: &Path, source: SkillSource) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("Failed to read skills directory: {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();

            // Check for SKILL.md files in subdirectories or direct .md files
            if path.is_dir() {
                // Skill in subdirectory: skill-name/SKILL.md
                let skill_file = path.join("SKILL.md");
                if skill_file.exists() {
                    self.load_skill_file(&skill_file, source)?;
                }
            } else if path.extension().and_then(|s| s.to_str()) == Some("md") {
                // Direct .md file: skill-name.md
                self.load_skill_file(&path, source)?;
            }
        }

        Ok(())
    }

    /// Load a single skill file (metadata only)
    fn load_skill_file(&mut self, path: &Path, source: SkillSource) -> Result<()> {
        match parser::parse_skill_metadata(path) {
            Ok(mut metadata) => {
                metadata.source = source;
                metadata.source_path = path.to_path_buf();

                // For subdirectory layout (skill-name/SKILL.md), record the parent dir
                // so get_resources() can lazily discover Level 3 resource files.
                if path.file_name().map(|f| f == "SKILL.md").unwrap_or(false) {
                    metadata.resources_dir = path.parent().map(|p| p.to_path_buf());
                }

                tracing::debug!(
                    "Loaded skill '{}' from {} ({})",
                    metadata.name,
                    path.display(),
                    source
                );

                // Project skills override personal skills with same name
                if source == SkillSource::Project || !self.skills.contains_key(&metadata.name) {
                    self.skills.insert(metadata.name.clone(), metadata);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to load skill from {}: {}", path.display(), e);
            }
        }

        Ok(())
    }

    /// Register a skill directly (for built-in skills)
    pub fn register(&mut self, metadata: SkillMetadata) {
        self.skills.insert(metadata.name.clone(), metadata);
    }

    /// Get skill metadata by name
    pub fn get_metadata(&self, name: &str) -> Option<&SkillMetadata> {
        self.skills.get(name)
    }

    /// Lazy load full skill content
    ///
    /// Returns cached skill if already loaded, otherwise loads from disk.
    pub fn get_skill(&mut self, name: &str) -> Result<&Skill> {
        if !self.loaded_cache.contains_key(name) {
            let metadata = self
                .skills
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", name))?;

            let skill = parser::parse_skill_file(&metadata.source_path)
                .with_context(|| format!("Failed to load skill '{}' from disk", name))?;

            self.loaded_cache.insert(name.to_string(), skill);
        }

        Ok(self
            .loaded_cache
            .get(name)
            .expect("just inserted into cache"))
    }

    /// Get a mutable reference to the full skill
    pub fn get_skill_mut(&mut self, name: &str) -> Result<&mut Skill> {
        if !self.loaded_cache.contains_key(name) {
            let metadata = self
                .skills
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", name))?;

            let skill = parser::parse_skill_file(&metadata.source_path)?;
            self.loaded_cache.insert(name.to_string(), skill);
        }

        Ok(self
            .loaded_cache
            .get_mut(name)
            .expect("just inserted into cache"))
    }

    /// Check if a skill exists
    pub fn contains(&self, name: &str) -> bool {
        self.skills.contains_key(name)
    }

    /// List all skill names
    pub fn list_skills(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.skills.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Get all metadata for semantic matching
    pub fn all_metadata(&self) -> Vec<&SkillMetadata> {
        self.skills.values().collect()
    }

    /// Get metadata for all skills from a specific source
    pub fn skills_by_source(&self, source: SkillSource) -> Vec<&SkillMetadata> {
        self.skills
            .values()
            .filter(|m| m.source == source)
            .collect()
    }

    /// Get the number of registered skills
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Clear the loaded skill cache
    ///
    /// Forces skills to be reloaded from disk on next access.
    pub fn clear_cache(&mut self) {
        self.loaded_cache.clear();
    }

    /// Download a skill from a remote registry and register it locally.
    ///
    /// The downloaded SKILL.md content is written into `install_dir` and then
    /// loaded into the in-memory registry.
    #[cfg(feature = "skills-registry")]
    pub async fn install_from_registry(
        &mut self,
        client: &super::registry_client::RegistryClient,
        name: &str,
        version_req: &semver::VersionReq,
        install_dir: &Path,
    ) -> Result<()> {
        let package = client
            .download(name, version_req)
            .await
            .with_context(|| format!("Failed to download skill '{}'", name))?;

        if !package.verify_checksum() {
            anyhow::bail!("Checksum verification failed for skill '{}'", name);
        }

        // Write SKILL.md into install_dir/<name>/SKILL.md
        let skill_dir = install_dir.join(name);
        std::fs::create_dir_all(&skill_dir)
            .with_context(|| format!("Failed to create directory {}", skill_dir.display()))?;
        let skill_path = skill_dir.join("SKILL.md");
        std::fs::write(&skill_path, &package.skill_content)
            .with_context(|| format!("Failed to write {}", skill_path.display()))?;

        // Load into registry
        self.load_skill_file(&skill_path, SkillSource::Personal)?;

        tracing::info!(
            "Installed skill '{}' v{} from registry",
            name,
            package.manifest.version
        );
        Ok(())
    }

    /// Package a local skill and publish it to a remote registry.
    #[cfg(feature = "skills-registry")]
    pub async fn publish_to_registry(
        &mut self,
        client: &super::registry_client::RegistryClient,
        skill_name: &str,
        manifest: super::manifest::SkillManifest,
    ) -> Result<()> {
        let metadata = self
            .skills
            .get(skill_name)
            .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", skill_name))?;

        let package =
            super::package::SkillPackage::from_skill_file(&metadata.source_path, manifest)
                .with_context(|| format!("Failed to package skill '{}'", skill_name))?;

        client
            .publish(&package)
            .await
            .with_context(|| format!("Failed to publish skill '{}'", skill_name))?;

        tracing::info!("Published skill '{}' to registry", skill_name);
        Ok(())
    }

    /// Remove a skill from the registry
    pub fn remove(&mut self, name: &str) -> Option<SkillMetadata> {
        self.loaded_cache.remove(name);
        self.skills.remove(name)
    }

    /// Discover Level 3 resource files for a skill (on-demand).
    ///
    /// Returns the files found in `scripts/`, `references/`, and `assets/`
    /// subdirectories inside the skill's directory. Only available for skills
    /// using the subdirectory layout (`skill-name/SKILL.md`).
    ///
    /// These are never loaded automatically — this method lets agents discover
    /// what supplementary files are available and request them as needed.
    pub fn get_resources(&self, name: &str) -> Result<SkillResources> {
        let metadata = self
            .skills
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", name))?;

        let Some(ref dir) = metadata.resources_dir else {
            return Ok(SkillResources::default());
        };

        Ok(SkillResources {
            scripts: collect_files(&dir.join("scripts")),
            references: collect_files(&dir.join("references")),
            assets: collect_files(&dir.join("assets")),
        })
    }

    /// Get skills that match a category
    pub fn skills_by_category(&self, category: &str) -> Vec<&SkillMetadata> {
        self.skills
            .values()
            .filter(|m| {
                m.metadata
                    .as_ref()
                    .and_then(|meta| meta.get("category"))
                    .map(|c| c == category)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Format a skill listing for display
    pub fn format_skill_list(&self) -> String {
        if self.skills.is_empty() {
            return "No skills available. Add skills to ~/.brainwires/skills/ or .brainwires/skills/".to_string();
        }

        let mut output = String::new();
        let mut personal: Vec<_> = self.skills_by_source(SkillSource::Personal);
        let mut project: Vec<_> = self.skills_by_source(SkillSource::Project);

        personal.sort_by(|a, b| a.name.cmp(&b.name));
        project.sort_by(|a, b| a.name.cmp(&b.name));

        if !project.is_empty() {
            output.push_str("## Project Skills\n\n");
            for skill in &project {
                output.push_str(&format!(
                    "- **{}**: {}\n",
                    skill.name,
                    truncate_description(&skill.description, 60)
                ));
            }
            output.push('\n');
        }

        if !personal.is_empty() {
            output.push_str("## Personal Skills\n\n");
            for skill in &personal {
                // Skip if overridden by project skill
                if project.iter().any(|p| p.name == skill.name) {
                    continue;
                }
                output.push_str(&format!(
                    "- **{}**: {}\n",
                    skill.name,
                    truncate_description(&skill.description, 60)
                ));
            }
        }

        output.push_str("\nUse `/skill <name>` to invoke a skill.\n");

        output
    }

    /// Format detailed skill info for display
    pub fn format_skill_detail(&self, name: &str) -> Result<String> {
        let metadata = self
            .get_metadata(name)
            .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", name))?;

        let mut output = String::new();
        output.push_str(&format!("# {}\n\n", metadata.name));
        output.push_str(&format!("**Description**: {}\n\n", metadata.description));
        output.push_str(&format!("**Source**: {}\n", metadata.source));
        output.push_str(&format!(
            "**Execution Mode**: {}\n",
            metadata.execution_mode()
        ));

        if let Some(ref tools) = metadata.allowed_tools {
            output.push_str(&format!("**Allowed Tools**: {}\n", tools.join(", ")));
        }

        if let Some(ref license) = metadata.license {
            output.push_str(&format!("**License**: {}\n", license));
        }

        if let Some(ref model) = metadata.model {
            output.push_str(&format!("**Model**: {}\n", model));
        }

        if let Some(ref meta) = metadata.metadata
            && !meta.is_empty()
        {
            output.push_str("\n**Metadata**:\n");
            for (key, value) in meta {
                output.push_str(&format!("  - {}: {}\n", key, value));
            }
        }

        output.push_str(&format!("\n**File**: {}\n", metadata.source_path.display()));

        // Show Level 3 resources if available
        if let Ok(resources) = self.get_resources(name)
            && !resources.is_empty()
        {
            output.push_str(&format!(
                "\n**Resources**: {} file(s) (scripts: {}, references: {}, assets: {})\n",
                resources.total_count(),
                resources.scripts.len(),
                resources.references.len(),
                resources.assets.len(),
            ));
        }

        Ok(output)
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Collect all files from a directory, returning their paths.
/// Returns an empty vec if the directory doesn't exist.
fn collect_files(dir: &std::path::Path) -> Vec<PathBuf> {
    if !dir.exists() {
        return Vec::new();
    }
    match std::fs::read_dir(dir) {
        Ok(entries) => {
            let mut files: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .collect();
            files.sort();
            files
        }
        Err(_) => Vec::new(),
    }
}

/// Truncate a description to a maximum length
fn truncate_description(desc: &str, max_len: usize) -> String {
    let first_line = desc.lines().next().unwrap_or(desc);
    if first_line.len() <= max_len {
        first_line.to_string()
    } else {
        format!("{}...", &first_line[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_skill(dir: &Path, name: &str, description: &str) -> PathBuf {
        let content = format!(
            r#"---
name: {}
description: {}
---

# {} Instructions

Do the thing."#,
            name, description, name
        );

        let path = dir.join(format!("{}.md", name));
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_registry_new() {
        let registry = SkillRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_register() {
        let mut registry = SkillRegistry::new();
        let metadata = SkillMetadata::new("test".to_string(), "A test skill".to_string());

        registry.register(metadata);

        assert!(registry.contains("test"));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_get_metadata() {
        let mut registry = SkillRegistry::new();
        let metadata = SkillMetadata::new("test".to_string(), "A test skill".to_string());

        registry.register(metadata);

        let retrieved = registry.get_metadata("test").unwrap();
        assert_eq!(retrieved.name, "test");
        assert_eq!(retrieved.description, "A test skill");

        assert!(registry.get_metadata("nonexistent").is_none());
    }

    #[test]
    fn test_registry_list_skills() {
        let mut registry = SkillRegistry::new();
        registry.register(SkillMetadata::new(
            "zebra".to_string(),
            "Z skill".to_string(),
        ));
        registry.register(SkillMetadata::new(
            "alpha".to_string(),
            "A skill".to_string(),
        ));
        registry.register(SkillMetadata::new(
            "beta".to_string(),
            "B skill".to_string(),
        ));

        let names = registry.list_skills();
        assert_eq!(names, vec!["alpha", "beta", "zebra"]); // Sorted
    }

    #[test]
    fn test_registry_load_from_directory() {
        let temp = TempDir::new().unwrap();

        create_test_skill(temp.path(), "skill-a", "First skill");
        create_test_skill(temp.path(), "skill-b", "Second skill");

        let mut registry = SkillRegistry::new();
        registry
            .load_from_directory(temp.path(), SkillSource::Personal)
            .unwrap();

        assert_eq!(registry.len(), 2);
        assert!(registry.contains("skill-a"));
        assert!(registry.contains("skill-b"));
    }

    #[test]
    fn test_registry_load_subdirectory_skill() {
        let temp = TempDir::new().unwrap();

        // Create skill in subdirectory
        let skill_dir = temp.path().join("my-skill");
        std::fs::create_dir(&skill_dir).unwrap();

        let content = r#"---
name: my-skill
description: A skill in a subdirectory
---

Instructions"#;
        std::fs::write(skill_dir.join("SKILL.md"), content).unwrap();

        let mut registry = SkillRegistry::new();
        registry
            .load_from_directory(temp.path(), SkillSource::Project)
            .unwrap();

        assert!(registry.contains("my-skill"));
        assert_eq!(
            registry.get_metadata("my-skill").unwrap().source,
            SkillSource::Project
        );
    }

    #[test]
    fn test_registry_project_overrides_personal() {
        let temp = TempDir::new().unwrap();

        let personal_dir = temp.path().join("personal");
        let project_dir = temp.path().join("project");
        std::fs::create_dir(&personal_dir).unwrap();
        std::fs::create_dir(&project_dir).unwrap();

        create_test_skill(&personal_dir, "same-skill", "Personal version");
        create_test_skill(&project_dir, "same-skill", "Project version");

        let mut registry = SkillRegistry::new();
        registry
            .discover_from(&[
                (personal_dir, SkillSource::Personal),
                (project_dir, SkillSource::Project),
            ])
            .unwrap();

        // Project should take precedence
        let metadata = registry.get_metadata("same-skill").unwrap();
        assert_eq!(metadata.source, SkillSource::Project);
        assert_eq!(metadata.description, "Project version");
    }

    #[test]
    fn test_registry_get_skill_lazy_load() {
        let temp = TempDir::new().unwrap();
        create_test_skill(temp.path(), "lazy-skill", "A lazily loaded skill");

        let mut registry = SkillRegistry::new();
        registry
            .load_from_directory(temp.path(), SkillSource::Personal)
            .unwrap();

        // Cache should be empty
        assert!(registry.loaded_cache.is_empty());

        // Get full skill (triggers lazy load)
        let skill = registry.get_skill("lazy-skill").unwrap();
        assert_eq!(skill.metadata.name, "lazy-skill");
        assert!(skill.instructions.contains("Instructions"));

        // Cache should now have the skill
        assert!(registry.loaded_cache.contains_key("lazy-skill"));
    }

    #[test]
    fn test_registry_reload() {
        let temp = TempDir::new().unwrap();
        create_test_skill(temp.path(), "original", "Original skill");

        let path = temp.path().to_path_buf();
        let mut registry = SkillRegistry::new();
        registry
            .discover_from(&[(path.clone(), SkillSource::Personal)])
            .unwrap();

        assert_eq!(registry.len(), 1);

        // Add another skill
        create_test_skill(temp.path(), "new-skill", "New skill");

        // Reload should pick up new skill
        registry.reload().unwrap();

        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_truncate_description() {
        assert_eq!(truncate_description("Short", 10), "Short");
        assert_eq!(
            truncate_description("This is a long description", 15),
            "This is a lo..."
        );
        assert_eq!(
            truncate_description("Line 1\nLine 2\nLine 3", 100),
            "Line 1"
        );
    }

    #[test]
    fn test_skills_by_category() {
        use std::collections::HashMap;

        let mut registry = SkillRegistry::new();

        let mut meta1 = HashMap::new();
        meta1.insert("category".to_string(), "testing".to_string());

        let mut skill1 = SkillMetadata::new("skill1".to_string(), "Desc".to_string());
        skill1.metadata = Some(meta1);

        let skill2 = SkillMetadata::new("skill2".to_string(), "Desc".to_string());

        registry.register(skill1);
        registry.register(skill2);

        let testing_skills = registry.skills_by_category("testing");
        assert_eq!(testing_skills.len(), 1);
        assert_eq!(testing_skills[0].name, "skill1");
    }
}
