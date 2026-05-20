//! Template Store - Storage for reusable plan templates
//!
//! Templates are saved plans that can be instantiated for new tasks.
//! Uses simple JSON file storage for simplicity.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use std::path::Path;

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanTemplate {
    /// Unique template ID
    pub template_id: String,

    /// Template name (user-friendly)
    pub name: String,

    /// Template description
    pub description: String,

    /// The template content (markdown with placeholders)
    pub content: String,

    /// Category for organization (e.g., "feature", "bugfix", "refactor")
    pub category: Option<String>,

    /// Tags for discovery
    pub tags: Vec<String>,

    /// Variables/placeholders in the template (e.g., ["{{component}}", "{{feature}}"])
    pub variables: Vec<String>,

    /// Original plan ID this template was derived from (if any)
    pub source_plan_id: Option<String>,

    /// Number of times this template has been used
    pub usage_count: u32,

    /// Creation timestamp
    pub created_at: i64,

    /// Last used timestamp
    pub last_used_at: Option<i64>,
}

impl PlanTemplate {
    /// Create a new template from a plan
    pub fn new(name: String, description: String, content: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        let template_id = uuid::Uuid::new_v4().to_string();

        // Extract variables from content (format: {{variable_name}})
        let variables = Self::extract_variables(&content);

        Self {
            template_id,
            name,
            description,
            content,
            category: None,
            tags: vec![],
            variables,
            source_plan_id: None,
            usage_count: 0,
            created_at: now,
            last_used_at: None,
        }
    }

    /// Create from an existing plan
    pub fn from_plan(
        name: String,
        description: String,
        plan_content: String,
        plan_id: String,
    ) -> Self {
        let mut template = Self::new(name, description, plan_content);
        template.source_plan_id = Some(plan_id);
        template
    }

    /// Set category
    pub fn with_category(mut self, category: String) -> Self {
        self.category = Some(category);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Extract variables from template content
    fn extract_variables(content: &str) -> Vec<String> {
        use regex::Regex;
        use std::sync::LazyLock;
        static RE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}").expect("valid regex"));
        let re = &*RE;
        let mut vars: Vec<String> = re
            .captures_iter(content)
            .map(|cap| cap[1].to_string())
            .collect();
        vars.sort();
        vars.dedup();
        vars
    }

    /// Instantiate template with variable substitutions
    pub fn instantiate(&self, substitutions: &std::collections::HashMap<String, String>) -> String {
        let mut result = self.content.clone();
        for (var, value) in substitutions {
            let placeholder = format!("{{{{{}}}}}", var);
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Increment usage count and update last_used_at
    pub fn mark_used(&mut self) {
        self.usage_count += 1;
        self.last_used_at = Some(chrono::Utc::now().timestamp());
    }
}

/// All templates stored together
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TemplateData {
    templates: Vec<PlanTemplate>,
}

/// Template store for persistence using JSON file
pub struct TemplateStore {
    file_path: PathBuf,
}

impl TemplateStore {
    /// Create a new template store
    ///
    /// `data_dir` is the directory where templates.json will be stored.
    pub fn new(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref();
        std::fs::create_dir_all(data_dir)?;
        let file_path = data_dir.join("templates.json");
        Ok(Self { file_path })
    }

    /// Load all templates from file
    fn load(&self) -> Result<TemplateData> {
        if !self.file_path.exists() {
            return Ok(TemplateData::default());
        }
        let content =
            std::fs::read_to_string(&self.file_path).context("Failed to read templates file")?;
        let data: TemplateData =
            serde_json::from_str(&content).context("Failed to parse templates file")?;
        Ok(data)
    }

    /// Save all templates to file
    fn save_data(&self, data: &TemplateData) -> Result<()> {
        let content =
            serde_json::to_string_pretty(data).context("Failed to serialize templates")?;
        std::fs::write(&self.file_path, content).context("Failed to write templates file")?;
        Ok(())
    }

    /// Save a template
    pub fn save(&self, template: &PlanTemplate) -> Result<()> {
        let mut data = self.load()?;

        // Remove existing template with same ID if any
        data.templates
            .retain(|t| t.template_id != template.template_id);

        // Add new template
        data.templates.push(template.clone());

        self.save_data(&data)
    }

    /// Get a template by ID
    pub fn get(&self, template_id: &str) -> Result<Option<PlanTemplate>> {
        let data = self.load()?;
        Ok(data
            .templates
            .into_iter()
            .find(|t| t.template_id == template_id))
    }

    /// Get a template by name (case-insensitive partial match)
    pub fn get_by_name(&self, name: &str) -> Result<Option<PlanTemplate>> {
        let data = self.load()?;
        let name_lower = name.to_lowercase();
        Ok(data.templates.into_iter().find(|t| {
            t.name.to_lowercase().contains(&name_lower) || t.template_id.starts_with(name)
        }))
    }

    /// List all templates
    pub fn list(&self) -> Result<Vec<PlanTemplate>> {
        let data = self.load()?;
        let mut templates = data.templates;
        // Sort by usage count (most used first), then by name
        templates.sort_by(|a, b| {
            b.usage_count
                .cmp(&a.usage_count)
                .then_with(|| a.name.cmp(&b.name))
        });
        Ok(templates)
    }

    /// List templates by category
    pub fn list_by_category(&self, category: &str) -> Result<Vec<PlanTemplate>> {
        let all = self.list()?;
        Ok(all
            .into_iter()
            .filter(|t| t.category.as_deref() == Some(category))
            .collect())
    }

    /// Search templates by name or tags
    pub fn search(&self, query: &str) -> Result<Vec<PlanTemplate>> {
        let all = self.list()?;
        let query_lower = query.to_lowercase();
        Ok(all
            .into_iter()
            .filter(|t| {
                t.name.to_lowercase().contains(&query_lower)
                    || t.description.to_lowercase().contains(&query_lower)
                    || t.tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .collect())
    }

    /// Delete a template
    pub fn delete(&self, template_id: &str) -> Result<bool> {
        let mut data = self.load()?;
        let original_len = data.templates.len();
        data.templates.retain(|t| t.template_id != template_id);
        let deleted = data.templates.len() < original_len;
        if deleted {
            self.save_data(&data)?;
        }
        Ok(deleted)
    }

    /// Update a template (save with same ID)
    pub fn update(&self, template: &PlanTemplate) -> Result<()> {
        self.save(template)
    }

    /// Mark template as used and save
    pub fn mark_used(&self, template_id: &str) -> Result<()> {
        if let Some(mut template) = self.get(template_id)? {
            template.mark_used();
            self.save(&template)?;
        }
        Ok(())
    }
}

impl TemplateStore {
    /// Create a template store in the system's default data directory
    ///
    /// Falls back to ~/.brainwires/ if no platform data dir is available.
    pub fn with_default_dir() -> Result<Self> {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".brainwires"));
        let data_dir = data_dir.join("brainwires");
        Self::new(data_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_template_creation() {
        let template = PlanTemplate::new(
            "Feature Implementation".to_string(),
            "Template for implementing new features".to_string(),
            "1. Create {{component}} component\n2. Add tests for {{feature}}".to_string(),
        );

        assert!(!template.template_id.is_empty());
        assert_eq!(template.name, "Feature Implementation");
        assert_eq!(template.variables.len(), 2);
        assert!(template.variables.contains(&"component".to_string()));
        assert!(template.variables.contains(&"feature".to_string()));
    }

    #[test]
    fn test_template_instantiation() {
        let template = PlanTemplate::new(
            "Test".to_string(),
            "Test template".to_string(),
            "Implement {{feature}} in {{module}}".to_string(),
        );

        let mut subs = HashMap::new();
        subs.insert("feature".to_string(), "authentication".to_string());
        subs.insert("module".to_string(), "auth".to_string());

        let result = template.instantiate(&subs);
        assert_eq!(result, "Implement authentication in auth");
    }

    #[test]
    fn test_extract_variables() {
        let content = "{{var1}} and {{var2}} and {{var1}} again";
        let vars = PlanTemplate::extract_variables(content);

        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&"var1".to_string()));
        assert!(vars.contains(&"var2".to_string()));
    }

    #[test]
    fn test_from_plan() {
        let template = PlanTemplate::from_plan(
            "My Template".to_string(),
            "Description".to_string(),
            "Content".to_string(),
            "plan-123".to_string(),
        );

        assert_eq!(template.source_plan_id, Some("plan-123".to_string()));
    }

    #[test]
    fn test_mark_used() {
        let mut template = PlanTemplate::new(
            "Test".to_string(),
            "Test".to_string(),
            "Content".to_string(),
        );

        assert_eq!(template.usage_count, 0);
        assert!(template.last_used_at.is_none());

        template.mark_used();

        assert_eq!(template.usage_count, 1);
        assert!(template.last_used_at.is_some());
    }

    #[test]
    fn test_with_category_and_tags() {
        let template = PlanTemplate::new(
            "Test".to_string(),
            "Test".to_string(),
            "Content".to_string(),
        )
        .with_category("feature".to_string())
        .with_tags(vec!["rust".to_string(), "api".to_string()]);

        assert_eq!(template.category, Some("feature".to_string()));
        assert_eq!(template.tags.len(), 2);
    }
}
