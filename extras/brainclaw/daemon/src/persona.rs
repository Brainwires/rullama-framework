//! Persona — system prompt resolution for the assistant identity.

use anyhow::{Context, Result};

use crate::config::PersonaSection;

/// Resolved persona with a concrete system prompt.
#[derive(Debug, Clone)]
pub struct Persona {
    /// Display name of the assistant.
    pub name: String,
    /// The resolved system prompt.
    pub system_prompt: String,
}

impl Persona {
    /// Build a persona from configuration.
    ///
    /// Resolution order:
    /// 1. If `system_prompt_file` is set, read from file.
    /// 2. If `system_prompt` is set inline, use it.
    /// 3. Otherwise, generate a default prompt using the persona name.
    pub fn from_config(config: &PersonaSection) -> Result<Self> {
        let system_prompt = if let Some(ref path) = config.system_prompt_file {
            std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read system prompt file: {path}"))?
        } else if let Some(ref prompt) = config.system_prompt {
            prompt.clone()
        } else {
            Self::default_prompt(&config.name)
        };

        Ok(Self {
            name: config.name.clone(),
            system_prompt,
        })
    }

    /// Generate the default system prompt for a given persona name.
    pub fn default_prompt(name: &str) -> String {
        format!(
            "You are {name}, a helpful AI assistant with access to tools for file operations, \
             code search, shell commands, git, web fetching, and validation. Use these tools \
             proactively to help the user accomplish their goals. Be concise and accurate."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PersonaSection;

    #[test]
    fn test_from_config_defaults() {
        let config = PersonaSection::default();
        let persona = Persona::from_config(&config).unwrap();
        assert_eq!(persona.name, "BrainClaw");
        assert!(persona.system_prompt.contains("BrainClaw"));
        assert!(persona.system_prompt.contains("helpful AI assistant"));
    }

    #[test]
    fn test_from_config_custom_prompt() {
        let config = PersonaSection {
            name: "TestBot".to_string(),
            system_prompt: Some("You are a test bot.".to_string()),
            system_prompt_file: None,
            context_files: Vec::new(),
        };
        let persona = Persona::from_config(&config).unwrap();
        assert_eq!(persona.name, "TestBot");
        assert_eq!(persona.system_prompt, "You are a test bot.");
    }

    #[test]
    fn test_from_config_file_based() {
        // Create a temporary file with a system prompt
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("prompt.txt");
        std::fs::write(&file_path, "You are a file-based bot.").unwrap();

        let config = PersonaSection {
            name: "FileBot".to_string(),
            system_prompt: None,
            system_prompt_file: Some(file_path.to_str().unwrap().to_string()),
            context_files: Vec::new(),
        };
        let persona = Persona::from_config(&config).unwrap();
        assert_eq!(persona.name, "FileBot");
        assert_eq!(persona.system_prompt, "You are a file-based bot.");
    }

    #[test]
    fn test_from_config_file_takes_precedence_over_inline() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("prompt.txt");
        std::fs::write(&file_path, "From file.").unwrap();

        let config = PersonaSection {
            name: "Bot".to_string(),
            system_prompt: Some("From inline.".to_string()),
            system_prompt_file: Some(file_path.to_str().unwrap().to_string()),
            context_files: Vec::new(),
        };
        let persona = Persona::from_config(&config).unwrap();
        // File takes precedence
        assert_eq!(persona.system_prompt, "From file.");
    }

    #[test]
    fn test_from_config_missing_file_errors() {
        let config = PersonaSection {
            name: "Bot".to_string(),
            system_prompt: None,
            system_prompt_file: Some("/nonexistent/path/prompt.txt".to_string()),
            context_files: Vec::new(),
        };
        let result = Persona::from_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_prompt() {
        let prompt = Persona::default_prompt("Jarvis");
        assert!(prompt.contains("Jarvis"));
        assert!(prompt.contains("helpful AI assistant"));
        assert!(prompt.contains("tools"));
    }
}
