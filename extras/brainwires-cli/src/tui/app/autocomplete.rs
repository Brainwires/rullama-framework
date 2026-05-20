//! Autocomplete for Slash Commands and Arguments
//!
//! Provides autocomplete suggestions and navigation for:
//! - Slash commands (e.g., /help, /model, /clear)
//! - Command arguments (e.g., model names for /model)

use super::state::App;

pub(super) trait AutocompleteOps {
    fn update_autocomplete(&mut self);
    fn autocomplete_next(&mut self);
    fn autocomplete_prev(&mut self);
    fn autocomplete_accept(&mut self, add_space: bool);
    fn refresh_model_cache(&mut self);
}

impl AutocompleteOps for App {
    /// Update autocomplete suggestions based on current input
    fn update_autocomplete(&mut self) {
        let input = self.input_text();
        // Only show autocomplete if input starts with '/'
        if !input.starts_with('/') {
            self.show_autocomplete = false;
            self.autocomplete_suggestions.clear();
            self.autocomplete_index = 0;
            self.autocomplete_title = "Slash Commands".to_string();
            return;
        }

        // Check if we're completing a /model argument
        let input_lower = input.to_lowercase();
        if input_lower.starts_with("/model ") {
            // Extract the partial model name (everything after "/model ")
            let partial_model = input[7..].to_string(); // len("/model ") = 7

            // Refresh model cache if empty
            if self.cached_model_ids.is_empty() {
                self.refresh_model_cache();
            }

            // Filter models that contain the partial input (case-insensitive)
            let partial_lower = partial_model.to_lowercase();
            let mut matching_models: Vec<String> = self
                .cached_model_ids
                .iter()
                .filter(|model| model.to_lowercase().contains(&partial_lower))
                .cloned()
                .collect();

            // Sort alphabetically for consistent display
            matching_models.sort();

            // Update autocomplete state for model completion
            if matching_models.is_empty() {
                self.show_autocomplete = false;
                self.autocomplete_suggestions.clear();
                self.autocomplete_index = 0;
            } else {
                self.show_autocomplete = true;
                self.autocomplete_suggestions = matching_models;
                self.autocomplete_title = "Models".to_string();
                // Reset index if it's out of bounds
                if self.autocomplete_index >= self.autocomplete_suggestions.len() {
                    self.autocomplete_index = 0;
                }
            }
            return;
        }

        // Standard command completion
        // Extract the partial command (everything after '/')
        let partial_cmd = &input[1..];

        // Get all available commands from the command executor
        let all_commands: Vec<String> = self
            .command_executor
            .registry()
            .commands()
            .keys()
            .map(|s| s.to_string())
            .collect();

        // Filter commands that start with the partial input
        let mut matching_commands: Vec<String> = all_commands
            .iter()
            .filter(|cmd| cmd.starts_with(partial_cmd))
            .cloned()
            .collect();

        // Also surface discovered skills — typing `/<skill-name>` should
        // autocomplete to the full name the same way built-in commands do.
        if let Some(ref registry) = self.skill_registry {
            for name in registry.list_skills() {
                if name.starts_with(partial_cmd) && !matching_commands.iter().any(|c| c == name) {
                    matching_commands.push(name.to_string());
                }
            }
        }

        // Sort alphabetically for consistent display
        matching_commands.sort();

        // Update autocomplete state
        if matching_commands.is_empty() {
            self.show_autocomplete = false;
            self.autocomplete_suggestions.clear();
            self.autocomplete_index = 0;
        } else {
            self.show_autocomplete = true;
            self.autocomplete_suggestions = matching_commands;
            self.autocomplete_title = "Slash Commands".to_string();
            // Reset index if it's out of bounds
            if self.autocomplete_index >= self.autocomplete_suggestions.len() {
                self.autocomplete_index = 0;
            }
        }
    }

    /// Select next autocomplete suggestion
    fn autocomplete_next(&mut self) {
        if !self.autocomplete_suggestions.is_empty() {
            self.autocomplete_index =
                (self.autocomplete_index + 1) % self.autocomplete_suggestions.len();
        }
    }

    /// Select previous autocomplete suggestion
    fn autocomplete_prev(&mut self) {
        if !self.autocomplete_suggestions.is_empty() {
            if self.autocomplete_index == 0 {
                self.autocomplete_index = self.autocomplete_suggestions.len() - 1;
            } else {
                self.autocomplete_index -= 1;
            }
        }
    }

    /// Accept current autocomplete suggestion
    /// Behavior depends on context:
    /// - For commands: adds a space after if `add_space` is true (Tab behavior)
    /// - For model arguments: completes the full command
    /// - For /model command: always adds space to trigger model autocomplete
    fn autocomplete_accept(&mut self, add_space: bool) {
        if self.show_autocomplete && !self.autocomplete_suggestions.is_empty() {
            let selected = self.autocomplete_suggestions[self.autocomplete_index].clone();

            // Check if we're completing a model argument
            if self.autocomplete_title == "Models" {
                // Complete the /model command with the selected model
                self.input_state.set_text(format!("/model {}", selected));
                self.show_autocomplete = false;
                self.autocomplete_suggestions.clear();
                self.autocomplete_index = 0;
                self.autocomplete_title = "Slash Commands".to_string();
            } else {
                // Standard command completion
                // For /model command, always add space to trigger model autocomplete
                let needs_space = add_space || selected == "model";
                let new_input = if needs_space {
                    format!("/{} ", selected)
                } else {
                    format!("/{}", selected)
                };
                self.input_state.set_text(new_input);
                self.show_autocomplete = false;
                self.autocomplete_suggestions.clear();
                self.autocomplete_index = 0;
                self.autocomplete_title = "Slash Commands".to_string();

                // If we just completed /model, trigger autocomplete again to show models
                if selected == "model" {
                    self.update_autocomplete();
                }
            }
        }
    }

    /// Refresh the cached model IDs from the model cache file
    fn refresh_model_cache(&mut self) {
        // Try to load from cache file synchronously
        // The cache is populated by previous API calls
        if let Some(cache) = Self::load_model_cache() {
            self.cached_model_ids = cache;
        } else {
            // Provide some fallback models if cache is empty
            self.cached_model_ids = vec![
                "claude-haiku-4-5-20251001".to_string(),
                "claude-sonnet-4-5-20250929".to_string(),
                "claude-opus-4-1-20250805".to_string(),
                "gemini-2.5-pro".to_string(),
                "llama-3.3-70b-versatile".to_string(),
            ];
        }
    }
}

impl App {
    /// Load model IDs from the cache file
    /// Only includes models with "chat" ability and either "tools" or "tool use" ability
    fn load_model_cache() -> Option<Vec<String>> {
        use crate::utils::paths::PlatformPaths;
        use std::fs;

        let cache_path = PlatformPaths::brainwires_data_dir()
            .ok()?
            .join("models_cache.json");
        if !cache_path.exists() {
            return None;
        }

        let content = fs::read_to_string(&cache_path).ok()?;

        // Parse the cache JSON to extract model IDs with ability filtering
        #[derive(serde::Deserialize)]
        struct ModelCache {
            models: Vec<CachedModel>,
        }

        #[derive(serde::Deserialize)]
        struct CachedModel {
            model_id: String,
            abilities: String,
        }

        let cache: ModelCache = serde_json::from_str(&content).ok()?;

        // Filter models: must have "chat" ability AND ("tools" OR "tool use") ability
        let filtered_models: Vec<String> = cache
            .models
            .into_iter()
            .filter(|m| {
                let abilities_lower = m.abilities.to_lowercase();
                let has_chat = abilities_lower.contains("chat");
                let has_tools =
                    abilities_lower.contains("tools") || abilities_lower.contains("tool use");
                has_chat && has_tools
            })
            .map(|m| m.model_id)
            .collect();

        if filtered_models.is_empty() {
            None
        } else {
            Some(filtered_models)
        }
    }
}
