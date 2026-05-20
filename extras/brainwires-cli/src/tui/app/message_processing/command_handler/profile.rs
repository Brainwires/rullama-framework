//! Personal knowledge (PKS) slash commands — /profile, /profile:*

use super::super::super::state::{App, TuiMessage};

impl App {
    /// Handle /profile command - show profile summary
    pub(super) async fn handle_profile_show(&mut self) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::personal::{
            PersonalFactMatcher, PersonalKnowledgeCache,
        };

        let result = match PlatformPaths::personal_knowledge_db() {
            Ok(db_path) => match PersonalKnowledgeCache::new(&db_path, 100) {
                Ok(cache) => {
                    let facts: Vec<_> = cache.all_facts().cloned().collect();
                    if facts.is_empty() {
                        "👤 Your Profile\n\n\
                            No personal facts learned yet.\n\n\
                            Use these commands to build your profile:\n\
                            • /profile:set <key> <value> - Set a fact\n\
                            • /profile:name <your_name>  - Set your name\n\n\
                            The system also learns from conversation patterns like:\n\
                            • \"My name is...\"\n\
                            • \"I prefer...\"\n\
                            • \"I'm working on...\""
                            .to_string()
                    } else {
                        let matcher = PersonalFactMatcher::new(0.0, 30, true);
                        let fact_refs: Vec<_> = facts.iter().collect();
                        matcher.format_profile_summary(&fact_refs)
                    }
                }
                Err(e) => format!("Failed to load profile: {}", e),
            },
            Err(e) => format!("Failed to get profile database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: result,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /profile:set command
    pub(super) async fn handle_profile_set(&mut self, key: &str, value: &str, local_only: bool) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::personal::{
            PersonalFact, PersonalFactCategory, PersonalFactSource, PersonalKnowledgeCache,
        };

        // Infer category from key name
        let category = match key.to_lowercase().as_str() {
            "name" | "role" | "team" | "organization" | "company" => PersonalFactCategory::Identity,
            "timezone" | "limitation" | "restriction" => PersonalFactCategory::Constraint,
            "skill" | "expert" | "proficient" | "knows" => PersonalFactCategory::Capability,
            "project" | "working_on" | "current_task" => PersonalFactCategory::Context,
            _ => PersonalFactCategory::Preference,
        };

        let fact = PersonalFact::new(
            category,
            key.to_string(),
            value.to_string(),
            None,
            PersonalFactSource::ExplicitStatement,
            local_only,
        );

        let result = match PlatformPaths::personal_knowledge_db() {
            Ok(db_path) => match PersonalKnowledgeCache::new(&db_path, 100) {
                Ok(mut cache) => match cache.upsert_fact(fact.clone()) {
                    Ok(_) => {
                        let local_str = if local_only { " (local only)" } else { "" };
                        format!(
                            "✅ Set profile fact{}\n\n\
                                    **{}** = {}\n\
                                    Category: {:?}",
                            local_str, key, value, category
                        )
                    }
                    Err(e) => format!("❌ Failed to save fact: {}", e),
                },
                Err(e) => format!("Failed to load profile: {}", e),
            },
            Err(e) => format!("Failed to get profile database path: {}", e),
        };

        self.add_console_message(result);
        self.clear_input();
    }

    /// Handle /profile:list command
    pub(super) async fn handle_profile_list(&mut self, category: Option<&str>) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::personal::{
            PersonalFactCategory, PersonalKnowledgeCache,
        };

        let result = match PlatformPaths::personal_knowledge_db() {
            Ok(db_path) => match PersonalKnowledgeCache::new(&db_path, 100) {
                Ok(cache) => {
                    let facts: Vec<_> = if let Some(cat_str) = category {
                        let cat = match cat_str.to_lowercase().as_str() {
                            "identity" => PersonalFactCategory::Identity,
                            "preference" => PersonalFactCategory::Preference,
                            "capability" => PersonalFactCategory::Capability,
                            "context" => PersonalFactCategory::Context,
                            "constraint" => PersonalFactCategory::Constraint,
                            "relationship" => PersonalFactCategory::Relationship,
                            _ => {
                                self.add_console_message(format!(
                                    "❌ Invalid category: {}",
                                    cat_str
                                ));
                                return;
                            }
                        };
                        cache.facts_by_category(cat).into_iter().cloned().collect()
                    } else {
                        cache.all_facts().cloned().collect()
                    };

                    if facts.is_empty() {
                        "No personal facts found.".to_string()
                    } else {
                        let mut output = format!("👤 Personal Facts ({} total)\n\n", facts.len());
                        for (i, fact) in facts.iter().take(30).enumerate() {
                            let local_marker = if fact.local_only { " 🔒" } else { "" };
                            output.push_str(&format!(
                                "{}. **{}**{} ({:?})\n   {}\n   Confidence: {:.0}%\n\n",
                                i + 1,
                                fact.key,
                                local_marker,
                                fact.category,
                                fact.value,
                                fact.confidence * 100.0
                            ));
                        }
                        if facts.len() > 30 {
                            output.push_str(&format!("...and {} more", facts.len() - 30));
                        }
                        output
                    }
                }
                Err(e) => format!("Failed to load profile: {}", e),
            },
            Err(e) => format!("Failed to get profile database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: result,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /profile:search command
    pub(super) async fn handle_profile_search(&mut self, query: &str) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::personal::PersonalKnowledgeCache;

        let result = match PlatformPaths::personal_knowledge_db() {
            Ok(db_path) => match PersonalKnowledgeCache::new(&db_path, 100) {
                Ok(cache) => {
                    let matches = cache.search_facts(query);

                    if matches.is_empty() {
                        format!("No facts found matching \"{}\"", query)
                    } else {
                        let mut output = format!("🔍 Search Results for \"{}\"\n\n", query);
                        for (i, fact) in matches.iter().enumerate() {
                            let local_marker = if fact.local_only { " 🔒" } else { "" };
                            output.push_str(&format!(
                                "{}. **{}**{} ({:?})\n   {}\n   Confidence: {:.0}%\n\n",
                                i + 1,
                                fact.key,
                                local_marker,
                                fact.category,
                                fact.value,
                                fact.confidence * 100.0
                            ));
                        }
                        output
                    }
                }
                Err(e) => format!("Failed to load profile: {}", e),
            },
            Err(e) => format!("Failed to get profile database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: result,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /profile:delete command
    pub(super) async fn handle_profile_delete(&mut self, id_or_key: &str) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::personal::PersonalKnowledgeCache;

        let result = match PlatformPaths::personal_knowledge_db() {
            Ok(db_path) => {
                match PersonalKnowledgeCache::new(&db_path, 100) {
                    Ok(mut cache) => {
                        // Try to delete by ID first, then by key
                        match cache.remove_fact(id_or_key) {
                            Ok(true) => format!("✅ Deleted fact: {}", id_or_key),
                            Ok(false) => {
                                // Try by key
                                match cache.remove_fact_by_key(id_or_key) {
                                    Ok(true) => format!("✅ Deleted fact with key: {}", id_or_key),
                                    _ => format!("❌ Fact not found: {}", id_or_key),
                                }
                            }
                            Err(e) => format!("❌ Failed to delete fact: {}", e),
                        }
                    }
                    Err(e) => format!("Failed to load profile: {}", e),
                }
            }
            Err(e) => format!("Failed to get profile database path: {}", e),
        };

        self.add_console_message(result);
        self.clear_input();
    }

    /// Handle /profile:sync command
    pub(super) async fn handle_profile_sync(&mut self) {
        // For now, just show a message - actual sync would require
        // the HTTP client and backend URL from config
        self.add_console_message("🔄 Syncing personal profile with server...".to_string());

        // TODO: Implement actual sync with PersonalKnowledgeApiClient
        self.add_console_message(
            "ℹ️  Server sync not yet implemented - facts stored locally".to_string(),
        );

        self.clear_input();
    }

    /// Handle /profile:export command
    pub(super) async fn handle_profile_export(&mut self, path: Option<&str>) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::personal::PersonalKnowledgeCache;

        let export_path = path.map(std::path::PathBuf::from).unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("brainwires-profile.json")
        });

        let result = match PlatformPaths::personal_knowledge_db() {
            Ok(db_path) => {
                match PersonalKnowledgeCache::new(&db_path, 100) {
                    Ok(cache) => {
                        match cache.export_json() {
                            Ok(json) => {
                                // Write to file
                                match std::fs::write(&export_path, &json) {
                                    Ok(_) => {
                                        let count = json.matches("\"id\"").count();
                                        format!(
                                            "✅ Exported {} facts to:\n{}",
                                            count,
                                            export_path.display()
                                        )
                                    }
                                    Err(e) => format!("❌ Failed to write file: {}", e),
                                }
                            }
                            Err(e) => format!("❌ Export failed: {}", e),
                        }
                    }
                    Err(e) => format!("Failed to load profile: {}", e),
                }
            }
            Err(e) => format!("Failed to get profile database path: {}", e),
        };

        self.add_console_message(result);
        self.clear_input();
    }

    /// Handle /profile:import command
    pub(super) async fn handle_profile_import(&mut self, path: &str) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::personal::PersonalKnowledgeCache;

        let import_path = std::path::PathBuf::from(path);

        let result = match PlatformPaths::personal_knowledge_db() {
            Ok(db_path) => {
                match PersonalKnowledgeCache::new(&db_path, 100) {
                    Ok(mut cache) => {
                        // Read file first
                        match std::fs::read_to_string(&import_path) {
                            Ok(json) => match cache.import_json(&json) {
                                Ok(result) => format!(
                                    "✅ Imported {} new facts, updated {} existing facts from:\n{}",
                                    result.imported,
                                    result.updated,
                                    import_path.display()
                                ),
                                Err(e) => format!("❌ Import failed: {}", e),
                            },
                            Err(e) => format!("❌ Failed to read file: {}", e),
                        }
                    }
                    Err(e) => format!("Failed to load profile: {}", e),
                }
            }
            Err(e) => format!("Failed to get profile database path: {}", e),
        };

        self.add_console_message(result);
        self.clear_input();
    }

    /// Handle /profile:stats command
    pub(super) async fn handle_profile_stats(&mut self) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::personal::{
            PersonalFactCategory, PersonalKnowledgeCache,
        };

        let result = match PlatformPaths::personal_knowledge_db() {
            Ok(db_path) => match PersonalKnowledgeCache::new(&db_path, 100) {
                Ok(cache) => {
                    let stats = cache.stats();
                    format!(
                        "📊 Personal Knowledge Statistics\n\n\
                            Total facts: {}\n\
                            Local-only facts: {}\n\
                            Average confidence: {:.0}%\n\
                            Pending submissions: {}\n\
                            Pending feedback: {}\n\
                            Last sync: {}\n\n\
                            By Category:\n\
                            • Identity: {}\n\
                            • Preference: {}\n\
                            • Capability: {}\n\
                            • Context: {}\n\
                            • Constraint: {}\n\
                            • Relationship: {}",
                        stats.total_facts,
                        stats.local_only_facts,
                        stats.avg_confidence * 100.0,
                        stats.pending_submissions,
                        stats.pending_feedback,
                        if stats.last_sync > 0 {
                            chrono::DateTime::from_timestamp(stats.last_sync, 0)
                                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                                .unwrap_or_else(|| "Unknown".to_string())
                        } else {
                            "Never".to_string()
                        },
                        stats
                            .by_category
                            .get(&PersonalFactCategory::Identity)
                            .unwrap_or(&0),
                        stats
                            .by_category
                            .get(&PersonalFactCategory::Preference)
                            .unwrap_or(&0),
                        stats
                            .by_category
                            .get(&PersonalFactCategory::Capability)
                            .unwrap_or(&0),
                        stats
                            .by_category
                            .get(&PersonalFactCategory::Context)
                            .unwrap_or(&0),
                        stats
                            .by_category
                            .get(&PersonalFactCategory::Constraint)
                            .unwrap_or(&0),
                        stats
                            .by_category
                            .get(&PersonalFactCategory::Relationship)
                            .unwrap_or(&0),
                    )
                }
                Err(e) => format!("Failed to load profile: {}", e),
            },
            Err(e) => format!("Failed to get profile database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: result,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }
}
