//! Knowledge (BKS) slash commands — /learn, /knowledge, /knowledge:*

use super::super::super::state::{App, TuiMessage};

impl App {
    /// Handle /learn command - teach a behavioral truth
    pub(super) async fn handle_learn_truth(&mut self, rule: &str, rationale: Option<&str>) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::cache::BehavioralKnowledgeCache;
        use brainwires::knowledge::bks_pks::truth::{BehavioralTruth, TruthCategory, TruthSource};

        // Infer category from the rule text
        let category = if rule.to_lowercase().contains("--") || rule.to_lowercase().contains("flag")
        {
            TruthCategory::CommandUsage
        } else if rule.to_lowercase().contains("instead") || rule.to_lowercase().contains("spawn") {
            TruthCategory::TaskStrategy
        } else if rule.to_lowercase().contains("error") || rule.to_lowercase().contains("fail") {
            TruthCategory::ErrorRecovery
        } else {
            TruthCategory::ToolBehavior
        };

        // Extract context pattern (first few words)
        let context = rule
            .split_whitespace()
            .take(3)
            .collect::<Vec<_>>()
            .join(" ");

        // Create the truth
        let truth = BehavioralTruth::new(
            category,
            context,
            rule.to_string(),
            rationale.unwrap_or("Explicitly taught by user").to_string(),
            TruthSource::ExplicitCommand,
            None, // created_by
        );

        // Try to save to cache
        let save_result = match PlatformPaths::knowledge_db() {
            Ok(db_path) => match BehavioralKnowledgeCache::new(&db_path, 100) {
                Ok(mut cache) => cache.add_truth(truth.clone()).map(|_| true),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };

        let msg = match save_result {
            Ok(_) => format!(
                "📚 Learned new behavioral truth:\n\n\
                **Rule:** {}\n\
                **Category:** {:?}\n\
                **Rationale:** {}\n\
                **Confidence:** {:.0}%\n\n\
                This truth will be shared with all Brainwires users once synced.",
                truth.rule,
                truth.category,
                truth.rationale,
                truth.confidence * 100.0
            ),
            Err(e) => format!("❌ Failed to save truth: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: msg,
            created_at: chrono::Utc::now().timestamp(),
        });

        self.clear_input();
    }

    /// Handle /knowledge command - show status
    pub(super) async fn handle_knowledge_status(&mut self) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::cache::BehavioralKnowledgeCache;

        // Try to load cache to get stats
        let stats_msg = match PlatformPaths::knowledge_db() {
            Ok(db_path) => match BehavioralKnowledgeCache::new(&db_path, 100) {
                Ok(cache) => {
                    let stats = cache.stats();
                    format!(
                        "📊 Behavioral Knowledge System Status\n\n\
                            Total truths: {}\n\
                            Average confidence: {:.0}%\n\
                            Pending submissions: {}\n\
                            Last sync: {}\n\n\
                            Commands:\n\
                            • /learn <rule>        - Teach a new truth\n\
                            • /knowledge:list      - List all truths\n\
                            • /knowledge:search    - Search truths\n\
                            • /knowledge:sync      - Force sync with server\n\
                            • /knowledge:contradict <id> - Report incorrect truth",
                        stats.total_truths,
                        stats.avg_confidence * 100.0,
                        stats.pending_submissions,
                        if stats.last_sync > 0 {
                            chrono::DateTime::from_timestamp(stats.last_sync, 0)
                                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                                .unwrap_or_else(|| "Unknown".to_string())
                        } else {
                            "Never".to_string()
                        }
                    )
                }
                Err(e) => format!("Failed to load knowledge cache: {}", e),
            },
            Err(e) => format!("Failed to get knowledge database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: stats_msg,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /knowledge:list command
    pub(super) async fn handle_knowledge_list(&mut self, category: Option<&str>) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::cache::BehavioralKnowledgeCache;
        use brainwires::knowledge::bks_pks::truth::TruthCategory;

        let result = match PlatformPaths::knowledge_db() {
            Ok(db_path) => match BehavioralKnowledgeCache::new(&db_path, 100) {
                Ok(cache) => {
                    let truths: Vec<_> = if let Some(cat_str) = category {
                        let cat = match cat_str.to_lowercase().as_str() {
                            "command" => TruthCategory::CommandUsage,
                            "strategy" => TruthCategory::TaskStrategy,
                            "tool" => TruthCategory::ToolBehavior,
                            "error" => TruthCategory::ErrorRecovery,
                            "resource" => TruthCategory::ResourceManagement,
                            "pattern" => TruthCategory::PatternAvoidance,
                            _ => {
                                self.add_console_message(format!(
                                    "❌ Invalid category: {}",
                                    cat_str
                                ));
                                return;
                            }
                        };
                        cache.truths_by_category(cat).into_iter().cloned().collect()
                    } else {
                        cache.all_truths().cloned().collect()
                    };

                    if truths.is_empty() {
                        "No learned truths found.".to_string()
                    } else {
                        let mut output = format!("📚 Learned Truths ({} total)\n\n", truths.len());
                        for (i, truth) in truths.iter().take(20).enumerate() {
                            output.push_str(&format!(
                                "{}. **{}** ({:?})\n   {}\n   Confidence: {:.0}% | Uses: {}\n\n",
                                i + 1,
                                &truth.id[..8.min(truth.id.len())],
                                truth.category,
                                truth.rule,
                                truth.confidence * 100.0,
                                truth.reinforcements
                            ));
                        }
                        if truths.len() > 20 {
                            output.push_str(&format!("...and {} more", truths.len() - 20));
                        }
                        output
                    }
                }
                Err(e) => format!("Failed to load knowledge cache: {}", e),
            },
            Err(e) => format!("Failed to get knowledge database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: result,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /knowledge:search command
    pub(super) async fn handle_knowledge_search(&mut self, query: &str) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::cache::BehavioralKnowledgeCache;
        use brainwires::knowledge::bks_pks::matcher::ContextMatcher;

        let result = match PlatformPaths::knowledge_db() {
            Ok(db_path) => match BehavioralKnowledgeCache::new(&db_path, 100) {
                Ok(cache) => {
                    let matcher = ContextMatcher::new(0.0, 30, 10);
                    let truths: Vec<_> = cache.all_truths().cloned().collect();
                    let matches = matcher.search(query, truths.iter());

                    if matches.is_empty() {
                        format!("No truths found matching \"{}\"", query)
                    } else {
                        let mut output = format!("🔍 Search Results for \"{}\"\n\n", query);
                        for (i, m) in matches.iter().enumerate() {
                            output.push_str(&format!(
                                    "{}. **{}** ({:?})\n   {}\n   Confidence: {:.0}% | Score: {:.0}%\n\n",
                                    i + 1,
                                    &m.truth.id[..8.min(m.truth.id.len())],
                                    m.truth.category,
                                    m.truth.rule,
                                    m.effective_confidence * 100.0,
                                    m.match_score * 100.0
                                ));
                        }
                        output
                    }
                }
                Err(e) => format!("Failed to load knowledge cache: {}", e),
            },
            Err(e) => format!("Failed to get knowledge database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: result,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /knowledge:sync command
    pub(super) async fn handle_knowledge_sync(&mut self) {
        // For now, just show a message - actual sync would require
        // the HTTP client and backend URL from config
        self.add_console_message("🔄 Syncing with Brainwires server...".to_string());

        // TODO: Implement actual sync when backend endpoints are ready
        self.add_console_message(
            "ℹ️  Server sync not yet implemented - truths stored locally".to_string(),
        );

        self.clear_input();
    }

    /// Handle /knowledge:contradict command
    pub(super) async fn handle_knowledge_contradict(&mut self, id: &str, reason: Option<&str>) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::cache::BehavioralKnowledgeCache;

        let result = match PlatformPaths::knowledge_db() {
            Ok(db_path) => {
                match BehavioralKnowledgeCache::new(&db_path, 100) {
                    Ok(mut cache) => {
                        // Get and update the truth
                        if let Some(truth) = cache.get_truth_mut(id) {
                            truth.contradict(0.1); // Default EMA alpha
                            let reason_str = reason
                                .map(|r| format!("\nReason: {}", r))
                                .unwrap_or_default();
                            format!(
                                "✅ Contradicted truth: {}{}\nNew confidence: {:.0}%",
                                id,
                                reason_str,
                                truth.confidence * 100.0
                            )
                        } else {
                            format!("❌ Truth not found: {}", id)
                        }
                    }
                    Err(e) => format!("Failed to load knowledge cache: {}", e),
                }
            }
            Err(e) => format!("Failed to get knowledge database path: {}", e),
        };

        self.add_console_message(result);
        self.clear_input();
    }

    /// Handle /knowledge:delete command
    pub(super) async fn handle_knowledge_delete(&mut self, id: &str) {
        use crate::utils::paths::PlatformPaths;
        use brainwires::knowledge::bks_pks::cache::BehavioralKnowledgeCache;

        let result = match PlatformPaths::knowledge_db() {
            Ok(db_path) => match BehavioralKnowledgeCache::new(&db_path, 100) {
                Ok(mut cache) => match cache.remove_truth(id) {
                    Ok(true) => format!("✅ Deleted truth: {}", id),
                    Ok(false) => format!("❌ Truth not found: {}", id),
                    Err(e) => format!("❌ Failed to delete truth: {}", e),
                },
                Err(e) => format!("Failed to load knowledge cache: {}", e),
            },
            Err(e) => format!("Failed to get knowledge database path: {}", e),
        };

        self.add_console_message(result);
        self.clear_input();
    }
}
