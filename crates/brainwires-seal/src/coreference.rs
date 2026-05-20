//! Coreference Resolution for Multi-Turn Conversations
//!
//! Resolves anaphoric references (pronouns, definite NPs, ellipsis) to concrete
//! entities from the conversation history. This is critical for understanding
//! queries like "fix it", "update the file", or "what does that function do?"
//!
//! ## Approach
//!
//! Uses a salience-based ranking algorithm that considers:
//! - **Recency**: More recently mentioned entities score higher
//! - **Frequency**: Entities mentioned multiple times score higher
//! - **Graph centrality**: Important entities in the relationship graph score higher
//! - **Type matching**: Entity type compatibility with the reference
//! - **Syntactic prominence**: Subjects score higher than objects
//!
//! ## Example
//!
//! ```rust,ignore
//! let resolver = CoreferenceResolver::new();
//! let dialog_state = DialogState::new();
//!
//! // After discussing "main.rs"
//! dialog_state.mention_entity("main.rs", EntityType::File);
//!
//! let refs = resolver.detect_references("Fix it and run the tests");
//! // refs[0] = UnresolvedReference { text: "it", ref_type: SingularNeutral }
//!
//! let resolved = resolver.resolve(&refs, &dialog_state, &entity_store, None);
//! // resolved[0].antecedent = "main.rs"
//! ```

use brainwires_core::graph::{EntityStoreT, EntityType, RelationshipGraphT};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// --- LazyLock regex statics for coreference pattern detection ---

// Pronoun patterns
static RE_SINGULAR_NEUTRAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(it|this|that)\b").expect("valid regex"));
static RE_PLURAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(they|them|those|these)\b").expect("valid regex"));

// Definite NP patterns
static RE_THE_FILE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bthe\s+(file|files)\b").expect("valid regex"));
static RE_THE_FUNCTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bthe\s+(function|method|fn)\b").expect("valid regex"));
static RE_THE_TYPE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\bthe\s+(type|struct|class|enum|interface)\b").expect("valid regex")
});
static RE_THE_ERROR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bthe\s+(error|bug|issue)\b").expect("valid regex"));
static RE_THE_VARIABLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bthe\s+(variable|var|const|let)\b").expect("valid regex"));
static RE_THE_COMMAND: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bthe\s+(command|cmd)\b").expect("valid regex"));

// Demonstrative patterns
static RE_DEMO_FILE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(that|this)\s+(file)\b").expect("valid regex"));
static RE_DEMO_FUNCTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(that|this)\s+(function|method|fn)\b").expect("valid regex"));
static RE_DEMO_TYPE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(that|this)\s+(type|struct|class|enum)\b").expect("valid regex")
});
static RE_DEMO_ERROR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(that|this)\s+(error|bug|issue)\b").expect("valid regex"));

/// Types of anaphoric references we can detect
#[derive(Debug, Clone, PartialEq)]
pub enum ReferenceType {
    /// Singular neutral pronouns: "it", "this", "that"
    SingularNeutral,
    /// Plural pronouns: "they", "them", "those", "these"
    Plural,
    /// Definite noun phrase with entity type: "the file", "the function"
    DefiniteNP {
        /// The entity type referenced by the noun phrase.
        entity_type: EntityType,
    },
    /// Demonstrative with entity type: "that error", "this type"
    Demonstrative {
        /// The entity type referenced by the demonstrative.
        entity_type: EntityType,
    },
    /// Missing subject from context (implied reference)
    Ellipsis,
}

impl ReferenceType {
    /// Get compatible entity types for this reference type
    pub fn compatible_types(&self) -> Vec<EntityType> {
        match self {
            ReferenceType::SingularNeutral => vec![
                EntityType::File,
                EntityType::Function,
                EntityType::Type,
                EntityType::Variable,
                EntityType::Error,
                EntityType::Concept,
                EntityType::Command,
            ],
            ReferenceType::Plural => vec![
                EntityType::File,
                EntityType::Function,
                EntityType::Type,
                EntityType::Variable,
                EntityType::Error,
            ],
            ReferenceType::DefiniteNP { entity_type } => vec![entity_type.clone()],
            ReferenceType::Demonstrative { entity_type } => vec![entity_type.clone()],
            ReferenceType::Ellipsis => vec![
                EntityType::File,
                EntityType::Function,
                EntityType::Type,
                EntityType::Command,
            ],
        }
    }
}

/// An unresolved reference detected in user input
#[derive(Debug, Clone)]
pub struct UnresolvedReference {
    /// The text of the reference (e.g., "it", "the file")
    pub text: String,
    /// Type of reference
    pub ref_type: ReferenceType,
    /// Character offset in the original message
    pub start: usize,
    /// Character offset end
    pub end: usize,
}

/// A resolved reference with its antecedent
#[derive(Debug, Clone)]
pub struct ResolvedReference {
    /// The original reference
    pub reference: UnresolvedReference,
    /// The resolved entity name
    pub antecedent: String,
    /// Entity type of the antecedent
    pub entity_type: EntityType,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Salience breakdown for debugging
    pub salience: SalienceScore,
}

/// Salience factors for ranking antecedent candidates
#[derive(Debug, Clone, Default)]
pub struct SalienceScore {
    /// How recently mentioned (0.0 - 1.0), weight: 0.35
    pub recency: f32,
    /// How often mentioned (0.0 - 1.0), weight: 0.15
    pub frequency: f32,
    /// Importance in relationship graph (0.0 - 1.0), weight: 0.20
    pub graph_centrality: f32,
    /// Type compatibility (0.0 or 1.0), weight: 0.20
    pub type_match: f32,
    /// Subject position bonus (0.0 - 1.0), weight: 0.10
    pub syntactic_prominence: f32,
}

impl SalienceScore {
    /// Compute weighted total score
    pub fn total(&self) -> f32 {
        self.recency * 0.35
            + self.frequency * 0.15
            + self.graph_centrality * 0.20
            + self.type_match * 0.20
            + self.syntactic_prominence * 0.10
    }
}

/// Dialog state for tracking entities across conversation turns
#[derive(Debug, Clone, Default)]
pub struct DialogState {
    /// Stack of entities in current focus (most recent first)
    pub focus_stack: Vec<String>,
    /// Entity name -> turn number when mentioned
    pub mention_history: HashMap<String, Vec<u32>>,
    /// Current turn number
    pub current_turn: u32,
    /// Entities that were recently modified/acted upon
    pub recently_modified: Vec<String>,
    /// Entity type cache
    entity_types: HashMap<String, EntityType>,
}

impl DialogState {
    /// Create a new dialog state
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance to the next turn
    pub fn next_turn(&mut self) {
        self.current_turn += 1;
    }

    /// Record a mention of an entity
    pub fn mention_entity(&mut self, name: &str, entity_type: EntityType) {
        // Add to focus stack (remove if already present, add to front)
        self.focus_stack.retain(|n| n != name);
        self.focus_stack.insert(0, name.to_string());

        // Limit focus stack size
        if self.focus_stack.len() > 20 {
            self.focus_stack.truncate(20);
        }

        // Record mention with turn number
        self.mention_history
            .entry(name.to_string())
            .or_default()
            .push(self.current_turn);

        // Cache entity type
        self.entity_types.insert(name.to_string(), entity_type);
    }

    /// Mark an entity as recently modified
    pub fn mark_modified(&mut self, name: &str) {
        self.recently_modified.retain(|n| n != name);
        self.recently_modified.insert(0, name.to_string());

        // Limit modified list size
        if self.recently_modified.len() > 10 {
            self.recently_modified.truncate(10);
        }
    }

    /// Get the entity type for a name (if known)
    pub fn get_entity_type(&self, name: &str) -> Option<&EntityType> {
        self.entity_types.get(name)
    }

    /// Get recency score for an entity (1.0 for most recent, decays with age)
    pub fn recency_score(&self, name: &str) -> f32 {
        // Check focus stack position
        if let Some(pos) = self.focus_stack.iter().position(|n| n == name) {
            let focus_score = 1.0 - (pos as f32 / self.focus_stack.len() as f32);

            // Bonus for recently modified
            let modified_bonus = if self.recently_modified.contains(&name.to_string()) {
                0.2
            } else {
                0.0
            };

            (focus_score + modified_bonus).min(1.0)
        } else {
            // Check mention history
            if let Some(turns) = self.mention_history.get(name) {
                if let Some(&last_turn) = turns.last() {
                    let age = self.current_turn.saturating_sub(last_turn) as f32;
                    (-0.1 * age).exp() // Exponential decay
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
    }

    /// Get frequency score for an entity
    pub fn frequency_score(&self, name: &str) -> f32 {
        if let Some(turns) = self.mention_history.get(name) {
            let count = turns.len() as f32;
            // Logarithmic scaling to prevent domination by very frequent entities
            (count.ln_1p() / 3.0).min(1.0)
        } else {
            0.0
        }
    }

    /// Clear the state for a new conversation
    pub fn clear(&mut self) {
        self.focus_stack.clear();
        self.mention_history.clear();
        self.current_turn = 0;
        self.recently_modified.clear();
        self.entity_types.clear();
    }
}

/// Pattern definition for detecting references
struct ReferencePattern {
    regex: &'static Regex,
    ref_type_fn: fn(&regex::Captures) -> ReferenceType,
}

/// Coreference resolver for multi-turn conversations
pub struct CoreferenceResolver {
    /// Patterns for detecting pronoun references
    pronoun_patterns: Vec<ReferencePattern>,
    /// Patterns for detecting definite NP references
    definite_np_patterns: Vec<ReferencePattern>,
    /// Patterns for detecting demonstrative references
    demonstrative_patterns: Vec<ReferencePattern>,
}

impl CoreferenceResolver {
    /// Create a new coreference resolver
    pub fn new() -> Self {
        Self {
            pronoun_patterns: Self::build_pronoun_patterns(),
            definite_np_patterns: Self::build_definite_np_patterns(),
            demonstrative_patterns: Self::build_demonstrative_patterns(),
        }
    }

    fn build_pronoun_patterns() -> Vec<ReferencePattern> {
        vec![
            ReferencePattern {
                regex: &RE_SINGULAR_NEUTRAL,
                ref_type_fn: |_| ReferenceType::SingularNeutral,
            },
            ReferencePattern {
                regex: &RE_PLURAL,
                ref_type_fn: |_| ReferenceType::Plural,
            },
        ]
    }

    fn build_definite_np_patterns() -> Vec<ReferencePattern> {
        vec![
            ReferencePattern {
                regex: &RE_THE_FILE,
                ref_type_fn: |_| ReferenceType::DefiniteNP {
                    entity_type: EntityType::File,
                },
            },
            ReferencePattern {
                regex: &RE_THE_FUNCTION,
                ref_type_fn: |_| ReferenceType::DefiniteNP {
                    entity_type: EntityType::Function,
                },
            },
            ReferencePattern {
                regex: &RE_THE_TYPE,
                ref_type_fn: |_| ReferenceType::DefiniteNP {
                    entity_type: EntityType::Type,
                },
            },
            ReferencePattern {
                regex: &RE_THE_ERROR,
                ref_type_fn: |_| ReferenceType::DefiniteNP {
                    entity_type: EntityType::Error,
                },
            },
            ReferencePattern {
                regex: &RE_THE_VARIABLE,
                ref_type_fn: |_| ReferenceType::DefiniteNP {
                    entity_type: EntityType::Variable,
                },
            },
            ReferencePattern {
                regex: &RE_THE_COMMAND,
                ref_type_fn: |_| ReferenceType::DefiniteNP {
                    entity_type: EntityType::Command,
                },
            },
        ]
    }

    fn build_demonstrative_patterns() -> Vec<ReferencePattern> {
        vec![
            ReferencePattern {
                regex: &RE_DEMO_FILE,
                ref_type_fn: |_| ReferenceType::Demonstrative {
                    entity_type: EntityType::File,
                },
            },
            ReferencePattern {
                regex: &RE_DEMO_FUNCTION,
                ref_type_fn: |_| ReferenceType::Demonstrative {
                    entity_type: EntityType::Function,
                },
            },
            ReferencePattern {
                regex: &RE_DEMO_TYPE,
                ref_type_fn: |_| ReferenceType::Demonstrative {
                    entity_type: EntityType::Type,
                },
            },
            ReferencePattern {
                regex: &RE_DEMO_ERROR,
                ref_type_fn: |_| ReferenceType::Demonstrative {
                    entity_type: EntityType::Error,
                },
            },
        ]
    }

    /// Detect unresolved references in a message
    pub fn detect_references(&self, message: &str) -> Vec<UnresolvedReference> {
        let mut references = Vec::new();
        let lower = message.to_lowercase();

        // Check demonstratives first (they're more specific)
        for pattern in &self.demonstrative_patterns {
            for cap in pattern.regex.captures_iter(&lower) {
                if let Some(m) = cap.get(0) {
                    references.push(UnresolvedReference {
                        text: m.as_str().to_string(),
                        ref_type: (pattern.ref_type_fn)(&cap),
                        start: m.start(),
                        end: m.end(),
                    });
                }
            }
        }

        // Check definite NPs
        for pattern in &self.definite_np_patterns {
            for cap in pattern.regex.captures_iter(&lower) {
                if let Some(m) = cap.get(0) {
                    // Skip if already covered by a demonstrative
                    let overlaps = references
                        .iter()
                        .any(|r| r.start <= m.start() && r.end >= m.end());
                    if !overlaps {
                        references.push(UnresolvedReference {
                            text: m.as_str().to_string(),
                            ref_type: (pattern.ref_type_fn)(&cap),
                            start: m.start(),
                            end: m.end(),
                        });
                    }
                }
            }
        }

        // Check pronouns last
        for pattern in &self.pronoun_patterns {
            for cap in pattern.regex.captures_iter(&lower) {
                if let Some(m) = cap.get(0) {
                    // Skip if already covered
                    let overlaps = references
                        .iter()
                        .any(|r| r.start <= m.start() && r.end >= m.end());
                    if !overlaps {
                        references.push(UnresolvedReference {
                            text: m.as_str().to_string(),
                            ref_type: (pattern.ref_type_fn)(&cap),
                            start: m.start(),
                            end: m.end(),
                        });
                    }
                }
            }
        }

        // Sort by position
        references.sort_by_key(|r| r.start);
        references
    }

    /// Resolve references using dialog state and entity store
    pub fn resolve(
        &self,
        references: &[UnresolvedReference],
        dialog_state: &DialogState,
        entity_store: &dyn EntityStoreT,
        graph: Option<&dyn RelationshipGraphT>,
    ) -> Vec<ResolvedReference> {
        let mut resolved = Vec::new();

        for reference in references {
            if let Some(resolution) =
                self.resolve_single(reference, dialog_state, entity_store, graph)
            {
                resolved.push(resolution);
            }
        }

        resolved
    }

    /// Resolve a single reference
    fn resolve_single(
        &self,
        reference: &UnresolvedReference,
        dialog_state: &DialogState,
        entity_store: &dyn EntityStoreT,
        graph: Option<&dyn RelationshipGraphT>,
    ) -> Option<ResolvedReference> {
        let compatible_types = reference.ref_type.compatible_types();

        // Gather candidates from dialog state and entity store
        let mut candidates: Vec<(&str, &EntityType, SalienceScore)> = Vec::new();

        // Check focus stack first (most likely candidates)
        for name in &dialog_state.focus_stack {
            if let Some(entity_type) = dialog_state.get_entity_type(name)
                && compatible_types.contains(entity_type)
            {
                let salience = self.compute_salience(name, entity_type, dialog_state, graph);
                candidates.push((name, entity_type, salience));
            }
        }

        // Also check entity store for additional candidates
        let entity_names: Vec<(String, EntityType)> = compatible_types
            .iter()
            .flat_map(|et| {
                entity_store
                    .entity_names_by_type(et)
                    .into_iter()
                    .map(move |name| (name, et.clone()))
            })
            .collect();

        for (entity_name, entity_type) in &entity_names {
            // Skip if already in candidates
            if candidates
                .iter()
                .any(|(n, _, _)| *n == entity_name.as_str())
            {
                continue;
            }

            let salience = self.compute_salience(entity_name, entity_type, dialog_state, graph);
            candidates.push((entity_name, entity_type, salience));
        }

        // Sort by salience score
        candidates.sort_by(|a, b| {
            b.2.total()
                .partial_cmp(&a.2.total())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take the best candidate
        candidates
            .first()
            .map(|(name, entity_type, salience)| ResolvedReference {
                reference: reference.clone(),
                antecedent: name.to_string(),
                entity_type: (*entity_type).clone(),
                confidence: salience.total(),
                salience: salience.clone(),
            })
    }

    /// Compute salience score for a candidate antecedent
    fn compute_salience(
        &self,
        name: &str,
        _entity_type: &EntityType,
        dialog_state: &DialogState,
        graph: Option<&dyn RelationshipGraphT>,
    ) -> SalienceScore {
        let recency = dialog_state.recency_score(name);
        let frequency = dialog_state.frequency_score(name);

        let graph_centrality = if let Some(g) = graph {
            if let Some(node) = g.get_node(name) {
                node.importance
            } else {
                0.0
            }
        } else {
            0.5 // Neutral if no graph available
        };

        // Type match is handled at the candidate selection stage
        let type_match = 1.0;

        // Syntactic prominence - subjects in focus stack get bonus
        let syntactic_prominence = if dialog_state.focus_stack.first() == Some(&name.to_string()) {
            1.0
        } else if dialog_state.focus_stack.contains(&name.to_string()) {
            0.5
        } else {
            0.0
        };

        SalienceScore {
            recency,
            frequency,
            graph_centrality,
            type_match,
            syntactic_prominence,
        }
    }

    /// Rewrite message with resolved references
    pub fn rewrite_with_resolutions(
        &self,
        message: &str,
        resolutions: &[ResolvedReference],
    ) -> String {
        if resolutions.is_empty() {
            return message.to_string();
        }

        // Sort resolutions by position (descending) to replace from end first
        let mut sorted = resolutions.to_vec();
        sorted.sort_by(|a, b| b.reference.start.cmp(&a.reference.start));

        let mut result = message.to_string();
        let lower = message.to_lowercase();

        for resolution in sorted {
            // Find the actual position in the original string
            // (accounting for case differences)
            let search_start = resolution.reference.start;
            let search_end = resolution.reference.end;

            if search_end <= lower.len() && search_start < search_end {
                // Create the replacement with bracket notation
                let replacement = format!("[{}]", resolution.antecedent);

                // Replace in the result string
                // We need to find the corresponding position in the (possibly modified) result
                let ref_text = &lower[search_start..search_end];
                if let Some(pos) = result.to_lowercase().find(ref_text) {
                    result = format!(
                        "{}{}{}",
                        &result[..pos],
                        replacement,
                        &result[pos + (search_end - search_start)..]
                    );
                }
            }
        }

        result
    }
}

impl Default for CoreferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_knowledge::knowledge::EntityStore;

    #[test]
    fn test_detect_pronouns() {
        let resolver = CoreferenceResolver::new();
        let refs = resolver.detect_references("Fix it and run the tests");

        assert!(!refs.is_empty());
        assert!(refs.iter().any(|r| r.text == "it"));
        assert!(refs[0].ref_type == ReferenceType::SingularNeutral);
    }

    #[test]
    fn test_detect_definite_np() {
        let resolver = CoreferenceResolver::new();
        let refs = resolver.detect_references("Update the file with the new logic");

        assert!(refs.iter().any(|r| r.text == "the file"));
        assert!(refs.iter().any(|r| matches!(
            &r.ref_type,
            ReferenceType::DefiniteNP { entity_type } if *entity_type == EntityType::File
        )));
    }

    #[test]
    fn test_detect_demonstrative() {
        let resolver = CoreferenceResolver::new();
        let refs = resolver.detect_references("Fix that error in the code");

        assert!(refs.iter().any(|r| r.text == "that error"));
        assert!(refs.iter().any(|r| matches!(
            &r.ref_type,
            ReferenceType::Demonstrative { entity_type } if *entity_type == EntityType::Error
        )));
    }

    #[test]
    fn test_dialog_state_mention() {
        let mut state = DialogState::new();
        state.mention_entity("main.rs", EntityType::File);
        state.next_turn();
        state.mention_entity("config.toml", EntityType::File);

        // config.toml should be at the top of focus stack
        assert_eq!(state.focus_stack[0], "config.toml");
        assert_eq!(state.focus_stack[1], "main.rs");

        // Recency score should be higher for config.toml
        assert!(state.recency_score("config.toml") > state.recency_score("main.rs"));
    }

    #[test]
    fn test_dialog_state_frequency() {
        let mut state = DialogState::new();
        state.mention_entity("main.rs", EntityType::File);
        state.next_turn();
        state.mention_entity("main.rs", EntityType::File);
        state.next_turn();
        state.mention_entity("config.toml", EntityType::File);

        // main.rs mentioned twice, should have higher frequency
        assert!(state.frequency_score("main.rs") > state.frequency_score("config.toml"));
    }

    #[test]
    fn test_resolve_pronoun() {
        let resolver = CoreferenceResolver::new();
        let mut state = DialogState::new();
        let entity_store = EntityStore::new();

        state.mention_entity("src/main.rs", EntityType::File);
        state.next_turn();

        let refs = resolver.detect_references("Fix it");
        let resolved = resolver.resolve(&refs, &state, &entity_store, None);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].antecedent, "src/main.rs");
    }

    #[test]
    fn test_resolve_type_constrained() {
        let resolver = CoreferenceResolver::new();
        let mut state = DialogState::new();
        let entity_store = EntityStore::new();

        // Mention a file and a function
        state.mention_entity("main.rs", EntityType::File);
        state.mention_entity("process_data", EntityType::Function);
        state.next_turn();

        // "the function" should resolve to the function, not the file
        let refs = resolver.detect_references("Update the function");
        let resolved = resolver.resolve(&refs, &state, &entity_store, None);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].antecedent, "process_data");
    }

    #[test]
    fn test_rewrite_with_resolutions() {
        let resolver = CoreferenceResolver::new();
        let mut state = DialogState::new();
        let entity_store = EntityStore::new();

        state.mention_entity("main.rs", EntityType::File);
        state.next_turn();

        let refs = resolver.detect_references("Fix it and test");
        let resolved = resolver.resolve(&refs, &state, &entity_store, None);
        let rewritten = resolver.rewrite_with_resolutions("Fix it and test", &resolved);

        assert_eq!(rewritten, "Fix [main.rs] and test");
    }

    #[test]
    fn test_salience_score_total() {
        let score = SalienceScore {
            recency: 1.0,
            frequency: 0.5,
            graph_centrality: 0.8,
            type_match: 1.0,
            syntactic_prominence: 0.5,
        };

        // 1.0*0.35 + 0.5*0.15 + 0.8*0.20 + 1.0*0.20 + 0.5*0.10
        // = 0.35 + 0.075 + 0.16 + 0.20 + 0.05 = 0.835
        assert!((score.total() - 0.835).abs() < 0.001);
    }

    #[test]
    fn test_empty_references() {
        let resolver = CoreferenceResolver::new();
        let refs = resolver.detect_references("Build the project using cargo");

        // "the project" doesn't match our patterns
        // This is expected - we only detect specific entity type references
        assert!(refs.is_empty() || !refs.iter().any(|r| r.text == "the project"));
    }

    #[test]
    fn test_multiple_references() {
        let resolver = CoreferenceResolver::new();
        let refs = resolver.detect_references("Fix it and update the file");

        assert!(refs.len() >= 2);
        // Should have both "it" and "the file"
        let texts: Vec<_> = refs.iter().map(|r| r.text.as_str()).collect();
        assert!(texts.contains(&"it"));
        assert!(texts.contains(&"the file"));
    }
}
