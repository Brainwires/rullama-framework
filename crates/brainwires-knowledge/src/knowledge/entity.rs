//! Entity Types for Knowledge Graph
//!
//! Types for representing extracted entities and their relationships
//! in conversation memory. Used by the relationship graph and knowledge system.

use std::collections::{HashMap, HashSet};

// Re-export EntityType from core (canonical definition)
pub use brainwires_core::graph::EntityType;

/// A named entity extracted from conversation
#[derive(Debug, Clone)]
pub struct Entity {
    /// Display name of the entity.
    pub name: String,
    /// The kind of entity (file, function, type, etc.).
    pub entity_type: EntityType,
    /// Message IDs where this entity appears.
    pub message_ids: Vec<String>,
    /// Unix timestamp when first seen.
    pub first_seen: i64,
    /// Unix timestamp when last seen.
    pub last_seen: i64,
    /// Total number of mentions.
    pub mention_count: u32,
}

impl Entity {
    /// Create a new entity with its first mention.
    pub fn new(name: String, entity_type: EntityType, message_id: String, timestamp: i64) -> Self {
        Self {
            name,
            entity_type,
            message_ids: vec![message_id],
            first_seen: timestamp,
            last_seen: timestamp,
            mention_count: 1,
        }
    }

    /// Record an additional mention of this entity.
    pub fn add_mention(&mut self, message_id: String, timestamp: i64) {
        if !self.message_ids.contains(&message_id) {
            self.message_ids.push(message_id);
        }
        self.last_seen = timestamp.max(self.last_seen);
        self.mention_count += 1;
    }
}

/// Relationship between entities
#[derive(Debug, Clone)]
pub enum Relationship {
    /// One entity defines another.
    Defines {
        /// The defining entity.
        definer: String,
        /// The entity being defined.
        defined: String,
        /// Context of the definition.
        context: String,
    },
    /// One entity references another.
    References {
        /// Source entity.
        from: String,
        /// Target entity.
        to: String,
    },
    /// One entity modifies another.
    Modifies {
        /// The modifying entity.
        modifier: String,
        /// The modified entity.
        modified: String,
        /// Kind of modification.
        change_type: String,
    },
    /// One entity depends on another.
    DependsOn {
        /// The dependent entity.
        dependent: String,
        /// The dependency.
        dependency: String,
    },
    /// One entity contains another.
    Contains {
        /// The container entity.
        container: String,
        /// The contained entity.
        contained: String,
    },
    /// Two entities co-occur in a message.
    CoOccurs {
        /// First entity.
        entity_a: String,
        /// Second entity.
        entity_b: String,
        /// Message where co-occurrence was observed.
        message_id: String,
    },
}

/// Extraction result from a single message
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Extracted entities as (name, type) pairs.
    pub entities: Vec<(String, EntityType)>,
    /// Extracted relationships between entities.
    pub relationships: Vec<Relationship>,
}

// ── Memory poisoning detection ────────────────────────────────────────────────

/// Why two stored facts were flagged as a potential contradiction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContradictionKind {
    /// Two `Defines` relationships share the same definer + defined but have different contexts.
    ConflictingDefinition,
    /// Two `Modifies` relationships describe different change types for the same modifier + target.
    ConflictingModification,
}

/// A potential contradiction detected when inserting a new fact.
///
/// The store does **not** silently overwrite the existing entry; instead it
/// appends both relationships and records this event so callers can surface it
/// for human review.
#[derive(Debug, Clone)]
pub struct ContradictionEvent {
    /// What kind of contradiction was detected.
    pub kind: ContradictionKind,
    /// The entity key (e.g. `"file:main.rs"`) involved.
    pub subject: String,
    /// Context string from the previously stored relationship.
    pub existing_context: String,
    /// Context string from the newly inserted relationship.
    pub new_context: String,
}

// ── Entity store ──────────────────────────────────────────────────────────────

/// Entity store for tracking entities across a conversation
#[derive(Debug, Default)]
pub struct EntityStore {
    entities: HashMap<String, Entity>,
    relationships: Vec<Relationship>,
    /// Contradiction events accumulated since the last call to [`Self::drain_contradictions`].
    contradictions: Vec<ContradictionEvent>,
}

impl EntityStore {
    /// Create a new empty entity store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an extraction result, recording entities and relationships.
    pub fn add_extraction(&mut self, result: ExtractionResult, message_id: &str, timestamp: i64) {
        for (name, entity_type) in result.entities {
            let key = format!("{}:{}", entity_type.as_str(), name);
            if let Some(entity) = self.entities.get_mut(&key) {
                entity.add_mention(message_id.to_string(), timestamp);
            } else {
                self.entities.insert(
                    key,
                    Entity::new(name, entity_type, message_id.to_string(), timestamp),
                );
            }
        }

        // Check for contradictions before appending each relationship.
        for new_rel in result.relationships {
            self.check_and_record_contradiction(&new_rel);
            self.relationships.push(new_rel);
        }
    }

    /// Inspect `new_rel` against previously stored relationships and record any
    /// contradiction events.  Both the existing and the new relationship are kept
    /// so no information is silently discarded.
    fn check_and_record_contradiction(&mut self, new_rel: &Relationship) {
        match new_rel {
            Relationship::Defines {
                definer,
                defined,
                context: new_ctx,
            } => {
                for existing in &self.relationships {
                    if let Relationship::Defines {
                        definer: ex_definer,
                        defined: ex_defined,
                        context: ex_ctx,
                    } = existing
                        && ex_definer == definer
                        && ex_defined == defined
                        && ex_ctx != new_ctx
                    {
                        self.contradictions.push(ContradictionEvent {
                            kind: ContradictionKind::ConflictingDefinition,
                            subject: format!("{}::{}", definer, defined),
                            existing_context: ex_ctx.clone(),
                            new_context: new_ctx.clone(),
                        });
                        break;
                    }
                }
            }
            Relationship::Modifies {
                modifier,
                modified,
                change_type: new_change,
            } => {
                for existing in &self.relationships {
                    if let Relationship::Modifies {
                        modifier: ex_modifier,
                        modified: ex_modified,
                        change_type: ex_change,
                    } = existing
                        && ex_modifier == modifier
                        && ex_modified == modified
                        && ex_change != new_change
                    {
                        self.contradictions.push(ContradictionEvent {
                            kind: ContradictionKind::ConflictingModification,
                            subject: format!("{}::{}", modifier, modified),
                            existing_context: ex_change.clone(),
                            new_context: new_change.clone(),
                        });
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    /// Returns all contradiction events accumulated so far (for inspection /
    /// human review) without removing them.
    pub fn pending_contradictions(&self) -> &[ContradictionEvent] {
        &self.contradictions
    }

    /// Drains and returns all accumulated contradiction events, clearing the
    /// internal buffer.
    pub fn drain_contradictions(&mut self) -> Vec<ContradictionEvent> {
        std::mem::take(&mut self.contradictions)
    }

    /// Look up an entity by name and type.
    pub fn get(&self, name: &str, entity_type: &EntityType) -> Option<&Entity> {
        let key = format!("{}:{}", entity_type.as_str(), name);
        self.entities.get(&key)
    }

    /// Get all entities of a given type.
    pub fn get_by_type(&self, entity_type: &EntityType) -> Vec<&Entity> {
        self.entities
            .values()
            .filter(|e| &e.entity_type == entity_type)
            .collect()
    }

    /// Get the most-mentioned entities, up to `limit`.
    pub fn get_top_entities(&self, limit: usize) -> Vec<&Entity> {
        let mut entities: Vec<_> = self.entities.values().collect();
        entities.sort_by(|a, b| b.mention_count.cmp(&a.mention_count));
        entities.into_iter().take(limit).collect()
    }

    /// Get names of entities related to the given entity.
    pub fn get_related(&self, entity_name: &str) -> Vec<String> {
        let mut related = HashSet::new();
        for rel in &self.relationships {
            match rel {
                Relationship::CoOccurs {
                    entity_a, entity_b, ..
                } => {
                    if entity_a == entity_name {
                        related.insert(entity_b.clone());
                    } else if entity_b == entity_name {
                        related.insert(entity_a.clone());
                    }
                }
                Relationship::Contains {
                    container,
                    contained,
                } => {
                    if container == entity_name {
                        related.insert(contained.clone());
                    } else if contained == entity_name {
                        related.insert(container.clone());
                    }
                }
                Relationship::References { from, to } => {
                    if from == entity_name {
                        related.insert(to.clone());
                    } else if to == entity_name {
                        related.insert(from.clone());
                    }
                }
                Relationship::DependsOn {
                    dependent,
                    dependency,
                } => {
                    if dependent == entity_name {
                        related.insert(dependency.clone());
                    } else if dependency == entity_name {
                        related.insert(dependent.clone());
                    }
                }
                Relationship::Modifies {
                    modifier, modified, ..
                } => {
                    if modifier == entity_name {
                        related.insert(modified.clone());
                    } else if modified == entity_name {
                        related.insert(modifier.clone());
                    }
                }
                Relationship::Defines {
                    definer, defined, ..
                } => {
                    if definer == entity_name {
                        related.insert(defined.clone());
                    } else if defined == entity_name {
                        related.insert(definer.clone());
                    }
                }
            }
        }
        related.into_iter().collect()
    }

    /// Get all message IDs associated with an entity name.
    pub fn get_message_ids(&self, entity_name: &str) -> Vec<String> {
        self.entities
            .values()
            .filter(|e| e.name == entity_name)
            .flat_map(|e| e.message_ids.clone())
            .collect()
    }

    /// Iterate over all stored entities.
    pub fn all_entities(&self) -> impl Iterator<Item = &Entity> {
        self.entities.values()
    }

    /// Get all stored relationships.
    pub fn all_relationships(&self) -> &[Relationship] {
        &self.relationships
    }

    /// Get statistics about the entity store.
    pub fn stats(&self) -> EntityStoreStats {
        let mut by_type = HashMap::new();
        for entity in self.entities.values() {
            *by_type.entry(entity.entity_type.as_str()).or_insert(0) += 1;
        }
        EntityStoreStats {
            total_entities: self.entities.len(),
            total_relationships: self.relationships.len(),
            entities_by_type: by_type,
        }
    }
}

impl brainwires_core::graph::EntityStoreT for EntityStore {
    fn entity_names_by_type(&self, entity_type: &EntityType) -> Vec<String> {
        self.get_by_type(entity_type)
            .iter()
            .map(|e| e.name.clone())
            .collect()
    }

    fn top_entity_info(&self, limit: usize) -> Vec<(String, EntityType)> {
        self.get_top_entities(limit)
            .iter()
            .map(|e| (e.name.clone(), e.entity_type.clone()))
            .collect()
    }
}

/// Statistics about the entity store.
#[derive(Debug)]
pub struct EntityStoreStats {
    /// Total number of entities.
    pub total_entities: usize,
    /// Total number of relationships.
    pub total_relationships: usize,
    /// Entity counts grouped by type.
    pub entities_by_type: HashMap<&'static str, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_as_str() {
        assert_eq!(EntityType::File.as_str(), "file");
        assert_eq!(EntityType::Function.as_str(), "function");
    }

    #[test]
    fn test_entity_lifecycle() {
        let mut entity = Entity::new("main.rs".into(), EntityType::File, "msg-1".into(), 100);
        assert_eq!(entity.mention_count, 1);
        entity.add_mention("msg-2".into(), 200);
        assert_eq!(entity.mention_count, 2);
        assert_eq!(entity.last_seen, 200);
    }

    #[test]
    fn test_entity_store() {
        let mut store = EntityStore::new();
        let result = ExtractionResult {
            entities: vec![
                ("main.rs".into(), EntityType::File),
                ("process".into(), EntityType::Function),
            ],
            relationships: vec![],
        };
        store.add_extraction(result, "msg-1", 100);
        assert_eq!(store.stats().total_entities, 2);
    }

    // ── Memory poisoning detection ─────────────────────────────────────────

    #[test]
    fn test_no_contradiction_on_fresh_store() {
        let mut store = EntityStore::new();
        let result = ExtractionResult {
            entities: vec![],
            relationships: vec![Relationship::Defines {
                definer: "main".into(),
                defined: "return_type".into(),
                context: "returns i32".into(),
            }],
        };
        store.add_extraction(result, "msg-1", 100);
        assert!(store.pending_contradictions().is_empty());
    }

    #[test]
    fn test_contradicting_definitions_flagged() {
        let mut store = EntityStore::new();

        store.add_extraction(
            ExtractionResult {
                entities: vec![],
                relationships: vec![Relationship::Defines {
                    definer: "main".into(),
                    defined: "return_type".into(),
                    context: "returns i32".into(),
                }],
            },
            "msg-1",
            100,
        );

        // Same definer/defined, different context → contradiction
        store.add_extraction(
            ExtractionResult {
                entities: vec![],
                relationships: vec![Relationship::Defines {
                    definer: "main".into(),
                    defined: "return_type".into(),
                    context: "returns String".into(),
                }],
            },
            "msg-2",
            200,
        );

        let contradictions = store.pending_contradictions();
        assert_eq!(contradictions.len(), 1);
        assert_eq!(
            contradictions[0].kind,
            ContradictionKind::ConflictingDefinition
        );
        assert_eq!(contradictions[0].subject, "main::return_type");
        assert_eq!(contradictions[0].existing_context, "returns i32");
        assert_eq!(contradictions[0].new_context, "returns String");
    }

    #[test]
    fn test_identical_definitions_not_flagged() {
        let mut store = EntityStore::new();

        for msg_id in ["msg-1", "msg-2"] {
            store.add_extraction(
                ExtractionResult {
                    entities: vec![],
                    relationships: vec![Relationship::Defines {
                        definer: "Config".into(),
                        defined: "timeout".into(),
                        context: "30 seconds".into(),
                    }],
                },
                msg_id,
                100,
            );
        }

        assert!(store.pending_contradictions().is_empty());
    }

    #[test]
    fn test_contradicting_modifications_flagged() {
        let mut store = EntityStore::new();

        store.add_extraction(
            ExtractionResult {
                entities: vec![],
                relationships: vec![Relationship::Modifies {
                    modifier: "patch_v2".into(),
                    modified: "timeout".into(),
                    change_type: "increase".into(),
                }],
            },
            "msg-1",
            100,
        );

        store.add_extraction(
            ExtractionResult {
                entities: vec![],
                relationships: vec![Relationship::Modifies {
                    modifier: "patch_v2".into(),
                    modified: "timeout".into(),
                    change_type: "decrease".into(),
                }],
            },
            "msg-2",
            200,
        );

        let contradictions = store.pending_contradictions();
        assert_eq!(contradictions.len(), 1);
        assert_eq!(
            contradictions[0].kind,
            ContradictionKind::ConflictingModification
        );
    }

    #[test]
    fn test_drain_contradictions_clears_buffer() {
        let mut store = EntityStore::new();

        for ctx in ["returns i32", "returns String"] {
            store.add_extraction(
                ExtractionResult {
                    entities: vec![],
                    relationships: vec![Relationship::Defines {
                        definer: "main".into(),
                        defined: "return_type".into(),
                        context: ctx.into(),
                    }],
                },
                "msg-1",
                100,
            );
        }

        assert!(!store.pending_contradictions().is_empty());
        let drained = store.drain_contradictions();
        assert_eq!(drained.len(), 1);
        assert!(store.pending_contradictions().is_empty());
    }

    #[test]
    fn test_both_relationships_retained_after_contradiction() {
        let mut store = EntityStore::new();

        store.add_extraction(
            ExtractionResult {
                entities: vec![],
                relationships: vec![Relationship::Defines {
                    definer: "fn".into(),
                    defined: "x".into(),
                    context: "old".into(),
                }],
            },
            "msg-1",
            100,
        );

        store.add_extraction(
            ExtractionResult {
                entities: vec![],
                relationships: vec![Relationship::Defines {
                    definer: "fn".into(),
                    defined: "x".into(),
                    context: "new".into(),
                }],
            },
            "msg-2",
            200,
        );

        // Both relationships are kept — no silent overwrite
        assert_eq!(store.all_relationships().len(), 2);
        let event = &store.pending_contradictions()[0];
        assert_eq!(event.existing_context, "old");
        assert_eq!(event.new_context, "new");
    }
}
