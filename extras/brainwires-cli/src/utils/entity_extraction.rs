//! Entity Extraction for Conversation Memory
//!
//! Extracts named entities (files, functions, people, concepts) from messages
//! to build a relationship graph for better context retrieval.
//!
//! Entity types (EntityType, Entity, Relationship, ExtractionResult, EntityStore)
//! are re-exported from the brainwires-storage framework crate.
//!
//! # Local LLM Enhancement
//!
//! When the `llama-cpp-2` feature is enabled, entity extraction can be enhanced
//! with semantic understanding via [`EntityEnhancer`], which:
//! - Extracts entities beyond regex patterns (semantic entities)
//! - Classifies relationships semantically (not just co-occurrence)
//! - Identifies domain concepts dynamically

// Re-export entity types from framework (brainwires-knowledge::knowledge)
pub use brainwires::knowledge::entity::{
    Entity, EntityStore, EntityStoreStats, EntityType, ExtractionResult, Relationship,
};

use regex::Regex;
use std::collections::HashSet;

use brainwires::reasoning::{EntityEnhancer, SemanticEntityType};

/// Entity extractor with compiled regex patterns
pub struct EntityExtractor {
    file_pattern: Regex,
    rust_fn_pattern: Regex,
    js_fn_pattern: Regex,
    python_fn_pattern: Regex,
    type_pattern: Regex,
    var_pattern: Regex,
    error_pattern: Regex,
    command_pattern: Regex,
    concepts: HashSet<String>,
}

impl EntityExtractor {
    pub fn new() -> Self {
        Self {
            file_pattern: Regex::new(
                r"([\w./\-]+\.(?:rs|js|ts|tsx|jsx|py|go|java|c|cpp|h|hpp|md|json|toml|yaml|yml|sh|sql|html|css))\b"
            ).unwrap(),
            rust_fn_pattern: Regex::new(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)").unwrap(),
            js_fn_pattern: Regex::new(r"function\s+(\w+)").unwrap(),
            python_fn_pattern: Regex::new(r"def\s+(\w+)").unwrap(),
            type_pattern: Regex::new(r"(?:struct|class|interface|enum|type)\s+(\w+)").unwrap(),
            var_pattern: Regex::new(r"(?:let|const|var|mut)\s+(\w+)").unwrap(),
            error_pattern: Regex::new(r"(\w+Error)(?:\s|:|\()").unwrap(),
            command_pattern: Regex::new(r"(?:^|\s)(?:cargo|npm|yarn|pip|git|docker)\s+(\w+)").unwrap(),
            concepts: [
                "api", "rest", "graphql", "authentication", "authorization",
                "database", "cache", "queue", "websocket", "http",
                "encryption", "jwt", "oauth", "session", "middleware",
                "component", "hook", "state", "async", "promise",
                "vector", "embedding", "rag", "llm", "prompt", "token",
            ].iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn extract(&self, content: &str, message_id: &str) -> ExtractionResult {
        let mut entities = Vec::new();
        let mut seen = HashSet::new();

        // Extract files
        for cap in self.file_pattern.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if seen.insert((name.clone(), EntityType::File)) {
                    entities.push((name, EntityType::File));
                }
            }
        }

        // Extract Rust functions
        for cap in self.rust_fn_pattern.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if seen.insert((name.clone(), EntityType::Function)) {
                    entities.push((name, EntityType::Function));
                }
            }
        }

        // Extract JS functions
        for cap in self.js_fn_pattern.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if seen.insert((name.clone(), EntityType::Function)) {
                    entities.push((name, EntityType::Function));
                }
            }
        }

        // Extract Python functions
        for cap in self.python_fn_pattern.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if seen.insert((name.clone(), EntityType::Function)) {
                    entities.push((name, EntityType::Function));
                }
            }
        }

        // Extract types
        for cap in self.type_pattern.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if seen.insert((name.clone(), EntityType::Type)) {
                    entities.push((name, EntityType::Type));
                }
            }
        }

        // Extract variables (only longer names)
        for cap in self.var_pattern.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if name.len() > 3 && seen.insert((name.clone(), EntityType::Variable)) {
                    entities.push((name, EntityType::Variable));
                }
            }
        }

        // Extract errors
        for cap in self.error_pattern.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if seen.insert((name.clone(), EntityType::Error)) {
                    entities.push((name, EntityType::Error));
                }
            }
        }

        // Extract commands
        for cap in self.command_pattern.captures_iter(content) {
            if let Some(m) = cap.get(1) {
                let name = m.as_str().to_string();
                if seen.insert((name.clone(), EntityType::Command)) {
                    entities.push((name, EntityType::Command));
                }
            }
        }

        // Extract concepts
        let lower = content.to_lowercase();
        for concept in &self.concepts {
            if lower.contains(concept) && seen.insert((concept.clone(), EntityType::Concept)) {
                entities.push((concept.clone(), EntityType::Concept));
            }
        }

        let relationships = self.extract_relationships(&entities, message_id, content);

        ExtractionResult {
            entities,
            relationships,
        }
    }

    fn extract_relationships(
        &self,
        entities: &[(String, EntityType)],
        message_id: &str,
        content: &str,
    ) -> Vec<Relationship> {
        let mut relationships = Vec::new();

        // Co-occurrence relationships
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                relationships.push(Relationship::CoOccurs {
                    entity_a: entities[i].0.clone(),
                    entity_b: entities[j].0.clone(),
                    message_id: message_id.to_string(),
                });
            }
        }

        // File contains function/type relationships
        let files: Vec<_> = entities
            .iter()
            .filter(|(_, t)| *t == EntityType::File)
            .collect();
        let code_entities: Vec<_> = entities
            .iter()
            .filter(|(_, t)| matches!(t, EntityType::Function | EntityType::Type))
            .collect();

        if files.len() == 1 {
            for (entity, _) in &code_entities {
                relationships.push(Relationship::Contains {
                    container: files[0].0.clone(),
                    contained: entity.clone(),
                });
            }
        }

        // Look for modification patterns
        let modify_patterns = [
            "changed", "updated", "modified", "fixed", "added", "removed",
        ];
        let lower = content.to_lowercase();
        for pattern in modify_patterns {
            if lower.contains(pattern) {
                for (file, t) in &files {
                    if *t == EntityType::File {
                        relationships.push(Relationship::Modifies {
                            modifier: "user".to_string(),
                            modified: file.clone(),
                            change_type: pattern.to_string(),
                        });
                    }
                }
            }
        }

        relationships
    }

    pub fn count_entities(&self, content: &str) -> usize {
        let mut count = 0;
        count += self.file_pattern.find_iter(content).count();
        count += self.rust_fn_pattern.find_iter(content).count();
        count += self.type_pattern.find_iter(content).count();
        count
    }
}

impl Default for EntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityExtractor {
    /// Extract entities with optional local inference enhancement
    ///
    /// When an enhancer is provided, this method:
    /// 1. Performs regex-based extraction
    /// 2. Uses local LLM to extract additional semantic entities
    /// 3. Merges the results, deduplicating by name
    pub async fn extract_enhanced(
        &self,
        content: &str,
        message_id: &str,
        enhancer: Option<&EntityEnhancer>,
    ) -> ExtractionResult {
        // Get regex-based extraction first
        let regex_result = self.extract(content, message_id);

        // Enhance with local LLM if available
        if let Some(e) = enhancer {
            let enhancement = e.enhance(content).await;

            // Merge entities
            let mut merged_entities = regex_result.entities.clone();
            let mut seen: HashSet<String> = regex_result
                .entities
                .iter()
                .map(|(name, _)| name.to_lowercase())
                .collect();

            for enhanced_entity in enhancement.entities {
                let name_lower = enhanced_entity.name.to_lowercase();
                if !seen.contains(&name_lower) {
                    seen.insert(name_lower);
                    // Convert SemanticEntityType to EntityType
                    if let Some(entity_type) =
                        convert_semantic_to_entity_type(&enhanced_entity.entity_type)
                    {
                        merged_entities.push((enhanced_entity.name, entity_type));
                    }
                }
            }

            // Merge relationships
            let mut merged_relationships = regex_result.relationships;
            for enhanced_rel in enhancement.relationships {
                merged_relationships.push(convert_enhanced_relationship(&enhanced_rel, message_id));
            }

            // Add extracted concepts as Concept entities
            for concept in enhancement.concepts {
                let concept_lower = concept.to_lowercase();
                if !seen.contains(&concept_lower) {
                    seen.insert(concept_lower);
                    merged_entities.push((concept, EntityType::Concept));
                }
            }

            ExtractionResult {
                entities: merged_entities,
                relationships: merged_relationships,
            }
        } else {
            regex_result
        }
    }

    /// Extract with heuristic enhancement (no local LLM call)
    ///
    /// Uses pattern-based heuristics from EntityEnhancer to find
    /// additional entities like URLs, paths, bug references, etc.
    pub fn extract_with_heuristics(
        &self,
        content: &str,
        message_id: &str,
        enhancer: &EntityEnhancer,
    ) -> ExtractionResult {
        let regex_result = self.extract(content, message_id);

        // Get heuristic entities
        let heuristic_entities = enhancer.extract_heuristic(content);

        // Merge
        let mut merged_entities = regex_result.entities.clone();
        let mut seen: HashSet<String> = regex_result
            .entities
            .iter()
            .map(|(name, _)| name.to_lowercase())
            .collect();

        for enhanced_entity in heuristic_entities {
            let name_lower = enhanced_entity.name.to_lowercase();
            if !seen.contains(&name_lower) {
                seen.insert(name_lower);
                if let Some(entity_type) =
                    convert_semantic_to_entity_type(&enhanced_entity.entity_type)
                {
                    merged_entities.push((enhanced_entity.name, entity_type));
                }
            }
        }

        ExtractionResult {
            entities: merged_entities,
            relationships: regex_result.relationships,
        }
    }
}

/// Convert SemanticEntityType to EntityType
fn convert_semantic_to_entity_type(semantic: &SemanticEntityType) -> Option<EntityType> {
    match semantic {
        SemanticEntityType::File => Some(EntityType::File),
        SemanticEntityType::Function => Some(EntityType::Function),
        SemanticEntityType::Type => Some(EntityType::Type),
        SemanticEntityType::Variable => Some(EntityType::Variable),
        SemanticEntityType::Module | SemanticEntityType::Package => Some(EntityType::File),
        SemanticEntityType::Concept
        | SemanticEntityType::Pattern
        | SemanticEntityType::Algorithm
        | SemanticEntityType::Protocol => Some(EntityType::Concept),
        SemanticEntityType::Command | SemanticEntityType::Operation | SemanticEntityType::Task => {
            Some(EntityType::Command)
        }
        SemanticEntityType::Error | SemanticEntityType::Bug => Some(EntityType::Error),
        SemanticEntityType::Fix | SemanticEntityType::Feature => Some(EntityType::Concept),
        SemanticEntityType::Person | SemanticEntityType::Role => None, // No equivalent
        SemanticEntityType::Url | SemanticEntityType::Path | SemanticEntityType::Identifier => {
            Some(EntityType::File)
        }
    }
}

/// Convert EnhancedRelationship to Relationship
fn convert_enhanced_relationship(
    enhanced: &brainwires::reasoning::EnhancedRelationship,
    message_id: &str,
) -> Relationship {
    use brainwires::reasoning::RelationType;

    match &enhanced.relation_type {
        RelationType::Contains => Relationship::Contains {
            container: enhanced.from.clone(),
            contained: enhanced.to.clone(),
        },
        RelationType::DefinedIn => Relationship::Defines {
            definer: enhanced.to.clone(), // reversed: defined_in means to defines from
            defined: enhanced.from.clone(),
            context: String::new(),
        },
        RelationType::Calls | RelationType::Uses => Relationship::References {
            from: enhanced.from.clone(),
            to: enhanced.to.clone(),
        },
        RelationType::Modifies | RelationType::Creates | RelationType::Deletes => {
            Relationship::Modifies {
                modifier: enhanced.from.clone(),
                modified: enhanced.to.clone(),
                change_type: enhanced.relation_type.as_str().to_string(),
            }
        }
        RelationType::DependsOn => Relationship::DependsOn {
            dependent: enhanced.from.clone(),
            dependency: enhanced.to.clone(),
        },
        RelationType::Fixes | RelationType::Replaces => Relationship::Modifies {
            modifier: enhanced.from.clone(),
            modified: enhanced.to.clone(),
            change_type: enhanced.relation_type.as_str().to_string(),
        },
        _ => Relationship::CoOccurs {
            entity_a: enhanced.from.clone(),
            entity_b: enhanced.to.clone(),
            message_id: message_id.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_files() {
        let extractor = EntityExtractor::new();
        let content = "Check the file src/main.rs";
        let result = extractor.extract(content, "msg1");
        let files: Vec<_> = result
            .entities
            .iter()
            .filter(|(_, t)| *t == EntityType::File)
            .collect();
        assert!(files.iter().any(|(n, _)| n == "src/main.rs"));
    }

    #[test]
    fn test_extract_rust_functions() {
        let extractor = EntityExtractor::new();
        let content = "fn main() { } pub async fn process_data() { }";
        let result = extractor.extract(content, "msg1");
        let funcs: Vec<_> = result
            .entities
            .iter()
            .filter(|(_, t)| *t == EntityType::Function)
            .collect();
        assert!(funcs.iter().any(|(n, _)| n == "main"));
        assert!(funcs.iter().any(|(n, _)| n == "process_data"));
    }

    #[test]
    fn test_extract_types() {
        let extractor = EntityExtractor::new();
        let content = "struct Message { } enum Role { User }";
        let result = extractor.extract(content, "msg1");
        let types: Vec<_> = result
            .entities
            .iter()
            .filter(|(_, t)| *t == EntityType::Type)
            .collect();
        assert!(types.iter().any(|(n, _)| n == "Message"));
        assert!(types.iter().any(|(n, _)| n == "Role"));
    }

    #[test]
    fn test_extract_concepts() {
        let extractor = EntityExtractor::new();
        let content = "We need to implement authentication with jwt tokens";
        let result = extractor.extract(content, "msg1");
        let concepts: Vec<_> = result
            .entities
            .iter()
            .filter(|(_, t)| *t == EntityType::Concept)
            .collect();
        assert!(concepts.iter().any(|(n, _)| n == "authentication"));
        assert!(concepts.iter().any(|(n, _)| n == "jwt"));
    }

    #[test]
    fn test_entity_store() {
        let extractor = EntityExtractor::new();
        let mut store = EntityStore::new();
        let content1 = "Working on src/main.rs with fn process";
        let result1 = extractor.extract(content1, "msg1");
        store.add_extraction(result1, "msg1", 1000);
        let content2 = "Updated src/main.rs again";
        let result2 = extractor.extract(content2, "msg2");
        store.add_extraction(result2, "msg2", 2000);
        let entity = store.get("src/main.rs", &EntityType::File);
        assert!(entity.is_some());
        assert_eq!(entity.unwrap().mention_count, 2);
    }
}
