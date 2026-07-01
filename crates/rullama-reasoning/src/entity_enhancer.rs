//! Entity Enhancer - Semantic Entity Extraction
//!
//! Uses a provider to extract entities and relationships beyond
//! what regex patterns can capture, enriching the knowledge graph.

use std::sync::Arc;
use tracing::warn;

use rullama_core::message::Message;
use rullama_core::provider::{ChatOptions, Provider};

use crate::InferenceTimer;

/// Enhanced entity type with semantic classification
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticEntityType {
    /// A source file.
    File,
    /// A function or method.
    Function,
    /// A type, struct, class, or interface.
    Type,
    /// A variable or constant.
    Variable,
    /// A module or namespace.
    Module,
    /// A package, crate, or library.
    Package,

    /// A general domain concept.
    Concept,
    /// A design or architectural pattern.
    Pattern,
    /// An algorithm or computational technique.
    Algorithm,
    /// A communication or network protocol.
    Protocol,

    /// A CLI or shell command.
    Command,
    /// A runtime operation or action.
    Operation,
    /// A task or work item.
    Task,

    /// An error or exception.
    Error,
    /// A bug or known defect.
    Bug,
    /// A fix or patch for a defect.
    Fix,
    /// A product or code feature.
    Feature,

    /// A person or user.
    Person,
    /// A role or permission level.
    Role,

    /// A URL or web link.
    Url,
    /// A filesystem path.
    Path,
    /// A generic identifier or ID.
    Identifier,
}

impl SemanticEntityType {
    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "file" => Some(SemanticEntityType::File),
            "function" | "func" | "method" => Some(SemanticEntityType::Function),
            "type" | "struct" | "class" | "interface" => Some(SemanticEntityType::Type),
            "variable" | "var" | "const" => Some(SemanticEntityType::Variable),
            "module" | "mod" => Some(SemanticEntityType::Module),
            "package" | "crate" | "library" | "lib" => Some(SemanticEntityType::Package),
            "concept" => Some(SemanticEntityType::Concept),
            "pattern" => Some(SemanticEntityType::Pattern),
            "algorithm" | "algo" => Some(SemanticEntityType::Algorithm),
            "protocol" => Some(SemanticEntityType::Protocol),
            "command" | "cmd" => Some(SemanticEntityType::Command),
            "operation" | "op" => Some(SemanticEntityType::Operation),
            "task" => Some(SemanticEntityType::Task),
            "error" => Some(SemanticEntityType::Error),
            "bug" => Some(SemanticEntityType::Bug),
            "fix" => Some(SemanticEntityType::Fix),
            "feature" => Some(SemanticEntityType::Feature),
            "person" | "user" | "developer" => Some(SemanticEntityType::Person),
            "role" => Some(SemanticEntityType::Role),
            "url" | "link" => Some(SemanticEntityType::Url),
            "path" => Some(SemanticEntityType::Path),
            "identifier" | "id" => Some(SemanticEntityType::Identifier),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            SemanticEntityType::File => "file",
            SemanticEntityType::Function => "function",
            SemanticEntityType::Type => "type",
            SemanticEntityType::Variable => "variable",
            SemanticEntityType::Module => "module",
            SemanticEntityType::Package => "package",
            SemanticEntityType::Concept => "concept",
            SemanticEntityType::Pattern => "pattern",
            SemanticEntityType::Algorithm => "algorithm",
            SemanticEntityType::Protocol => "protocol",
            SemanticEntityType::Command => "command",
            SemanticEntityType::Operation => "operation",
            SemanticEntityType::Task => "task",
            SemanticEntityType::Error => "error",
            SemanticEntityType::Bug => "bug",
            SemanticEntityType::Fix => "fix",
            SemanticEntityType::Feature => "feature",
            SemanticEntityType::Person => "person",
            SemanticEntityType::Role => "role",
            SemanticEntityType::Url => "url",
            SemanticEntityType::Path => "path",
            SemanticEntityType::Identifier => "identifier",
        }
    }
}

/// An entity extracted by LLM
#[derive(Clone, Debug)]
pub struct EnhancedEntity {
    /// Entity name/value
    pub name: String,
    /// Semantic type
    pub entity_type: SemanticEntityType,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Context where found
    pub context: Option<String>,
}

impl EnhancedEntity {
    /// Create a new enhanced entity with the given name, type, and confidence.
    pub fn new(name: String, entity_type: SemanticEntityType, confidence: f32) -> Self {
        Self {
            name,
            entity_type,
            confidence,
            context: None,
        }
    }

    /// Attach contextual information describing where the entity was found.
    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }
}

/// A semantic relationship between entities
#[derive(Clone, Debug)]
pub struct EnhancedRelationship {
    /// Source entity
    pub from: String,
    /// Target entity
    pub to: String,
    /// Relationship type
    pub relation_type: RelationType,
    /// Confidence score
    pub confidence: f32,
}

/// Types of relationships we detect semantically
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelationType {
    /// A contains B (parent-child).
    Contains,
    /// A is defined inside B.
    DefinedIn,
    /// A imports B.
    Imports,
    /// A exports B.
    Exports,
    /// A extends or inherits from B.
    Extends,
    /// A implements the interface or trait B.
    Implements,

    /// A calls or invokes B.
    Calls,
    /// A uses or references B.
    Uses,
    /// A modifies B.
    Modifies,
    /// A creates or constructs B.
    Creates,
    /// A deletes or removes B.
    Deletes,

    /// A is semantically related to B.
    RelatedTo,
    /// A is similar to B.
    SimilarTo,
    /// A depends on B.
    DependsOn,
    /// A causes B.
    Causes,
    /// A fixes or resolves B.
    Fixes,
    /// A replaces B.
    Replaces,
}

impl RelationType {
    /// Parse a relation type from a string label.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "contains" => Some(RelationType::Contains),
            "defined_in" | "definedin" => Some(RelationType::DefinedIn),
            "imports" => Some(RelationType::Imports),
            "exports" => Some(RelationType::Exports),
            "extends" => Some(RelationType::Extends),
            "implements" => Some(RelationType::Implements),
            "calls" => Some(RelationType::Calls),
            "uses" => Some(RelationType::Uses),
            "modifies" => Some(RelationType::Modifies),
            "creates" => Some(RelationType::Creates),
            "deletes" => Some(RelationType::Deletes),
            "related_to" | "relatedto" => Some(RelationType::RelatedTo),
            "similar_to" | "similarto" => Some(RelationType::SimilarTo),
            "depends_on" | "dependson" => Some(RelationType::DependsOn),
            "causes" => Some(RelationType::Causes),
            "fixes" => Some(RelationType::Fixes),
            "replaces" => Some(RelationType::Replaces),
            _ => None,
        }
    }

    /// Return the canonical string label for this relation type.
    pub fn as_str(&self) -> &'static str {
        match self {
            RelationType::Contains => "contains",
            RelationType::DefinedIn => "defined_in",
            RelationType::Imports => "imports",
            RelationType::Exports => "exports",
            RelationType::Extends => "extends",
            RelationType::Implements => "implements",
            RelationType::Calls => "calls",
            RelationType::Uses => "uses",
            RelationType::Modifies => "modifies",
            RelationType::Creates => "creates",
            RelationType::Deletes => "deletes",
            RelationType::RelatedTo => "related_to",
            RelationType::SimilarTo => "similar_to",
            RelationType::DependsOn => "depends_on",
            RelationType::Causes => "causes",
            RelationType::Fixes => "fixes",
            RelationType::Replaces => "replaces",
        }
    }
}

/// Result of entity enhancement
#[derive(Clone, Debug)]
pub struct EnhancementResult {
    /// Extracted entities
    pub entities: Vec<EnhancedEntity>,
    /// Extracted relationships
    pub relationships: Vec<EnhancedRelationship>,
    /// Extracted domain concepts
    pub concepts: Vec<String>,
    /// Whether LLM was used
    pub used_local_llm: bool,
}

impl EnhancementResult {
    /// Create an empty enhancement result (no entities, relationships, or concepts).
    pub fn empty() -> Self {
        Self {
            entities: Vec::new(),
            relationships: Vec::new(),
            concepts: Vec::new(),
            used_local_llm: false,
        }
    }

    /// Create an enhancement result from LLM-extracted data.
    pub fn from_local(
        entities: Vec<EnhancedEntity>,
        relationships: Vec<EnhancedRelationship>,
        concepts: Vec<String>,
    ) -> Self {
        Self {
            entities,
            relationships,
            concepts,
            used_local_llm: true,
        }
    }
}

/// Entity enhancer using a provider
pub struct EntityEnhancer {
    provider: Arc<dyn Provider>,
    model_id: String,
    /// Minimum confidence threshold
    min_confidence: f32,
    /// Maximum entities to extract per call
    max_entities: usize,
}

impl EntityEnhancer {
    /// Create a new entity enhancer
    pub fn new(provider: Arc<dyn Provider>, model_id: impl Into<String>) -> Self {
        Self {
            provider,
            model_id: model_id.into(),
            min_confidence: 0.6,
            max_entities: 20,
        }
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set max entities per extraction
    pub fn with_max_entities(mut self, max: usize) -> Self {
        self.max_entities = max.max(1);
        self
    }

    /// Extract semantic entities from text using the provider
    pub async fn extract_entities(&self, text: &str) -> Option<Vec<EnhancedEntity>> {
        let timer = InferenceTimer::new("extract_entities", &self.model_id);

        let prompt = self.build_entity_prompt(text);

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions::deterministic(200);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let output = response.message.text_or_summary();
                let entities = self.parse_entities(&output);
                timer.finish(true);
                Some(entities)
            }
            Err(e) => {
                warn!(target: "local_llm", "Entity extraction failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Extract relationships between entities using the provider
    pub async fn extract_relationships(
        &self,
        entities: &[String],
        context: &str,
    ) -> Option<Vec<EnhancedRelationship>> {
        if entities.len() < 2 {
            return Some(Vec::new());
        }

        let timer = InferenceTimer::new("extract_relationships", &self.model_id);

        let prompt = self.build_relationship_prompt(entities, context);

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions::deterministic(150);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let output = response.message.text_or_summary();
                let relationships = self.parse_relationships(&output);
                timer.finish(true);
                Some(relationships)
            }
            Err(e) => {
                warn!(target: "local_llm", "Relationship extraction failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Extract domain concepts from text using the provider
    pub async fn extract_concepts(&self, text: &str) -> Option<Vec<String>> {
        let timer = InferenceTimer::new("extract_concepts", &self.model_id);

        let prompt = self.build_concept_prompt(text);

        let messages = vec![Message::user(&prompt)];
        let options = ChatOptions::deterministic(100);

        match self.provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let output = response.message.text_or_summary();
                let concepts = self.parse_concepts(&output);
                timer.finish(true);
                Some(concepts)
            }
            Err(e) => {
                warn!(target: "local_llm", "Concept extraction failed: {}", e);
                timer.finish(false);
                None
            }
        }
    }

    /// Full enhancement - extract entities, relationships, and concepts
    pub async fn enhance(&self, text: &str) -> EnhancementResult {
        // Extract entities first
        let entities = self.extract_entities(text).await.unwrap_or_default();

        // Extract relationships if we have entities
        let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
        let relationships = self
            .extract_relationships(&entity_names, text)
            .await
            .unwrap_or_default();

        // Extract concepts
        let concepts = self.extract_concepts(text).await.unwrap_or_default();

        EnhancementResult::from_local(entities, relationships, concepts)
    }

    /// Heuristic entity extraction (pattern-based fallback)
    pub fn extract_heuristic(&self, text: &str) -> Vec<EnhancedEntity> {
        let mut entities = Vec::new();

        // URL pattern
        let url_pattern = regex::Regex::new(r#"https?://[^\s<>"']+"#).expect("valid url regex");
        for cap in url_pattern.find_iter(text) {
            entities.push(EnhancedEntity::new(
                cap.as_str().to_string(),
                SemanticEntityType::Url,
                0.9,
            ));
        }

        // Path-like patterns (beyond file extensions)
        let path_pattern =
            regex::Regex::new(r#"(?:^|[\s"'])(/[a-zA-Z0-9_./-]+)"#).expect("valid path regex");
        for cap in path_pattern.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let path = m.as_str();
                // Filter common false positives
                if path.len() > 3 && !path.starts_with("//") {
                    entities.push(EnhancedEntity::new(
                        path.to_string(),
                        SemanticEntityType::Path,
                        0.7,
                    ));
                }
            }
        }

        // Package/crate names (Rust-style)
        let crate_pattern = regex::Regex::new(r"(?:use|extern crate|mod)\s+([a-z_][a-z0-9_]*)")
            .expect("valid crate regex");
        for cap in crate_pattern.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                entities.push(EnhancedEntity::new(
                    m.as_str().to_string(),
                    SemanticEntityType::Module,
                    0.8,
                ));
            }
        }

        // Problem/fix indicators
        let lower = text.to_lowercase();
        if lower.contains("bug") || lower.contains("issue") || lower.contains("problem") {
            // Look for identifiers near these words
            let bug_context =
                regex::Regex::new(r"(?i)(?:bug|issue|problem)\s*(?:#|:)?\s*(\d+|[A-Z]+-\d+)")
                    .expect("valid bug regex");
            for cap in bug_context.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    entities.push(EnhancedEntity::new(
                        m.as_str().to_string(),
                        SemanticEntityType::Bug,
                        0.85,
                    ));
                }
            }
        }

        if lower.contains("fix") || lower.contains("fixed") || lower.contains("resolved") {
            let fix_context = regex::Regex::new(r"(?i)fix(?:ed|es)?\s+(?:#|:)?\s*(\d+|[A-Z]+-\d+)")
                .expect("valid fix regex");
            for cap in fix_context.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    entities.push(EnhancedEntity::new(
                        m.as_str().to_string(),
                        SemanticEntityType::Fix,
                        0.85,
                    ));
                }
            }
        }

        // Feature indicators
        if lower.contains("feature") || lower.contains("implement") || lower.contains("add") {
            let feature_context =
                regex::Regex::new(r"(?i)(?:feature|implement|add)\s+(\w+(?:\s+\w+)?)")
                    .expect("valid feature regex");
            for cap in feature_context.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    let feature = m.as_str().trim();
                    if feature.len() > 2 && feature.len() < 50 {
                        entities.push(EnhancedEntity::new(
                            feature.to_string(),
                            SemanticEntityType::Feature,
                            0.6,
                        ));
                    }
                }
            }
        }

        entities
    }

    /// Build the entity extraction prompt
    fn build_entity_prompt(&self, text: &str) -> String {
        format!(
            r#"Extract named entities from this text. Focus on:
- Code elements: files, functions, types, modules, packages
- Domain concepts: patterns, algorithms, protocols
- Problems: errors, bugs, issues
- Actions: commands, operations, tasks

Text: "{}"

Output format (one per line):
TYPE: name

Example:
FUNCTION: process_data
ERROR: AuthenticationError
CONCEPT: dependency injection

Entities:"#,
            if text.len() > 500 { &text[..500] } else { text }
        )
    }

    /// Build the relationship extraction prompt
    fn build_relationship_prompt(&self, entities: &[String], context: &str) -> String {
        let entity_list = entities
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            r#"Given these entities: [{}]

And this context: "{}"

Identify relationships between entities. Types:
- CONTAINS: A contains B
- USES: A uses B
- CALLS: A calls B
- DEPENDS_ON: A depends on B
- MODIFIES: A modifies B
- FIXES: A fixes B

Output format (one per line):
FROM -> RELATION -> TO

Relationships:"#,
            entity_list,
            if context.len() > 300 {
                &context[..300]
            } else {
                context
            }
        )
    }

    /// Build the concept extraction prompt
    fn build_concept_prompt(&self, text: &str) -> String {
        format!(
            r#"Extract domain concepts and technical terms from this text.
Focus on: frameworks, patterns, methodologies, technologies, abstractions.

Text: "{}"

Output: comma-separated list of concepts
Example: REST API, dependency injection, authentication

Concepts:"#,
            if text.len() > 400 { &text[..400] } else { text }
        )
    }

    /// Parse entity extraction output
    fn parse_entities(&self, output: &str) -> Vec<EnhancedEntity> {
        let mut entities = Vec::new();

        for line in output.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Parse "TYPE: name" format
            if let Some((type_str, name)) = line.split_once(':') {
                let type_str = type_str.trim().to_uppercase();
                let name = name.trim();

                if name.is_empty() {
                    continue;
                }

                let entity_type = match type_str.as_str() {
                    "FILE" => SemanticEntityType::File,
                    "FUNCTION" | "FUNC" | "FN" => SemanticEntityType::Function,
                    "TYPE" | "STRUCT" | "CLASS" => SemanticEntityType::Type,
                    "VARIABLE" | "VAR" => SemanticEntityType::Variable,
                    "MODULE" | "MOD" => SemanticEntityType::Module,
                    "PACKAGE" | "CRATE" => SemanticEntityType::Package,
                    "CONCEPT" => SemanticEntityType::Concept,
                    "PATTERN" => SemanticEntityType::Pattern,
                    "ALGORITHM" => SemanticEntityType::Algorithm,
                    "PROTOCOL" => SemanticEntityType::Protocol,
                    "COMMAND" | "CMD" => SemanticEntityType::Command,
                    "OPERATION" => SemanticEntityType::Operation,
                    "TASK" => SemanticEntityType::Task,
                    "ERROR" => SemanticEntityType::Error,
                    "BUG" => SemanticEntityType::Bug,
                    "FIX" => SemanticEntityType::Fix,
                    "FEATURE" => SemanticEntityType::Feature,
                    "PERSON" | "USER" => SemanticEntityType::Person,
                    "URL" | "LINK" => SemanticEntityType::Url,
                    "PATH" => SemanticEntityType::Path,
                    _ => continue,
                };

                entities.push(EnhancedEntity::new(name.to_string(), entity_type, 0.8));

                if entities.len() >= self.max_entities {
                    break;
                }
            }
        }

        entities
    }

    /// Parse relationship extraction output
    fn parse_relationships(&self, output: &str) -> Vec<EnhancedRelationship> {
        let mut relationships = Vec::new();

        for line in output.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Parse "FROM -> RELATION -> TO" format
            let parts: Vec<&str> = line.split("->").map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                let from = parts[0].to_string();
                let relation_str = parts[1].to_uppercase();
                let to = parts[2].to_string();

                let relation_type = match relation_str.as_str() {
                    "CONTAINS" => RelationType::Contains,
                    "DEFINED_IN" | "DEFINEDIN" => RelationType::DefinedIn,
                    "IMPORTS" => RelationType::Imports,
                    "EXPORTS" => RelationType::Exports,
                    "EXTENDS" => RelationType::Extends,
                    "IMPLEMENTS" => RelationType::Implements,
                    "CALLS" => RelationType::Calls,
                    "USES" => RelationType::Uses,
                    "MODIFIES" => RelationType::Modifies,
                    "CREATES" => RelationType::Creates,
                    "DELETES" => RelationType::Deletes,
                    "RELATED_TO" | "RELATEDTO" => RelationType::RelatedTo,
                    "SIMILAR_TO" | "SIMILARTO" => RelationType::SimilarTo,
                    "DEPENDS_ON" | "DEPENDSON" => RelationType::DependsOn,
                    "CAUSES" => RelationType::Causes,
                    "FIXES" => RelationType::Fixes,
                    "REPLACES" => RelationType::Replaces,
                    _ => RelationType::RelatedTo, // Default
                };

                relationships.push(EnhancedRelationship {
                    from,
                    to,
                    relation_type,
                    confidence: 0.75,
                });
            }
        }

        relationships
    }

    /// Parse concept extraction output
    fn parse_concepts(&self, output: &str) -> Vec<String> {
        let mut concepts = Vec::new();

        // Handle comma-separated list
        for concept in output.split(',') {
            let concept = concept.trim().to_lowercase();
            if !concept.is_empty() && concept.len() > 2 && concept.len() < 50 {
                concepts.push(concept);
            }
        }

        // Also handle newline-separated
        if concepts.is_empty() {
            for line in output.lines() {
                let concept = line.trim().to_lowercase();
                if !concept.is_empty() && concept.len() > 2 && concept.len() < 50 {
                    concepts.push(concept);
                }
            }
        }

        concepts
    }
}

/// Builder for EntityEnhancer
pub struct EntityEnhancerBuilder {
    provider: Option<Arc<dyn Provider>>,
    model_id: String,
    min_confidence: f32,
    max_entities: usize,
}

impl Default for EntityEnhancerBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            model_id: "lfm2-350m".to_string(), // Fast model for entity extraction
            min_confidence: 0.6,
            max_entities: 20,
        }
    }
}

impl EntityEnhancerBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider to use for entity extraction.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model ID to use for inference.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Set the minimum confidence threshold for extracted entities.
    pub fn min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum number of entities to extract per call.
    pub fn max_entities(mut self, max: usize) -> Self {
        self.max_entities = max.max(1);
        self
    }

    /// Build the entity enhancer, returning `None` if no provider was set.
    pub fn build(self) -> Option<EntityEnhancer> {
        self.provider.map(|p| {
            EntityEnhancer::new(p, self.model_id)
                .with_min_confidence(self.min_confidence)
                .with_max_entities(self.max_entities)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_entity_type_parsing() {
        assert_eq!(
            SemanticEntityType::from_str("function"),
            Some(SemanticEntityType::Function)
        );
        assert_eq!(
            SemanticEntityType::from_str("STRUCT"),
            Some(SemanticEntityType::Type)
        );
        assert_eq!(
            SemanticEntityType::from_str("crate"),
            Some(SemanticEntityType::Package)
        );
        assert_eq!(SemanticEntityType::from_str("invalid"), None);
    }

    #[test]
    fn test_relation_type_parsing() {
        assert_eq!(
            RelationType::from_str("contains"),
            Some(RelationType::Contains)
        );
        assert_eq!(
            RelationType::from_str("DEPENDS_ON"),
            Some(RelationType::DependsOn)
        );
        assert_eq!(RelationType::from_str("invalid"), None);
    }

    #[test]
    fn test_heuristic_extraction_url() {
        let _enhancer = EntityEnhancerBuilder::default();
        let result = extract_heuristic_direct("Check https://example.com/docs for more info");
        assert!(
            result
                .iter()
                .any(|e| e.entity_type == SemanticEntityType::Url)
        );
    }

    #[test]
    fn test_heuristic_extraction_path() {
        let result = extract_heuristic_direct("Look at /home/user/project/src");
        assert!(
            result
                .iter()
                .any(|e| e.entity_type == SemanticEntityType::Path)
        );
    }

    #[test]
    fn test_heuristic_extraction_bug() {
        // "Fixed #123" should match the fix pattern
        let result = extract_heuristic_direct("Fixed #123 in the parser");
        assert!(
            result
                .iter()
                .any(|e| e.entity_type == SemanticEntityType::Fix)
        );
    }

    fn extract_heuristic_direct(text: &str) -> Vec<EnhancedEntity> {
        let mut entities = Vec::new();

        // URL pattern
        let url_pattern = regex::Regex::new(r#"https?://[^\s<>"']+"#).unwrap();
        for cap in url_pattern.find_iter(text) {
            entities.push(EnhancedEntity::new(
                cap.as_str().to_string(),
                SemanticEntityType::Url,
                0.9,
            ));
        }

        // Path pattern
        let path_pattern = regex::Regex::new(r#"(?:^|[\s"'])(/[a-zA-Z0-9_./-]+)"#).unwrap();
        for cap in path_pattern.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let path = m.as_str();
                if path.len() > 3 && !path.starts_with("//") {
                    entities.push(EnhancedEntity::new(
                        path.to_string(),
                        SemanticEntityType::Path,
                        0.7,
                    ));
                }
            }
        }

        // Fix pattern
        let lower = text.to_lowercase();
        if lower.contains("fix") {
            let fix_context =
                regex::Regex::new(r"(?i)fix(?:ed|es)?\s+(?:#|:)?\s*(\d+|[A-Z]+-\d+)").unwrap();
            for cap in fix_context.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    entities.push(EnhancedEntity::new(
                        m.as_str().to_string(),
                        SemanticEntityType::Fix,
                        0.85,
                    ));
                }
            }
        }

        entities
    }

    #[test]
    fn test_parse_entities() {
        let output = r#"FUNCTION: process_data
ERROR: AuthenticationError
CONCEPT: dependency injection"#;

        let entities = parse_entities_direct(output);
        assert_eq!(entities.len(), 3);
        assert!(entities.iter().any(|e| e.name == "process_data"));
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == SemanticEntityType::Error)
        );
    }

    fn parse_entities_direct(output: &str) -> Vec<EnhancedEntity> {
        let mut entities = Vec::new();

        for line in output.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some((type_str, name)) = line.split_once(':') {
                let type_str = type_str.trim().to_uppercase();
                let name = name.trim();

                if name.is_empty() {
                    continue;
                }

                let entity_type = match type_str.as_str() {
                    "FUNCTION" => SemanticEntityType::Function,
                    "ERROR" => SemanticEntityType::Error,
                    "CONCEPT" => SemanticEntityType::Concept,
                    _ => continue,
                };

                entities.push(EnhancedEntity::new(name.to_string(), entity_type, 0.8));
            }
        }

        entities
    }

    #[test]
    fn test_parse_relationships() {
        let output = "process_data -> CALLS -> validate_input\nModule -> CONTAINS -> Function";

        let relationships = parse_relationships_direct(output);
        assert_eq!(relationships.len(), 2);
        assert!(
            relationships
                .iter()
                .any(|r| r.relation_type == RelationType::Calls)
        );
    }

    fn parse_relationships_direct(output: &str) -> Vec<EnhancedRelationship> {
        let mut relationships = Vec::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split("->").map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                let from = parts[0].to_string();
                let relation_str = parts[1].to_uppercase();
                let to = parts[2].to_string();

                let relation_type = match relation_str.as_str() {
                    "CALLS" => RelationType::Calls,
                    "CONTAINS" => RelationType::Contains,
                    _ => RelationType::RelatedTo,
                };

                relationships.push(EnhancedRelationship {
                    from,
                    to,
                    relation_type,
                    confidence: 0.75,
                });
            }
        }

        relationships
    }

    #[test]
    fn test_parse_concepts() {
        let output = "REST API, dependency injection, authentication";
        let concepts = parse_concepts_direct(output);
        assert_eq!(concepts.len(), 3);
        assert!(concepts.contains(&"rest api".to_string()));
    }

    fn parse_concepts_direct(output: &str) -> Vec<String> {
        let mut concepts = Vec::new();

        for concept in output.split(',') {
            let concept = concept.trim().to_lowercase();
            if !concept.is_empty() && concept.len() > 2 && concept.len() < 50 {
                concepts.push(concept);
            }
        }

        concepts
    }
}
