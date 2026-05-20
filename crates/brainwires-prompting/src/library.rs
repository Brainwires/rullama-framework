//! Technique Library with BKS Integration
//!
//! This module provides a library of all 15 prompting techniques with metadata,
//! integrated with the Behavioral Knowledge System (BKS) for querying shared
//! technique effectiveness across users.

use super::techniques::{
    ComplexityLevel, PromptingTechnique, TaskCharacteristic, TechniqueCategory, TechniqueMetadata,
};
use anyhow::Result;
use brainwires_knowledge::knowledge::bks_pks::{BehavioralKnowledgeCache, BehavioralTruth};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Library of all prompting techniques
pub struct TechniqueLibrary {
    techniques: HashMap<PromptingTechnique, TechniqueMetadata>,
    bks_cache: Option<Arc<Mutex<BehavioralKnowledgeCache>>>,
}

impl TechniqueLibrary {
    /// Create a new technique library with all 15 techniques
    pub fn new() -> Self {
        let mut techniques = HashMap::new();

        // === Role Assignment (1 technique) ===
        techniques.insert(
            PromptingTechnique::RolePlaying,
            TechniqueMetadata::new(
                PromptingTechnique::RolePlaying,
                TechniqueCategory::RoleAssignment,
                "Role Playing",
                "Assign expert role to elicit domain-specific knowledge",
                "You are a {role} with expertise in {domain}. ",
                vec![
                    TaskCharacteristic::MultiStepReasoning,
                    TaskCharacteristic::CodeGeneration,
                    TaskCharacteristic::AlgorithmicProblem,
                ],
                0.0, // Always usable
                ComplexityLevel::Simple,
                true,
            ),
        );

        // === Emotional Stimulus (2 techniques) ===
        techniques.insert(
            PromptingTechnique::EmotionPrompting,
            TechniqueMetadata::new(
                PromptingTechnique::EmotionPrompting,
                TechniqueCategory::EmotionalStimulus,
                "Emotion Prompting",
                "Add emotional cues to increase engagement",
                "This is an important {task_type} that requires {quality}. ",
                vec![
                    TaskCharacteristic::MultiStepReasoning,
                    TaskCharacteristic::NumericalCalculation,
                ],
                0.0,
                ComplexityLevel::Simple,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::StressPrompting,
            TechniqueMetadata::new(
                PromptingTechnique::StressPrompting,
                TechniqueCategory::EmotionalStimulus,
                "Stress Prompting",
                "Induce moderate stress conditions for focus",
                "This task requires immediate attention and precision. Time is limited. ",
                vec![
                    TaskCharacteristic::LogicalDeduction,
                    TaskCharacteristic::AlgorithmicProblem,
                ],
                0.0,
                ComplexityLevel::Simple,
                true,
            ),
        );

        // === Reasoning (7 techniques) ===
        techniques.insert(
            PromptingTechnique::ChainOfThought,
            TechniqueMetadata::new(
                PromptingTechnique::ChainOfThought,
                TechniqueCategory::Reasoning,
                "Chain-of-Thought",
                "Require explicit step-by-step reasoning",
                "Think step by step. Show your reasoning process clearly. ",
                vec![
                    TaskCharacteristic::MultiStepReasoning,
                    TaskCharacteristic::LogicalDeduction,
                    TaskCharacteristic::NumericalCalculation,
                ],
                0.0, // Always usable
                ComplexityLevel::Simple,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::LogicOfThought,
            TechniqueMetadata::new(
                PromptingTechnique::LogicOfThought,
                TechniqueCategory::Reasoning,
                "Logic-of-Thought",
                "Embed propositional logic for formal reasoning",
                "Use propositional logic notation. Let P, Q, R represent propositions. Apply logical inference rules. ",
                vec![
                    TaskCharacteristic::LogicalDeduction,
                    TaskCharacteristic::MultiStepReasoning,
                ],
                0.8, // Requires high SEAL quality
                ComplexityLevel::Advanced,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::LeastToMost,
            TechniqueMetadata::new(
                PromptingTechnique::LeastToMost,
                TechniqueCategory::Reasoning,
                "Least-to-Most",
                "Decompose into simpler sub-problems progressively",
                "Break this problem into simpler sub-problems. Solve from simplest to most complex. ",
                vec![
                    TaskCharacteristic::MultiStepReasoning,
                    TaskCharacteristic::AlgorithmicProblem,
                ],
                0.5, // Moderate complexity
                ComplexityLevel::Moderate,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::ThreadOfThought,
            TechniqueMetadata::new(
                PromptingTechnique::ThreadOfThought,
                TechniqueCategory::Reasoning,
                "Thread-of-Thought",
                "Summarize long contexts progressively",
                "Summarize the context progressively as you reason through it. Maintain a running summary. ",
                vec![
                    TaskCharacteristic::LongContextSummarization,
                    TaskCharacteristic::MultiStepReasoning,
                ],
                0.7, // Advanced technique
                ComplexityLevel::Advanced,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::PlanAndSolve,
            TechniqueMetadata::new(
                PromptingTechnique::PlanAndSolve,
                TechniqueCategory::Reasoning,
                "Plan-and-Solve",
                "Generate execution plan first, then solve step by step",
                "First, devise a plan. Then, solve the problem step by step according to the plan. ",
                vec![
                    TaskCharacteristic::MultiStepReasoning,
                    TaskCharacteristic::LogicalDeduction,
                    TaskCharacteristic::AlgorithmicProblem,
                ],
                0.5, // Moderate complexity
                ComplexityLevel::Moderate,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::SkeletonOfThought,
            TechniqueMetadata::new(
                PromptingTechnique::SkeletonOfThought,
                TechniqueCategory::Reasoning,
                "Skeleton-of-Thought",
                "Generate skeleton, then fill details",
                "First, generate a skeleton outline. Then, fill in the details for each part. ",
                vec![
                    TaskCharacteristic::CreativeGeneration,
                    TaskCharacteristic::CodeGeneration,
                ],
                0.7, // Advanced technique
                ComplexityLevel::Advanced,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::ScratchpadPrompting,
            TechniqueMetadata::new(
                PromptingTechnique::ScratchpadPrompting,
                TechniqueCategory::Reasoning,
                "Scratchpad Prompting",
                "Provide draft space for intermediate steps",
                "Use the following scratchpad format for intermediate calculations:\n\
                <scratchpad>\n\
                [Your work here]\n\
                </scratchpad>\n",
                vec![
                    TaskCharacteristic::NumericalCalculation,
                    TaskCharacteristic::AlgorithmicProblem,
                ],
                0.0,
                ComplexityLevel::Simple,
                true,
            ),
        );

        // === Others (6 techniques) ===
        techniques.insert(
            PromptingTechnique::DecomposedPrompting,
            TechniqueMetadata::new(
                PromptingTechnique::DecomposedPrompting,
                TechniqueCategory::Others,
                "Decomposed Prompting",
                "Break into sub-tasks explicitly",
                "Decompose this task into independent sub-tasks. Solve each sub-task separately. ",
                vec![
                    TaskCharacteristic::MultiStepReasoning,
                    TaskCharacteristic::AlgorithmicProblem,
                ],
                0.5,
                ComplexityLevel::Moderate,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::IgnoreIrrelevantConditions,
            TechniqueMetadata::new(
                PromptingTechnique::IgnoreIrrelevantConditions,
                TechniqueCategory::Others,
                "Ignore Irrelevant Conditions",
                "Detect and disregard noise in the problem",
                "Identify and ignore any irrelevant information. Focus only on what's essential. ",
                vec![
                    TaskCharacteristic::LogicalDeduction,
                    TaskCharacteristic::MultiStepReasoning,
                ],
                0.6,
                ComplexityLevel::Moderate,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::HighlightedCoT,
            TechniqueMetadata::new(
                PromptingTechnique::HighlightedCoT,
                TechniqueCategory::Others,
                "Highlighted CoT",
                "Highlight essential information before reasoning",
                "First, highlight the essential information. Then, reason step by step based on the highlights. ",
                vec![
                    TaskCharacteristic::MultiStepReasoning,
                    TaskCharacteristic::LogicalDeduction,
                ],
                0.5,
                ComplexityLevel::Moderate,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::SkillsInContext,
            TechniqueMetadata::new(
                PromptingTechnique::SkillsInContext,
                TechniqueCategory::Others,
                "Skills-in-Context",
                "Compose basic skills for complex tasks",
                "Identify the basic skills required. Compose them systematically to solve the task. ",
                vec![
                    TaskCharacteristic::AlgorithmicProblem,
                    TaskCharacteristic::CodeGeneration,
                ],
                0.7,
                ComplexityLevel::Advanced,
                true,
            ),
        );

        techniques.insert(
            PromptingTechnique::AutomaticInformationFiltering,
            TechniqueMetadata::new(
                PromptingTechnique::AutomaticInformationFiltering,
                TechniqueCategory::Others,
                "Automatic Information Filtering",
                "Preprocess to remove irrelevant information",
                "Filter the input to retain only relevant information before processing. ",
                vec![
                    TaskCharacteristic::LongContextSummarization,
                    TaskCharacteristic::LogicalDeduction,
                ],
                0.6,
                ComplexityLevel::Moderate,
                true,
            ),
        );

        Self {
            techniques,
            bks_cache: None,
        }
    }

    /// Set BKS cache for querying shared technique effectiveness
    pub fn with_bks(mut self, bks_cache: Arc<Mutex<BehavioralKnowledgeCache>>) -> Self {
        self.bks_cache = Some(bks_cache);
        self
    }

    /// Get technique metadata by enum
    pub fn get(&self, technique: &PromptingTechnique) -> Option<&TechniqueMetadata> {
        self.techniques.get(technique)
    }

    /// Get all techniques
    pub fn get_all(&self) -> Vec<&TechniqueMetadata> {
        self.techniques.values().collect()
    }

    /// Get techniques filtered by SEAL quality score
    pub fn get_by_seal_quality(&self, seal_quality: f32) -> Vec<&TechniqueMetadata> {
        self.techniques
            .values()
            .filter(|t| t.min_seal_quality <= seal_quality)
            .collect()
    }

    /// Get techniques by category
    pub fn get_by_category(&self, category: TechniqueCategory) -> Vec<&TechniqueMetadata> {
        self.techniques
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Query BKS for technique effectiveness in a specific cluster
    ///
    /// Returns techniques that have been successfully promoted to BKS for this cluster,
    /// indicating they work well based on collective user experience.
    pub async fn get_bks_recommended_techniques(
        &self,
        cluster_id: &str,
    ) -> Result<Vec<PromptingTechnique>> {
        if let Some(ref bks_cache) = self.bks_cache {
            let bks = bks_cache.lock().await;

            // Query BKS for truths matching the cluster context
            // Example: "For numerical_reasoning, ChainOfThought achieves 92% success"
            let truths = bks.get_matching_truths(cluster_id);

            let mut recommended = Vec::new();
            for truth in truths {
                // Parse technique from truth rule/rationale
                if let Some(technique) = self.parse_technique_from_truth(truth) {
                    recommended.push(technique);
                }
            }

            Ok(recommended)
        } else {
            Ok(Vec::new())
        }
    }

    /// Parse technique enum from BKS truth rule/rationale
    fn parse_technique_from_truth(&self, truth: &BehavioralTruth) -> Option<PromptingTechnique> {
        // Example rule: "Use ChainOfThought for numerical reasoning"
        // Example rationale: "ChainOfThought achieves 92% success rate"
        let text = format!("{} {}", truth.rule, truth.rationale);

        for (technique_enum, metadata) in &self.techniques {
            // Check if technique name appears in the truth text
            if text.contains(&metadata.name) || text.contains(technique_enum.to_str()) {
                return Some(technique_enum.clone());
            }
        }

        None
    }

    /// Get count of techniques by complexity level
    pub fn count_by_complexity(&self, level: ComplexityLevel) -> usize {
        self.techniques
            .values()
            .filter(|t| t.complexity_level == level)
            .count()
    }
}

impl Default for TechniqueLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_contains_all_15_techniques() {
        let library = TechniqueLibrary::new();
        assert_eq!(library.techniques.len(), 15);
    }

    #[test]
    fn test_library_categories() {
        let library = TechniqueLibrary::new();

        assert_eq!(
            library
                .get_by_category(TechniqueCategory::RoleAssignment)
                .len(),
            1
        );
        assert_eq!(
            library
                .get_by_category(TechniqueCategory::EmotionalStimulus)
                .len(),
            2
        );
        assert_eq!(
            library.get_by_category(TechniqueCategory::Reasoning).len(),
            7
        );
        assert_eq!(library.get_by_category(TechniqueCategory::Others).len(), 5);
    }

    #[test]
    fn test_seal_quality_filtering() {
        let library = TechniqueLibrary::new();

        // Low quality (0.3) → only simple techniques
        let low_quality = library.get_by_seal_quality(0.3);
        assert!(low_quality.iter().all(|t| t.min_seal_quality <= 0.3));

        // High quality (0.9) → all techniques available
        let high_quality = library.get_by_seal_quality(0.9);
        assert_eq!(high_quality.len(), 15); // All techniques pass
    }

    #[test]
    fn test_technique_string_conversion() {
        assert_eq!(
            PromptingTechnique::parse_id("chain_of_thought").unwrap(),
            PromptingTechnique::ChainOfThought
        );
        assert_eq!(
            PromptingTechnique::parse_id("CoT").unwrap(),
            PromptingTechnique::ChainOfThought
        );
        assert_eq!(
            PromptingTechnique::ChainOfThought.to_str(),
            "chain_of_thought"
        );
    }
}
