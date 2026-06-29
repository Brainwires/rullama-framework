//! Prompting Technique Definitions
//!
//! This module defines the 15 prompting techniques from the paper
//! "Adaptive Selection of Prompting Techniques" (arXiv:2510.18162),
//! with SEAL quality integration for intelligent technique filtering.

use serde::{Deserialize, Serialize};

/// Prompting technique categories from the paper
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TechniqueCategory {
    /// Role assignment (e.g., "You are an expert...")
    RoleAssignment,
    /// Emotional stimulus (e.g., "This is important...")
    EmotionalStimulus,
    /// Reasoning techniques (e.g., Chain-of-Thought)
    Reasoning,
    /// Supporting techniques (e.g., Scratchpad)
    Others,
}

/// 15 prompting techniques from the paper (Table 1)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PromptingTechnique {
    /// Role-playing persona assignment.
    RolePlaying,

    /// Emotion-based prompting stimulus.
    EmotionPrompting,
    /// Stress / urgency-based prompting stimulus.
    StressPrompting,

    /// Chain-of-thought reasoning.
    ChainOfThought,
    /// Logic-of-thought structured reasoning.
    LogicOfThought,
    /// Least-to-most decomposition.
    LeastToMost,
    /// Thread-of-thought sequential reasoning.
    ThreadOfThought,
    /// Plan-and-solve two-stage reasoning.
    PlanAndSolve,
    /// Skeleton-of-thought parallel generation.
    SkeletonOfThought,
    /// Scratchpad-based working memory.
    ScratchpadPrompting,

    /// Decomposed multi-step prompting.
    DecomposedPrompting,
    /// Ignore irrelevant conditions filtering.
    IgnoreIrrelevantConditions,
    /// Highlighted chain-of-thought.
    HighlightedCoT,
    /// Skills-in-context grounding.
    SkillsInContext,
    /// Automatic information filtering.
    AutomaticInformationFiltering,
}

impl PromptingTechnique {
    /// Parse technique from string ID
    pub fn parse_id(s: &str) -> Result<Self, &'static str> {
        match s.to_lowercase().as_str() {
            "role_playing" | "roleplaying" => Ok(Self::RolePlaying),
            "emotion_prompting" | "emotionprompting" => Ok(Self::EmotionPrompting),
            "stress_prompting" | "stressprompting" => Ok(Self::StressPrompting),
            "chain_of_thought" | "chainofthought" | "cot" => Ok(Self::ChainOfThought),
            "logic_of_thought" | "logicofthought" | "lot" => Ok(Self::LogicOfThought),
            "least_to_most" | "leasttomost" => Ok(Self::LeastToMost),
            "thread_of_thought" | "threadofthought" | "tot" => Ok(Self::ThreadOfThought),
            "plan_and_solve" | "planandsolve" => Ok(Self::PlanAndSolve),
            "skeleton_of_thought" | "skeletonofthought" | "sot" => Ok(Self::SkeletonOfThought),
            "scratchpad_prompting" | "scratchpadprompting" | "scratchpad" => {
                Ok(Self::ScratchpadPrompting)
            }
            "decomposed_prompting" | "decomposedprompting" => Ok(Self::DecomposedPrompting),
            "ignore_irrelevant_conditions" | "ignoreirrelevantconditions" => {
                Ok(Self::IgnoreIrrelevantConditions)
            }
            "highlighted_cot" | "highlightedcot" => Ok(Self::HighlightedCoT),
            "skills_in_context" | "skillsincontext" => Ok(Self::SkillsInContext),
            "automatic_information_filtering" | "automaticinformationfiltering" => {
                Ok(Self::AutomaticInformationFiltering)
            }
            _ => Err("Unknown technique"),
        }
    }

    /// Convert technique to string ID
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::RolePlaying => "role_playing",
            Self::EmotionPrompting => "emotion_prompting",
            Self::StressPrompting => "stress_prompting",
            Self::ChainOfThought => "chain_of_thought",
            Self::LogicOfThought => "logic_of_thought",
            Self::LeastToMost => "least_to_most",
            Self::ThreadOfThought => "thread_of_thought",
            Self::PlanAndSolve => "plan_and_solve",
            Self::SkeletonOfThought => "skeleton_of_thought",
            Self::ScratchpadPrompting => "scratchpad_prompting",
            Self::DecomposedPrompting => "decomposed_prompting",
            Self::IgnoreIrrelevantConditions => "ignore_irrelevant_conditions",
            Self::HighlightedCoT => "highlighted_cot",
            Self::SkillsInContext => "skills_in_context",
            Self::AutomaticInformationFiltering => "automatic_information_filtering",
        }
    }
}

/// Complexity level for SEAL quality filtering
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComplexityLevel {
    /// Use when SEAL quality < 0.5 (simple, basic techniques)
    Simple,
    /// Use when SEAL quality 0.5-0.8 (moderate complexity)
    Moderate,
    /// Use when SEAL quality > 0.8 (advanced, sophisticated techniques)
    Advanced,
}

/// Task characteristics for technique matching
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskCharacteristic {
    /// Requires multi-step reasoning chains.
    MultiStepReasoning,
    /// Involves numerical calculations.
    NumericalCalculation,
    /// Requires logical deduction.
    LogicalDeduction,
    /// Creative or open-ended generation.
    CreativeGeneration,
    /// Summarization of long context.
    LongContextSummarization,
    /// Spatial reasoning tasks.
    SpatialReasoning,
    /// Visual understanding tasks.
    VisualUnderstanding,
    /// Code generation tasks.
    CodeGeneration,
    /// Algorithmic problem solving.
    AlgorithmicProblem,
}

/// Metadata for each technique (SEAL-enhanced)
#[derive(Debug, Clone)]
pub struct TechniqueMetadata {
    /// The prompting technique this metadata describes.
    pub technique: PromptingTechnique,
    /// Category of the technique.
    pub category: TechniqueCategory,
    /// Human-readable name.
    pub name: String,
    /// Description of the technique.
    pub description: String,
    /// Template string for generating prompts.
    pub template: String,
    /// Task characteristics this technique works best for.
    pub best_for: Vec<TaskCharacteristic>,

    // SEAL integration
    /// Minimum SEAL quality to use this technique (0.0-1.0)
    pub min_seal_quality: f32,
    /// Complexity level for filtering
    pub complexity_level: ComplexityLevel,
    /// Can this technique be promoted to BKS?
    pub bks_promotion_eligible: bool,
}

impl TechniqueMetadata {
    /// Create new technique metadata
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        technique: PromptingTechnique,
        category: TechniqueCategory,
        name: impl Into<String>,
        description: impl Into<String>,
        template: impl Into<String>,
        best_for: Vec<TaskCharacteristic>,
        min_seal_quality: f32,
        complexity_level: ComplexityLevel,
        bks_promotion_eligible: bool,
    ) -> Self {
        Self {
            technique,
            category,
            name: name.into(),
            description: description.into(),
            template: template.into(),
            best_for,
            min_seal_quality,
            complexity_level,
            bks_promotion_eligible,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_technique_enum_all_variants() {
        // Ensure all 15 techniques can be created
        let techniques = vec![
            PromptingTechnique::RolePlaying,
            PromptingTechnique::EmotionPrompting,
            PromptingTechnique::StressPrompting,
            PromptingTechnique::ChainOfThought,
            PromptingTechnique::LogicOfThought,
            PromptingTechnique::LeastToMost,
            PromptingTechnique::ThreadOfThought,
            PromptingTechnique::PlanAndSolve,
            PromptingTechnique::SkeletonOfThought,
            PromptingTechnique::ScratchpadPrompting,
            PromptingTechnique::DecomposedPrompting,
            PromptingTechnique::IgnoreIrrelevantConditions,
            PromptingTechnique::HighlightedCoT,
            PromptingTechnique::SkillsInContext,
            PromptingTechnique::AutomaticInformationFiltering,
        ];
        assert_eq!(techniques.len(), 15);
    }

    #[test]
    fn test_technique_category_variants() {
        let categories = [
            TechniqueCategory::RoleAssignment,
            TechniqueCategory::EmotionalStimulus,
            TechniqueCategory::Reasoning,
            TechniqueCategory::Others,
        ];
        assert_eq!(categories.len(), 4);
    }

    #[test]
    fn test_complexity_level_ordering() {
        // Test complexity levels
        let simple = ComplexityLevel::Simple;
        let moderate = ComplexityLevel::Moderate;
        let advanced = ComplexityLevel::Advanced;

        // Ensure they can be compared
        assert_eq!(simple, ComplexityLevel::Simple);
        assert_eq!(moderate, ComplexityLevel::Moderate);
        assert_eq!(advanced, ComplexityLevel::Advanced);
    }

    #[test]
    fn test_technique_string_conversion() {
        // Test from_str and to_string
        let technique = PromptingTechnique::ChainOfThought;
        let serialized = serde_json::to_string(&technique).unwrap();
        let deserialized: PromptingTechnique = serde_json::from_str(&serialized).unwrap();
        assert_eq!(technique, deserialized);
    }

    #[test]
    fn test_technique_metadata_creation() {
        let metadata = TechniqueMetadata::new(
            PromptingTechnique::ChainOfThought,
            TechniqueCategory::Reasoning,
            "Chain-of-Thought",
            "Step-by-step reasoning",
            "Think step by step",
            vec![TaskCharacteristic::MultiStepReasoning],
            0.0,
            ComplexityLevel::Simple,
            true,
        );

        assert_eq!(metadata.technique, PromptingTechnique::ChainOfThought);
        assert_eq!(metadata.category, TechniqueCategory::Reasoning);
        assert_eq!(metadata.name, "Chain-of-Thought");
        assert_eq!(metadata.min_seal_quality, 0.0);
        assert_eq!(metadata.complexity_level, ComplexityLevel::Simple);
        assert!(metadata.bks_promotion_eligible);
    }

    #[test]
    fn test_task_characteristic_variants() {
        let characteristics = [
            TaskCharacteristic::MultiStepReasoning,
            TaskCharacteristic::NumericalCalculation,
            TaskCharacteristic::LogicalDeduction,
            TaskCharacteristic::CreativeGeneration,
            TaskCharacteristic::LongContextSummarization,
            TaskCharacteristic::SpatialReasoning,
            TaskCharacteristic::VisualUnderstanding,
        ];
        assert_eq!(characteristics.len(), 7);
    }
}
