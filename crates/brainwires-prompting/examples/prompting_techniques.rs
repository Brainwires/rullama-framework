//! Prompting Techniques Overview
//!
//! Demonstrates listing, grouping, and filtering the 15 adaptive
//! prompting techniques by category, complexity level, and task
//! characteristics.

use brainwires_prompting::techniques::{
    ComplexityLevel, PromptingTechnique, TaskCharacteristic, TechniqueCategory, TechniqueMetadata,
};
use std::collections::HashMap;

/// All 15 technique variants in definition order.
const ALL_TECHNIQUES: [PromptingTechnique; 15] = [
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

fn main() {
    // 1. Setup — build metadata for every technique
    println!("=== Prompting Techniques Library ===\n");

    let library = build_library();
    println!("Loaded {} techniques\n", library.len());

    // 2. List every technique with its metadata
    println!("=== All Techniques ===\n");
    println!(
        "{:<6} {:<34} {:<20} {:<10} {:<6}",
        "#", "Name", "Category", "Level", "SEAL"
    );
    println!("{:-<80}", "");

    for (i, tech) in ALL_TECHNIQUES.iter().enumerate() {
        let meta = &library[tech];
        println!(
            "{:<6} {:<34} {:<20} {:<10} {:<6.1}",
            i + 1,
            meta.name,
            category_label(&meta.category),
            complexity_label(&meta.complexity_level),
            meta.min_seal_quality,
        );
    }
    println!();

    // 3. Group by TechniqueCategory
    println!("=== Grouped by Category ===\n");

    let categories = [
        TechniqueCategory::RoleAssignment,
        TechniqueCategory::EmotionalStimulus,
        TechniqueCategory::Reasoning,
        TechniqueCategory::Others,
    ];

    for cat in &categories {
        let members: Vec<&TechniqueMetadata> =
            library.values().filter(|m| m.category == *cat).collect();

        println!(
            "{} ({} technique{}):",
            category_label(cat),
            members.len(),
            if members.len() == 1 { "" } else { "s" }
        );
        for m in &members {
            println!("  - {} — {}", m.name, m.description);
        }
        println!();
    }

    // 4. Show techniques suitable for each ComplexityLevel
    println!("=== By Complexity Level ===\n");

    let levels = [
        ComplexityLevel::Simple,
        ComplexityLevel::Moderate,
        ComplexityLevel::Advanced,
    ];

    for level in &levels {
        let names: Vec<&str> = library
            .values()
            .filter(|m| m.complexity_level == *level)
            .map(|m| m.name.as_str())
            .collect();

        println!(
            "{:<10} ({}): {}",
            complexity_label(level),
            names.len(),
            names.join(", ")
        );
    }
    println!();

    // 5. Show techniques best-for specific TaskCharacteristics
    println!("=== By Task Characteristic ===\n");

    let characteristics = [
        TaskCharacteristic::MultiStepReasoning,
        TaskCharacteristic::NumericalCalculation,
        TaskCharacteristic::LogicalDeduction,
        TaskCharacteristic::CreativeGeneration,
        TaskCharacteristic::LongContextSummarization,
        TaskCharacteristic::CodeGeneration,
        TaskCharacteristic::AlgorithmicProblem,
    ];

    for tc in &characteristics {
        let names: Vec<&str> = library
            .values()
            .filter(|m| m.best_for.contains(tc))
            .map(|m| m.name.as_str())
            .collect();

        println!("{:?} ({}): {}", tc, names.len(), names.join(", "));
    }
    println!();

    // 6. Comparison table — SEAL thresholds
    println!("=== SEAL Quality Threshold Table ===\n");
    println!(
        "{:<34} {:<10} {:<10} {:<6}",
        "Technique", "Level", "MinSEAL", "BKS?"
    );
    println!("{:-<64}", "");

    let mut sorted: Vec<&TechniqueMetadata> = library.values().collect();
    sorted.sort_by(|a, b| a.min_seal_quality.partial_cmp(&b.min_seal_quality).unwrap());

    for meta in &sorted {
        println!(
            "{:<34} {:<10} {:<10.1} {:<6}",
            meta.name,
            complexity_label(&meta.complexity_level),
            meta.min_seal_quality,
            if meta.bks_promotion_eligible {
                "yes"
            } else {
                "no"
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn category_label(cat: &TechniqueCategory) -> &'static str {
    match cat {
        TechniqueCategory::RoleAssignment => "Role Assignment",
        TechniqueCategory::EmotionalStimulus => "Emotional Stimulus",
        TechniqueCategory::Reasoning => "Reasoning",
        TechniqueCategory::Others => "Others",
    }
}

fn complexity_label(level: &ComplexityLevel) -> &'static str {
    match level {
        ComplexityLevel::Simple => "Simple",
        ComplexityLevel::Moderate => "Moderate",
        ComplexityLevel::Advanced => "Advanced",
    }
}

/// Build metadata for all 15 techniques (mirrors TechniqueLibrary::new()).
fn build_library() -> HashMap<PromptingTechnique, TechniqueMetadata> {
    let mut m = HashMap::new();

    m.insert(
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
            0.0,
            ComplexityLevel::Simple,
            true,
        ),
    );
    m.insert(
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
    m.insert(
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
    m.insert(
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
            0.0,
            ComplexityLevel::Simple,
            true,
        ),
    );
    m.insert(
        PromptingTechnique::LogicOfThought,
        TechniqueMetadata::new(
            PromptingTechnique::LogicOfThought,
            TechniqueCategory::Reasoning,
            "Logic-of-Thought",
            "Embed propositional logic for formal reasoning",
            "Use propositional logic notation. Apply logical inference rules. ",
            vec![
                TaskCharacteristic::LogicalDeduction,
                TaskCharacteristic::MultiStepReasoning,
            ],
            0.8,
            ComplexityLevel::Advanced,
            true,
        ),
    );
    m.insert(
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
            0.5,
            ComplexityLevel::Moderate,
            true,
        ),
    );
    m.insert(
        PromptingTechnique::ThreadOfThought,
        TechniqueMetadata::new(
            PromptingTechnique::ThreadOfThought,
            TechniqueCategory::Reasoning,
            "Thread-of-Thought",
            "Summarize long contexts progressively",
            "Summarize the context progressively as you reason through it. ",
            vec![
                TaskCharacteristic::LongContextSummarization,
                TaskCharacteristic::MultiStepReasoning,
            ],
            0.7,
            ComplexityLevel::Advanced,
            true,
        ),
    );
    m.insert(
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
            0.5,
            ComplexityLevel::Moderate,
            true,
        ),
    );
    m.insert(
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
            0.7,
            ComplexityLevel::Advanced,
            true,
        ),
    );
    m.insert(
        PromptingTechnique::ScratchpadPrompting,
        TechniqueMetadata::new(
            PromptingTechnique::ScratchpadPrompting,
            TechniqueCategory::Reasoning,
            "Scratchpad Prompting",
            "Provide draft space for intermediate steps",
            "Use the scratchpad for intermediate calculations. ",
            vec![
                TaskCharacteristic::NumericalCalculation,
                TaskCharacteristic::AlgorithmicProblem,
            ],
            0.0,
            ComplexityLevel::Simple,
            true,
        ),
    );
    m.insert(
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
    m.insert(
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
    m.insert(
        PromptingTechnique::HighlightedCoT,
        TechniqueMetadata::new(
            PromptingTechnique::HighlightedCoT,
            TechniqueCategory::Others,
            "Highlighted CoT",
            "Highlight essential information before reasoning",
            "First, highlight the essential information. Then, reason step by step. ",
            vec![
                TaskCharacteristic::MultiStepReasoning,
                TaskCharacteristic::LogicalDeduction,
            ],
            0.5,
            ComplexityLevel::Moderate,
            true,
        ),
    );
    m.insert(
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
    m.insert(
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

    m
}
