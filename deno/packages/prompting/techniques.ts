/**
 * Definitions of the 15 prompting techniques from "Adaptive Selection of
 * Prompting Techniques" (arXiv:2510.18162). Provides the `PromptingTechnique`,
 * `TechniqueCategory`, `ComplexityLevel` and `TaskCharacteristic` string unions,
 * the `TECHNIQUE_METADATA` table describing each technique, and the lookup and
 * filter helpers (`getTechniqueMetadata`, `getTechniquesByCategory`,
 * `getTechniquesByComplexity`, `getTechniquesBySealQuality`, `parseTechniqueId`).
 * Equivalent to Rust's `rullama_prompting::techniques`.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Enums (represented as string unions for TypeScript ergonomics)
// ---------------------------------------------------------------------------

/** Prompting technique categories from the paper. */
export type TechniqueCategory =
  | "RoleAssignment"
  | "EmotionalStimulus"
  | "Reasoning"
  | "Others";

/** 15 prompting techniques from the paper (Table 1). */
export type PromptingTechnique =
  | "RolePlaying"
  | "EmotionPrompting"
  | "StressPrompting"
  | "ChainOfThought"
  | "LogicOfThought"
  | "LeastToMost"
  | "ThreadOfThought"
  | "PlanAndSolve"
  | "SkeletonOfThought"
  | "ScratchpadPrompting"
  | "DecomposedPrompting"
  | "IgnoreIrrelevantConditions"
  | "HighlightedCoT"
  | "SkillsInContext"
  | "AutomaticInformationFiltering";

/** Complexity level for SEAL quality filtering. */
export type ComplexityLevel =
  | "Simple"
  | "Moderate"
  | "Advanced";

/** Task characteristics for technique matching. */
export type TaskCharacteristic =
  | "MultiStepReasoning"
  | "NumericalCalculation"
  | "LogicalDeduction"
  | "CreativeGeneration"
  | "LongContextSummarization"
  | "SpatialReasoning"
  | "VisualUnderstanding"
  | "CodeGeneration"
  | "AlgorithmicProblem";

// ---------------------------------------------------------------------------
// All enum values (for iteration)
// ---------------------------------------------------------------------------

/** All 15 prompting techniques. */
export const ALL_TECHNIQUES: readonly PromptingTechnique[] = [
  "RolePlaying",
  "EmotionPrompting",
  "StressPrompting",
  "ChainOfThought",
  "LogicOfThought",
  "LeastToMost",
  "ThreadOfThought",
  "PlanAndSolve",
  "SkeletonOfThought",
  "ScratchpadPrompting",
  "DecomposedPrompting",
  "IgnoreIrrelevantConditions",
  "HighlightedCoT",
  "SkillsInContext",
  "AutomaticInformationFiltering",
] as const;

/** All technique categories. */
export const ALL_CATEGORIES: readonly TechniqueCategory[] = [
  "RoleAssignment",
  "EmotionalStimulus",
  "Reasoning",
  "Others",
] as const;

/** All complexity levels. */
export const ALL_COMPLEXITY_LEVELS: readonly ComplexityLevel[] = [
  "Simple",
  "Moderate",
  "Advanced",
] as const;

/** All task characteristics. */
export const ALL_TASK_CHARACTERISTICS: readonly TaskCharacteristic[] = [
  "MultiStepReasoning",
  "NumericalCalculation",
  "LogicalDeduction",
  "CreativeGeneration",
  "LongContextSummarization",
  "SpatialReasoning",
  "VisualUnderstanding",
  "CodeGeneration",
  "AlgorithmicProblem",
] as const;

// ---------------------------------------------------------------------------
// Technique ID string conversion (snake_case <-> enum)
// ---------------------------------------------------------------------------

const TECHNIQUE_TO_ID: Record<PromptingTechnique, string> = {
  RolePlaying: "role_playing",
  EmotionPrompting: "emotion_prompting",
  StressPrompting: "stress_prompting",
  ChainOfThought: "chain_of_thought",
  LogicOfThought: "logic_of_thought",
  LeastToMost: "least_to_most",
  ThreadOfThought: "thread_of_thought",
  PlanAndSolve: "plan_and_solve",
  SkeletonOfThought: "skeleton_of_thought",
  ScratchpadPrompting: "scratchpad_prompting",
  DecomposedPrompting: "decomposed_prompting",
  IgnoreIrrelevantConditions: "ignore_irrelevant_conditions",
  HighlightedCoT: "highlighted_cot",
  SkillsInContext: "skills_in_context",
  AutomaticInformationFiltering: "automatic_information_filtering",
};

const ID_TO_TECHNIQUE: Record<string, PromptingTechnique> = {};
for (const [technique, id] of Object.entries(TECHNIQUE_TO_ID)) {
  ID_TO_TECHNIQUE[id] = technique as PromptingTechnique;
  // Also map the camelCase (lowered) variant
  ID_TO_TECHNIQUE[technique.toLowerCase()] = technique as PromptingTechnique;
}
// Common abbreviations
ID_TO_TECHNIQUE["cot"] = "ChainOfThought";
ID_TO_TECHNIQUE["lot"] = "LogicOfThought";
ID_TO_TECHNIQUE["tot"] = "ThreadOfThought";
ID_TO_TECHNIQUE["sot"] = "SkeletonOfThought";
ID_TO_TECHNIQUE["scratchpad"] = "ScratchpadPrompting";

/** Convert a PromptingTechnique to its snake_case string ID. */
export function techniqueToId(technique: PromptingTechnique): string {
  return TECHNIQUE_TO_ID[technique];
}

/** Parse a string ID into a PromptingTechnique, or return undefined. */
export function parseTechniqueId(
  s: string,
): PromptingTechnique | undefined {
  return ID_TO_TECHNIQUE[s.toLowerCase()];
}

// ---------------------------------------------------------------------------
// Technique metadata
// ---------------------------------------------------------------------------

/** Metadata for each technique (SEAL-enhanced). */
export interface TechniqueMetadata {
  /** The prompting technique this metadata describes. */
  readonly technique: PromptingTechnique;
  /** Category of the technique. */
  readonly category: TechniqueCategory;
  /** Human-readable name. */
  readonly name: string;
  /** Description of the technique. */
  readonly description: string;
  /** Template string for generating prompts. */
  readonly template: string;
  /** Task characteristics this technique works best for. */
  readonly bestFor: readonly TaskCharacteristic[];
  /** Minimum SEAL quality to use this technique (0.0-1.0). */
  readonly minSealQuality: number;
  /** Complexity level for filtering. */
  readonly complexityLevel: ComplexityLevel;
  /** Can this technique be promoted to BKS? */
  readonly bksPromotionEligible: boolean;
}

// ---------------------------------------------------------------------------
// All 15 technique definitions (from TechniqueLibrary::new in Rust)
// ---------------------------------------------------------------------------

/** Complete registry of all 15 technique metadata definitions. */
export const TECHNIQUE_METADATA: ReadonlyMap<
  PromptingTechnique,
  TechniqueMetadata
> = new Map<PromptingTechnique, TechniqueMetadata>([
  // === Role Assignment (1 technique) ===
  [
    "RolePlaying",
    {
      technique: "RolePlaying",
      category: "RoleAssignment",
      name: "Role Playing",
      description: "Assign expert role to elicit domain-specific knowledge",
      template: "You are a {role} with expertise in {domain}. ",
      bestFor: ["MultiStepReasoning", "CodeGeneration", "AlgorithmicProblem"],
      minSealQuality: 0.0,
      complexityLevel: "Simple",
      bksPromotionEligible: true,
    },
  ],
  // === Emotional Stimulus (2 techniques) ===
  [
    "EmotionPrompting",
    {
      technique: "EmotionPrompting",
      category: "EmotionalStimulus",
      name: "Emotion Prompting",
      description: "Add emotional cues to increase engagement",
      template: "This is an important {task_type} that requires {quality}. ",
      bestFor: ["MultiStepReasoning", "NumericalCalculation"],
      minSealQuality: 0.0,
      complexityLevel: "Simple",
      bksPromotionEligible: true,
    },
  ],
  [
    "StressPrompting",
    {
      technique: "StressPrompting",
      category: "EmotionalStimulus",
      name: "Stress Prompting",
      description: "Induce moderate stress conditions for focus",
      template:
        "This task requires immediate attention and precision. Time is limited. ",
      bestFor: ["LogicalDeduction", "AlgorithmicProblem"],
      minSealQuality: 0.0,
      complexityLevel: "Simple",
      bksPromotionEligible: true,
    },
  ],
  // === Reasoning (7 techniques) ===
  [
    "ChainOfThought",
    {
      technique: "ChainOfThought",
      category: "Reasoning",
      name: "Chain-of-Thought",
      description: "Require explicit step-by-step reasoning",
      template: "Think step by step. Show your reasoning process clearly. ",
      bestFor: [
        "MultiStepReasoning",
        "LogicalDeduction",
        "NumericalCalculation",
      ],
      minSealQuality: 0.0,
      complexityLevel: "Simple",
      bksPromotionEligible: true,
    },
  ],
  [
    "LogicOfThought",
    {
      technique: "LogicOfThought",
      category: "Reasoning",
      name: "Logic-of-Thought",
      description: "Embed propositional logic for formal reasoning",
      template:
        "Use propositional logic notation. Let P, Q, R represent propositions. Apply logical inference rules. ",
      bestFor: ["LogicalDeduction", "MultiStepReasoning"],
      minSealQuality: 0.8,
      complexityLevel: "Advanced",
      bksPromotionEligible: true,
    },
  ],
  [
    "LeastToMost",
    {
      technique: "LeastToMost",
      category: "Reasoning",
      name: "Least-to-Most",
      description: "Decompose into simpler sub-problems progressively",
      template:
        "Break this problem into simpler sub-problems. Solve from simplest to most complex. ",
      bestFor: ["MultiStepReasoning", "AlgorithmicProblem"],
      minSealQuality: 0.5,
      complexityLevel: "Moderate",
      bksPromotionEligible: true,
    },
  ],
  [
    "ThreadOfThought",
    {
      technique: "ThreadOfThought",
      category: "Reasoning",
      name: "Thread-of-Thought",
      description: "Summarize long contexts progressively",
      template:
        "Summarize the context progressively as you reason through it. Maintain a running summary. ",
      bestFor: ["LongContextSummarization", "MultiStepReasoning"],
      minSealQuality: 0.7,
      complexityLevel: "Advanced",
      bksPromotionEligible: true,
    },
  ],
  [
    "PlanAndSolve",
    {
      technique: "PlanAndSolve",
      category: "Reasoning",
      name: "Plan-and-Solve",
      description: "Generate execution plan first, then solve step by step",
      template:
        "First, devise a plan. Then, solve the problem step by step according to the plan. ",
      bestFor: [
        "MultiStepReasoning",
        "LogicalDeduction",
        "AlgorithmicProblem",
      ],
      minSealQuality: 0.5,
      complexityLevel: "Moderate",
      bksPromotionEligible: true,
    },
  ],
  [
    "SkeletonOfThought",
    {
      technique: "SkeletonOfThought",
      category: "Reasoning",
      name: "Skeleton-of-Thought",
      description: "Generate skeleton, then fill details",
      template:
        "First, generate a skeleton outline. Then, fill in the details for each part. ",
      bestFor: ["CreativeGeneration", "CodeGeneration"],
      minSealQuality: 0.7,
      complexityLevel: "Advanced",
      bksPromotionEligible: true,
    },
  ],
  [
    "ScratchpadPrompting",
    {
      technique: "ScratchpadPrompting",
      category: "Reasoning",
      name: "Scratchpad Prompting",
      description: "Provide draft space for intermediate steps",
      template:
        "Use the following scratchpad format for intermediate calculations:\n<scratchpad>\n[Your work here]\n</scratchpad>\n",
      bestFor: ["NumericalCalculation", "AlgorithmicProblem"],
      minSealQuality: 0.0,
      complexityLevel: "Simple",
      bksPromotionEligible: true,
    },
  ],
  // === Others (5 techniques) ===
  [
    "DecomposedPrompting",
    {
      technique: "DecomposedPrompting",
      category: "Others",
      name: "Decomposed Prompting",
      description: "Break into sub-tasks explicitly",
      template:
        "Decompose this task into independent sub-tasks. Solve each sub-task separately. ",
      bestFor: ["MultiStepReasoning", "AlgorithmicProblem"],
      minSealQuality: 0.5,
      complexityLevel: "Moderate",
      bksPromotionEligible: true,
    },
  ],
  [
    "IgnoreIrrelevantConditions",
    {
      technique: "IgnoreIrrelevantConditions",
      category: "Others",
      name: "Ignore Irrelevant Conditions",
      description: "Detect and disregard noise in the problem",
      template:
        "Identify and ignore any irrelevant information. Focus only on what's essential. ",
      bestFor: ["LogicalDeduction", "MultiStepReasoning"],
      minSealQuality: 0.6,
      complexityLevel: "Moderate",
      bksPromotionEligible: true,
    },
  ],
  [
    "HighlightedCoT",
    {
      technique: "HighlightedCoT",
      category: "Others",
      name: "Highlighted CoT",
      description: "Highlight essential information before reasoning",
      template:
        "First, highlight the essential information. Then, reason step by step based on the highlights. ",
      bestFor: ["MultiStepReasoning", "LogicalDeduction"],
      minSealQuality: 0.5,
      complexityLevel: "Moderate",
      bksPromotionEligible: true,
    },
  ],
  [
    "SkillsInContext",
    {
      technique: "SkillsInContext",
      category: "Others",
      name: "Skills-in-Context",
      description: "Compose basic skills for complex tasks",
      template:
        "Identify the basic skills required. Compose them systematically to solve the task. ",
      bestFor: ["AlgorithmicProblem", "CodeGeneration"],
      minSealQuality: 0.7,
      complexityLevel: "Advanced",
      bksPromotionEligible: true,
    },
  ],
  [
    "AutomaticInformationFiltering",
    {
      technique: "AutomaticInformationFiltering",
      category: "Others",
      name: "Automatic Information Filtering",
      description: "Preprocess to remove irrelevant information",
      template:
        "Filter the input to retain only relevant information before processing. ",
      bestFor: ["LongContextSummarization", "LogicalDeduction"],
      minSealQuality: 0.6,
      complexityLevel: "Moderate",
      bksPromotionEligible: true,
    },
  ],
]);

// ---------------------------------------------------------------------------
// Convenience accessors
// ---------------------------------------------------------------------------

/** Get metadata for a specific technique. */
export function getTechniqueMetadata(
  technique: PromptingTechnique,
): TechniqueMetadata | undefined {
  return TECHNIQUE_METADATA.get(technique);
}

/** Get all technique metadata entries. */
export function getAllTechniqueMetadata(): TechniqueMetadata[] {
  return [...TECHNIQUE_METADATA.values()];
}

/** Get techniques filtered by minimum SEAL quality score. */
export function getTechniquesBySealQuality(
  sealQuality: number,
): TechniqueMetadata[] {
  return getAllTechniqueMetadata().filter(
    (t) => t.minSealQuality <= sealQuality,
  );
}

/** Get techniques by category. */
export function getTechniquesByCategory(
  category: TechniqueCategory,
): TechniqueMetadata[] {
  return getAllTechniqueMetadata().filter((t) => t.category === category);
}

/** Get techniques by complexity level. */
export function getTechniquesByComplexity(
  level: ComplexityLevel,
): TechniqueMetadata[] {
  return getAllTechniqueMetadata().filter(
    (t) => t.complexityLevel === level,
  );
}

/** Count techniques by complexity level. */
export function countByComplexity(level: ComplexityLevel): number {
  return getTechniquesByComplexity(level).length;
}
