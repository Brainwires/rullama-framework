/**
 * Prompt generation with adaptive technique selection. `PromptGenerator`
 * matches a task to a cluster, selects the techniques that fit its
 * characteristics, complexity and SEAL quality scores, and composes them into a
 * `GeneratedPrompt`; `inferRoleAndDomain` and `inferTaskType` derive the
 * role/domain hints the templates use.
 * Equivalent to Rust's `rullama_prompting::generator`.
 *
 * @module
 */

import type { TaskCluster } from "./cluster.ts";
import type { TaskClusterManager } from "./cluster.ts";
import type { PromptingTechnique, TechniqueMetadata } from "./techniques.ts";
import { TECHNIQUE_METADATA } from "./techniques.ts";

// ---------------------------------------------------------------------------
// GeneratedPrompt
// ---------------------------------------------------------------------------

/** Result of prompt generation. */
export interface GeneratedPrompt {
  /** The generated system prompt text. */
  readonly systemPrompt: string;
  /** ID of the cluster that was matched. */
  readonly clusterId: string;
  /** Techniques that were selected. */
  readonly techniques: readonly PromptingTechnique[];
  /** SEAL quality score used. */
  readonly sealQuality: number;
  /** Similarity score to matched cluster. */
  readonly similarityScore: number;
}

// ---------------------------------------------------------------------------
// PromptGenerator
// ---------------------------------------------------------------------------

/**
 * Generates optimized prompts based on task characteristics.
 *
 * Orchestrates:
 * 1. Task classification using cluster matching
 * 2. Technique selection with SEAL quality filtering
 * 3. Dynamic prompt composition from selected techniques
 */
export class PromptGenerator {
  private readonly clusterManager: TaskClusterManager;

  /** Create a generator that matches tasks against `clusterManager`'s clusters. */
  constructor(clusterManager: TaskClusterManager) {
    this.clusterManager = clusterManager;
  }

  /**
   * Generate an optimized prompt.
   *
   * @param taskDescription - The task to generate a prompt for.
   * @param taskEmbedding - Pre-computed embedding of the task.
   * @param sealQuality - Optional SEAL quality score (0-1, defaults to 0.5).
   * @returns GeneratedPrompt or undefined if no clusters are available.
   */
  generatePrompt(
    taskDescription: string,
    taskEmbedding: readonly number[],
    sealQuality?: number,
  ): GeneratedPrompt | undefined {
    const quality = sealQuality ?? 0.5;

    // Step 1: Find matching cluster
    const match = this.clusterManager.findMatchingCluster(
      taskEmbedding,
      quality,
    );
    if (!match) return undefined;
    const [cluster, similarity] = match;

    // Step 2: Select techniques
    const techniques = this.selectTechniques(cluster, quality);

    // Step 3: Compose prompt
    const systemPrompt = this.composePrompt(
      taskDescription,
      techniques,
      cluster.description,
    );

    return {
      systemPrompt,
      clusterId: cluster.id,
      techniques: techniques.map((t) => t.technique),
      sealQuality: quality,
      similarityScore: similarity,
    };
  }

  /**
   * Select techniques based on cluster, SEAL quality, and optional preferences.
   *
   * Selection rules (from the paper):
   * 1. Always include RolePlaying if it passes quality filter.
   * 2. Select one EmotionalStimulus technique.
   * 3. Select one Reasoning technique (complexity-gated).
   * 4. Optionally select an "Others" technique if quality > 0.6.
   */
  selectTechniques(
    cluster: TaskCluster,
    sealQuality: number,
    pksPreferred: readonly PromptingTechnique[] = [],
    bksRecommended: readonly PromptingTechnique[] = [],
  ): TechniqueMetadata[] {
    const selected: TechniqueMetadata[] = [];

    // 1. Always include role playing
    const role = TECHNIQUE_METADATA.get("RolePlaying");
    if (role && role.minSealQuality <= sealQuality) {
      selected.push(role);
    }

    // Helper: get metadata for cluster techniques
    const clusterMeta = cluster.techniques
      .map((t) => TECHNIQUE_METADATA.get(t))
      .filter((m): m is TechniqueMetadata => m !== undefined);

    // 2. Emotional stimulus
    const emotionOptions = clusterMeta.filter(
      (t) =>
        t.category === "EmotionalStimulus" && t.minSealQuality <= sealQuality,
    );
    const emotion = selectBestByPriority(
      pksPreferred,
      bksRecommended,
      emotionOptions,
    );
    if (emotion) selected.push(emotion);

    // 3. Reasoning technique (complexity-gated)
    const reasoningOptions = clusterMeta.filter(
      (t) => t.category === "Reasoning" && t.minSealQuality <= sealQuality,
    );
    const reasoning = selectReasoningByComplexity(
      pksPreferred,
      bksRecommended,
      reasoningOptions,
      sealQuality,
    );
    if (reasoning) selected.push(reasoning);

    // 4. "Others" category if quality is high enough
    if (sealQuality > 0.6) {
      const supportOptions = clusterMeta.filter(
        (t) => t.category === "Others" && t.minSealQuality <= sealQuality,
      );
      const support = selectBestByPriority(
        pksPreferred,
        bksRecommended,
        supportOptions,
      );
      if (support) selected.push(support);
    }

    return selected;
  }

  /**
   * Compose prompt from selected techniques.
   *
   * Order: RoleAssignment -> EmotionalStimulus -> Reasoning -> Others -> Task.
   */
  composePrompt(
    taskDescription: string,
    techniques: readonly TechniqueMetadata[],
    clusterDescription: string,
  ): string {
    const parts: string[] = [];

    // Role assignment (always first)
    const role = techniques.find((t) => t.category === "RoleAssignment");
    if (role) {
      parts.push(this.applyTemplate(role, taskDescription, clusterDescription));
    }

    // Emotional stimulus
    const emotion = techniques.find((t) => t.category === "EmotionalStimulus");
    if (emotion) {
      parts.push(
        this.applyTemplate(emotion, taskDescription, clusterDescription),
      );
    }

    // Reasoning
    const reasoning = techniques.find((t) => t.category === "Reasoning");
    if (reasoning) {
      parts.push(
        this.applyTemplate(reasoning, taskDescription, clusterDescription),
      );
    }

    // Others
    for (const tech of techniques.filter((t) => t.category === "Others")) {
      parts.push(this.applyTemplate(tech, taskDescription, clusterDescription));
    }

    // Task description
    parts.push(`\n# Task\n\n${taskDescription}`);

    return parts.join("\n\n");
  }

  /** Apply technique template with variable substitution. */
  applyTemplate(
    technique: TechniqueMetadata,
    taskDescription: string,
    clusterDescription: string,
  ): string {
    let result = technique.template;

    if (technique.technique === "RolePlaying") {
      const [role, domain] = inferRoleAndDomain(
        taskDescription,
        clusterDescription,
      );
      result = result.replace("{role}", role).replace("{domain}", domain);
    }

    if (technique.technique === "EmotionPrompting") {
      const taskType = inferTaskType(taskDescription);
      result = result
        .replace("{task_type}", taskType)
        .replace("{quality}", "precision and accuracy");
    }

    result = result
      .replace("{task}", taskDescription)
      .replace("{cluster}", clusterDescription);

    return result;
  }

  /** Get reference to the cluster manager. */
  getClusterManager(): TaskClusterManager {
    return this.clusterManager;
  }
}

// ---------------------------------------------------------------------------
// Selection helpers
// ---------------------------------------------------------------------------

/** Select best technique by priority (PKS > BKS > default). */
function selectBestByPriority(
  pks: readonly PromptingTechnique[],
  bks: readonly PromptingTechnique[],
  options: readonly TechniqueMetadata[],
): TechniqueMetadata | undefined {
  if (options.length === 0) return undefined;

  let best = options[0];
  let bestScore = scoreByPriority(best.technique, pks, bks);

  for (let i = 1; i < options.length; i++) {
    const score = scoreByPriority(options[i].technique, pks, bks);
    if (score > bestScore) {
      bestScore = score;
      best = options[i];
    }
  }

  return best;
}

/** Select reasoning technique based on complexity gating. */
function selectReasoningByComplexity(
  pks: readonly PromptingTechnique[],
  bks: readonly PromptingTechnique[],
  options: readonly TechniqueMetadata[],
  sealQuality: number,
): TechniqueMetadata | undefined {
  const complexity = sealQuality < 0.5
    ? "Simple"
    : sealQuality < 0.8
    ? "Moderate"
    : "Advanced";

  const filtered = options.filter(
    (t) => t.complexityLevel === complexity || t.complexityLevel === "Simple",
  );

  if (filtered.length === 0) return undefined;

  let best = filtered[0];
  let bestScore = computeReasoningScore(best, pks, bks, complexity);

  for (let i = 1; i < filtered.length; i++) {
    const score = computeReasoningScore(filtered[i], pks, bks, complexity);
    if (score > bestScore) {
      bestScore = score;
      best = filtered[i];
    }
  }

  return best;
}

function scoreByPriority(
  technique: PromptingTechnique,
  pks: readonly PromptingTechnique[],
  bks: readonly PromptingTechnique[],
): number {
  if (pks.includes(technique)) return 2;
  if (bks.includes(technique)) return 1;
  return 0;
}

function computeReasoningScore(
  meta: TechniqueMetadata,
  pks: readonly PromptingTechnique[],
  bks: readonly PromptingTechnique[],
  targetComplexity: string,
): number {
  const pksBonus = pks.includes(meta.technique) ? 100 : 0;
  const bksBonus = bks.includes(meta.technique) ? 50 : 0;
  const complexityBonus = meta.complexityLevel === targetComplexity ? 10 : 0;
  return pksBonus + bksBonus + complexityBonus;
}

// ---------------------------------------------------------------------------
// Inference heuristics
// ---------------------------------------------------------------------------

/** Infer role and domain from task and cluster description. */
export function inferRoleAndDomain(
  taskDescription: string,
  clusterDescription: string,
): [string, string] {
  const taskLower = taskDescription.toLowerCase();
  const clusterLower = clusterDescription.toLowerCase();

  if (
    taskLower.includes("code") || taskLower.includes("function") ||
    taskLower.includes("implement")
  ) {
    return ["software engineer", "software development"];
  }
  if (taskLower.includes("algorithm") || taskLower.includes("optimize")) {
    return ["computer scientist", "algorithms and data structures"];
  }
  if (taskLower.includes("calculate") || taskLower.includes("numerical")) {
    return ["mathematician", "numerical analysis"];
  }
  if (taskLower.includes("analyze") || taskLower.includes("understand")) {
    return ["analyst", "problem analysis"];
  }
  if (clusterLower.includes("code")) {
    return ["developer", "software engineering"];
  }
  return ["expert", "problem solving"];
}

/** Infer task type for Emotion Prompting. */
export function inferTaskType(taskDescription: string): string {
  const taskLower = taskDescription.toLowerCase();

  if (taskLower.includes("calculate") || taskLower.includes("compute")) {
    return "calculation";
  }
  if (taskLower.includes("implement") || taskLower.includes("create")) {
    return "implementation";
  }
  if (taskLower.includes("analyze") || taskLower.includes("understand")) {
    return "analysis";
  }
  if (taskLower.includes("fix") || taskLower.includes("debug")) {
    return "debugging";
  }
  return "task";
}
