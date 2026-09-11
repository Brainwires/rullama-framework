/**
 * Adaptive prompting for rullama: the catalog of 15 prompting techniques with
 * per-technique metadata (`ALL_TECHNIQUES`, `TECHNIQUE_METADATA`, the
 * `getTechniquesBy*` filters), `TaskClusterManager` for cosine-similarity task
 * clustering, `PromptGenerator` for composing a prompt from the techniques that
 * fit a task, `TemperatureOptimizer` for per-cluster sampling temperature, and
 * `PromptingLearningCoordinator` for tracking technique effectiveness over time.
 * Equivalent to Rust's `rullama-prompting` crate.
 *
 * @module
 */

// ── Techniques ──────────────────────────────────────────────────────────────
export {
  ALL_CATEGORIES,
  ALL_COMPLEXITY_LEVELS,
  ALL_TASK_CHARACTERISTICS,
  ALL_TECHNIQUES,
  countByComplexity,
  getAllTechniqueMetadata,
  getTechniqueMetadata,
  getTechniquesByCategory,
  getTechniquesByComplexity,
  getTechniquesBySealQuality,
  parseTechniqueId,
  TECHNIQUE_METADATA,
  techniqueToId,
} from "./techniques.ts";

export type {
  ComplexityLevel,
  PromptingTechnique,
  TaskCharacteristic,
  TechniqueCategory,
  TechniqueMetadata,
} from "./techniques.ts";

// ── Clustering ──────────────────────────────────────────────────────────────
export {
  computeCentroid,
  cosineSimilarity,
  createTaskCluster,
  euclideanDistance,
  TaskClusterManager,
  updateClusterSealMetrics,
} from "./cluster.ts";

export type { TaskCluster, TaskClusterInit } from "./cluster.ts";

// ── Generator ───────────────────────────────────────────────────────────────
export {
  inferRoleAndDomain,
  inferTaskType,
  PromptGenerator,
} from "./generator.ts";

export type { GeneratedPrompt } from "./generator.ts";

// ── Temperature ─────────────────────────────────────────────────────────────
export {
  createTemperaturePerformance,
  TemperatureOptimizer,
  temperatureScore,
  updateTemperaturePerformance,
} from "./temperature.ts";

export type { TemperaturePerformance } from "./temperature.ts";

// ── Learning Coordinator ────────────────────────────────────────────────────
export {
  bestTechnique,
  createTechniqueStats,
  promotableTechniques,
  PromptingLearningCoordinator,
  statsReliability,
  statsTotalUses,
  updateTechniqueStats,
} from "./coordinator.ts";

export type {
  ClusterSummary,
  TechniqueEffectivenessRecord,
  TechniqueStats,
} from "./coordinator.ts";
