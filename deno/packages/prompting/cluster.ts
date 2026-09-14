/**
 * Task clustering for technique selection. `TaskClusterManager` groups task
 * embeddings into `TaskCluster`s by cosine similarity (a simplification of the
 * Rust k-means implementation), tracks SEAL quality metrics per cluster, and
 * finds the best-matching cluster for a new task; `cosineSimilarity`,
 * `euclideanDistance` and `computeCentroid` are the vector helpers it uses.
 * Equivalent to Rust's `rullama_prompting::clustering`.
 *
 * @module
 */

import type { ComplexityLevel, PromptingTechnique } from "./techniques.ts";

// ---------------------------------------------------------------------------
// TaskCluster
// ---------------------------------------------------------------------------

/** A task cluster identified by semantic similarity. */
export interface TaskCluster {
  /** Unique cluster identifier. */
  readonly id: string;
  /** Human-readable description of this cluster. */
  readonly description: string;
  /** Embedding vector of the cluster centroid. */
  readonly embedding: readonly number[];
  /** Prompting techniques mapped to this cluster (typically 3-4). */
  readonly techniques: readonly PromptingTechnique[];
  /** Example task descriptions belonging to this cluster. */
  readonly exampleTasks: readonly string[];
  /** Example SEAL query cores for tasks in this cluster. */
  readonly sealQueryCores: readonly string[];
  /** Average SEAL quality score for tasks in this cluster. */
  readonly avgSealQuality: number;
  /** Recommended complexity level based on average SEAL quality. */
  readonly recommendedComplexity: ComplexityLevel;
}

/** Options for creating a TaskCluster (mutable fields before freezing). */
export interface TaskClusterInit {
  /** Unique cluster ID. */
  id: string;
  /** Human-readable description of the cluster. */
  description: string;
  /** Centroid embedding of the cluster. */
  embedding: number[];
  /** Techniques recommended for this cluster. */
  techniques: PromptingTechnique[];
  /** Representative task descriptions. */
  exampleTasks: string[];
  /** SEAL query cores associated with the cluster. */
  sealQueryCores?: string[];
  /** Average SEAL quality score (0-1) observed for the cluster. */
  avgSealQuality?: number;
  /** Complexity level the cluster's tasks usually need. */
  recommendedComplexity?: ComplexityLevel;
}

/** Create a new TaskCluster from init options. */
export function createTaskCluster(init: TaskClusterInit): TaskCluster {
  return {
    id: init.id,
    description: init.description,
    embedding: init.embedding,
    techniques: init.techniques,
    exampleTasks: init.exampleTasks,
    sealQueryCores: init.sealQueryCores ?? [],
    avgSealQuality: init.avgSealQuality ?? 0.5,
    recommendedComplexity: init.recommendedComplexity ?? "Moderate",
  };
}

/** Update SEAL-related metrics on a cluster, returning a new cluster. */
export function updateClusterSealMetrics(
  cluster: TaskCluster,
  queryCores: string[],
  avgQuality: number,
): TaskCluster {
  const complexity: ComplexityLevel = avgQuality < 0.5
    ? "Simple"
    : avgQuality < 0.8
    ? "Moderate"
    : "Advanced";

  return {
    ...cluster,
    sealQueryCores: queryCores,
    avgSealQuality: avgQuality,
    recommendedComplexity: complexity,
  };
}

// ---------------------------------------------------------------------------
// TaskClusterManager
// ---------------------------------------------------------------------------

/** Manages task clustering using distance-based grouping. */
export class TaskClusterManager {
  private clusters: TaskCluster[] = [];
  private readonly embeddingDim: number;

  /** Create a manager for embeddings of `embeddingDim` dimensions (default 768). */
  constructor(embeddingDim: number = 768) {
    this.embeddingDim = embeddingDim;
  }

  /** Get all clusters. */
  getClusters(): readonly TaskCluster[] {
    return this.clusters;
  }

  /** Add a cluster. */
  addCluster(cluster: TaskCluster): void {
    this.clusters.push(cluster);
  }

  /** Replace all clusters. */
  setClusters(clusters: TaskCluster[]): void {
    this.clusters = [...clusters];
  }

  /** Get cluster count. */
  clusterCount(): number {
    return this.clusters.length;
  }

  /** Get cluster by ID. */
  getClusterById(id: string): TaskCluster | undefined {
    return this.clusters.find((c) => c.id === id);
  }

  /** Get embedding dimension. */
  getEmbeddingDim(): number {
    return this.embeddingDim;
  }

  /**
   * Find the cluster most similar to a task embedding.
   *
   * Optionally boosts similarity when SEAL quality is high.
   *
   * @param taskEmbedding - Pre-computed embedding of the task.
   * @param sealQuality - Optional SEAL quality score (0-1).
   * @returns Tuple of [cluster, similarity] or undefined if no clusters.
   */
  findMatchingCluster(
    taskEmbedding: readonly number[],
    sealQuality?: number,
  ): [TaskCluster, number] | undefined {
    if (this.clusters.length === 0) return undefined;

    let bestCluster: TaskCluster | undefined;
    let bestSimilarity = -Infinity;

    for (const cluster of this.clusters) {
      let similarity = cosineSimilarity(taskEmbedding, cluster.embedding);

      // Boost similarity if SEAL quality is high
      if (sealQuality !== undefined && sealQuality > 0.7) {
        similarity *= 1.1;
      }

      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestCluster = cluster;
      }
    }

    if (bestCluster === undefined) return undefined;
    return [bestCluster, bestSimilarity];
  }

  /**
   * Build clusters from embeddings using simple distance-based grouping.
   *
   * This is a simplified version that assigns each task to its nearest
   * centroid from the provided seed centroids (no iterative k-means).
   *
   * @param taskEmbeddings - Array of embedding vectors.
   * @param taskDescriptions - Corresponding task descriptions.
   * @param numClusters - Desired number of clusters.
   * @returns Array of cluster assignment indices.
   */
  buildClustersFromEmbeddings(
    taskEmbeddings: readonly (readonly number[])[],
    taskDescriptions: readonly string[],
    numClusters: number,
  ): number[] {
    if (taskEmbeddings.length !== taskDescriptions.length) {
      throw new Error(
        `Embeddings and descriptions length mismatch: ${taskEmbeddings.length} vs ${taskDescriptions.length}`,
      );
    }
    if (taskEmbeddings.length < numClusters) {
      throw new Error(
        `Not enough tasks (${taskEmbeddings.length}) for ${numClusters} clusters`,
      );
    }

    // Seed centroids: pick evenly-spaced indices
    const step = Math.floor(taskEmbeddings.length / numClusters);
    const centroids: number[][] = [];
    for (let i = 0; i < numClusters; i++) {
      centroids.push([...taskEmbeddings[i * step]]);
    }

    // Assign each task to the nearest centroid
    const assignments: number[] = [];
    for (const emb of taskEmbeddings) {
      let bestIdx = 0;
      let bestSim = -Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const sim = cosineSimilarity(emb, centroids[c]);
        if (sim > bestSim) {
          bestSim = sim;
          bestIdx = c;
        }
      }
      assignments.push(bestIdx);
    }

    // Recompute centroids and build cluster objects
    const newClusters: TaskCluster[] = [];
    for (let c = 0; c < numClusters; c++) {
      const memberEmbeddings: number[][] = [];
      const memberTasks: string[] = [];

      for (let i = 0; i < assignments.length; i++) {
        if (assignments[i] === c) {
          memberEmbeddings.push([...taskEmbeddings[i]]);
          memberTasks.push(taskDescriptions[i]);
        }
      }

      if (memberTasks.length === 0) continue;

      const centroid = computeCentroid(memberEmbeddings);
      newClusters.push(
        createTaskCluster({
          id: `cluster_${c}`,
          description: `Cluster ${c}`,
          embedding: centroid,
          techniques: [],
          exampleTasks: memberTasks.slice(0, 5),
        }),
      );
    }

    this.clusters = newClusters;
    return assignments;
  }
}

// ---------------------------------------------------------------------------
// Vector math utilities
// ---------------------------------------------------------------------------

/** Compute cosine similarity between two vectors. */
export function cosineSimilarity(
  a: readonly number[],
  b: readonly number[],
): number {
  if (a.length !== b.length) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}

/** Compute Euclidean distance between two vectors. */
export function euclideanDistance(
  a: readonly number[],
  b: readonly number[],
): number {
  if (a.length !== b.length) return Infinity;

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/** Compute centroid (mean vector) of a set of embeddings. */
export function computeCentroid(
  embeddings: readonly (readonly number[])[],
): number[] {
  if (embeddings.length === 0) return [];

  const dim = embeddings[0].length;
  const centroid = new Array<number>(dim).fill(0);

  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      centroid[i] += emb[i];
    }
  }

  const n = embeddings.length;
  for (let i = 0; i < dim; i++) {
    centroid[i] /= n;
  }

  return centroid;
}
