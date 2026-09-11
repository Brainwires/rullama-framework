# @rullama/prompting

Adaptive prompting techniques + task clustering + temperature optimization. No
`@rullama/*` dependencies.

Extracted from `@rullama/knowledge` in v0.11.0 to mirror Rust's
`rullama-prompting` crate. (`@rullama/knowledge` now holds only the
knowledge-graph types and `BrainClient`.)

Contents:

- The 15 prompting techniques from arXiv:2510.18162 with metadata
  (`ALL_TECHNIQUES`, `TECHNIQUE_METADATA`, `getTechniquesByCategory` /
  `ByComplexity` / `BySealQuality`). Enum values are PascalCase string unions
  (`"ChainOfThought"`, `"Reasoning"`, `"Simple"`, `"CodeGeneration"`).
- `TaskClusterManager` -- cosine-similarity task clustering.
- `PromptGenerator` -- composes a system prompt from the techniques that fit a
  task's cluster and SEAL quality score.
- `TemperatureOptimizer` -- per-cluster sampling temperature.
- `PromptingLearningCoordinator` -- technique effectiveness tracking,
  `bestTechnique` / `promotableTechniques`.

## Install

```sh
deno add jsr:@rullama/prompting
```

## Quick Example

```ts
import {
  createTaskCluster,
  getTechniqueMetadata,
  getTechniquesByCategory,
  PromptGenerator,
  PromptingLearningCoordinator,
  TaskClusterManager,
  techniqueToId,
} from "@rullama/prompting";

// Browse the catalog
const cot = getTechniqueMetadata("ChainOfThought");
console.log(cot?.complexityLevel, techniqueToId("ChainOfThought")); // "chain_of_thought"
console.log(getTechniquesByCategory("Reasoning"));

// Cluster tasks (embeddings come from your EmbeddingProvider)
const clusters = new TaskClusterManager(3);
clusters.addCluster(createTaskCluster({
  id: "code",
  description: "Code generation tasks",
  embedding: [1, 0, 0],
  techniques: ["ChainOfThought", "PlanAndSolve"],
  exampleTasks: ["Implement a function"],
  recommendedComplexity: "Moderate",
}));

// Generate a prompt for a new task
const generator = new PromptGenerator(clusters);
const generated = generator.generatePrompt(
  "Write a binary search in TypeScript",
  [0.9, 0.1, 0],
  0.7, // SEAL quality score
);
console.log(generated?.clusterId, generated?.techniques);
console.log(generated?.systemPrompt);

// Learn which techniques work per cluster
const coordinator = new PromptingLearningCoordinator();
coordinator.recordOutcome(
  "code",
  ["ChainOfThought"],
  "Write a binary search in TypeScript",
  true, // success
  3, // iterations
  0.9, // quality score
);
console.log(coordinator.getClusterSummary("code"));
```
