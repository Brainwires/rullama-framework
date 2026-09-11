# Prompting

The `@rullama/prompting` package implements the 15 adaptive prompting techniques
from "Adaptive Selection of Prompting Techniques" (arXiv:2510.18162), plus task
clustering, prompt generation, temperature optimization and a learning
coordinator. (Before v0.11.0 this lived in `@rullama/knowledge`; that package
now holds only the knowledge-graph types -- see
[Knowledge](#knowledge-brainclient) below. RAG and code analysis are in
[rag.md](./rag.md).)

## Technique Catalog

Every enum is a PascalCase string union:

- `TechniqueCategory`:
  `"RoleAssignment" | "EmotionalStimulus" | "Reasoning" | "Others"`
- `ComplexityLevel`: `"Simple" | "Moderate" | "Advanced"`
- `PromptingTechnique`: `"RolePlaying"`, `"ChainOfThought"`, `"LeastToMost"`,
  `"PlanAndSolve"`, `"SkeletonOfThought"`, … (15 in `ALL_TECHNIQUES`)
- `TaskCharacteristic`: `"MultiStepReasoning"`, `"CodeGeneration"`,
  `"LogicalDeduction"`, … (`ALL_TASK_CHARACTERISTICS`)

```ts
import {
  ALL_TECHNIQUES,
  getTechniqueMetadata,
  getTechniquesByCategory,
  getTechniquesByComplexity,
  getTechniquesBySealQuality,
  parseTechniqueId,
  techniqueToId,
} from "@rullama/prompting";

const meta = getTechniqueMetadata("ChainOfThought"); // TechniqueMetadata
const reasoning = getTechniquesByCategory("Reasoning"); // PromptingTechnique[]
const simple = getTechniquesByComplexity("Simple");
const forQuality = getTechniquesBySealQuality(0.8); // techniques allowed at this SEAL score

techniqueToId("ChainOfThought"); // "chain_of_thought"
parseTechniqueId("chain_of_thought"); // "ChainOfThought" | undefined
```

See: `../examples/knowledge/prompting_techniques.ts`.

## Prompt Generation

`PromptGenerator` needs a `TaskClusterManager`: it matches the task embedding to
a cluster, picks the techniques that fit the cluster's characteristics and SEAL
quality, and composes a `GeneratedPrompt`.

```ts
import {
  createTaskCluster,
  PromptGenerator,
  TaskClusterManager,
} from "@rullama/prompting";

const clusters = new TaskClusterManager(3); // embedding dimension
clusters.addCluster(createTaskCluster({
  id: "code",
  description: "Code generation tasks",
  embedding: [1, 0, 0],
  techniques: ["ChainOfThought", "PlanAndSolve"],
  exampleTasks: ["Implement a function"],
  recommendedComplexity: "Moderate",
}));

const generator = new PromptGenerator(clusters);
const prompt = generator.generatePrompt(
  "Write a binary search in TypeScript",
  [0.9, 0.1, 0],
  0.7, // SEAL quality
);
// prompt?.systemPrompt, prompt?.techniques, prompt?.clusterId
```

## Learning and Temperature

- `PromptingLearningCoordinator` records per-cluster technique outcomes
  (`recordOutcome`) and exposes `bestTechnique` / `promotableTechniques`
  candidates for SEAL-driven adaptation.
- `TemperatureOptimizer` tracks `TemperaturePerformance` per cluster and returns
  the best sampling temperature (`getOptimalTemperature`).
- `TaskClusterManager.buildClustersFromEmbeddings` groups embeddings by cosine
  similarity; `cosineSimilarity`, `euclideanDistance`, `computeCentroid` are
  exported helpers.

## Knowledge (BrainClient)

`@rullama/knowledge` is the type contract for the "Open Brain" knowledge system:
`Thought` (with `ThoughtCategory` / `ThoughtSource` parsers), `Entity`,
`Relationship`, `ExtractionResult`, the request/response shapes, and the
`BrainClient` interface. No `BrainClient` implementation ships -- a concrete one
needs a storage backend and an embedding provider.

```ts
import type { BrainClient } from "@rullama/knowledge";
import { createThought } from "@rullama/knowledge";

const thought = createThought("The auth module uses JWT tokens");
thought.category = "insight";

// With a BrainClient implementation:
// await client.captureThought({ content: thought.content, category: "insight" });
// const hits = await client.searchMemory({ query: "authentication" });
```

`BrainClient` methods: `captureThought`, `searchMemory`, `listRecent`,
`getThought`, `searchKnowledge`, `memoryStats`, `deleteThought`.

See: `../examples/knowledge/knowledge_graph.ts`.

## Further Reading

- [RAG and code analysis](./rag.md)
- [Agents](./agents.md) for using prompts in agent loops
- [Extensibility](./extensibility.md) for implementing a custom `BrainClient`
