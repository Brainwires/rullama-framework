// Example: Prompting Techniques
// Demonstrates listing, grouping, and filtering the 15 adaptive prompting techniques
// by category, complexity level, and task characteristics.
// Run: deno run deno/examples/cognition/prompting_techniques.ts

import {
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
} from "@rullama/knowledge";

import type {
  ComplexityLevel,
  PromptingTechnique,
  TaskCharacteristic,
  TechniqueCategory,
  TechniqueMetadata,
} from "@rullama/knowledge";

function categoryLabel(cat: TechniqueCategory): string {
  const labels: Record<TechniqueCategory, string> = {
    RoleAssignment: "Role Assignment",
    EmotionalStimulus: "Emotional Stimulus",
    Reasoning: "Reasoning",
    Others: "Others",
  };
  return labels[cat];
}

async function main() {
  // 1. Setup -- load metadata for every technique
  console.log("=== Prompting Techniques Library ===\n");

  const allMetadata = getAllTechniqueMetadata();
  console.log(`Loaded ${allMetadata.length} techniques\n`);

  // 2. List every technique with its metadata
  console.log("=== All Techniques ===\n");
  console.log(
    `${"#".padEnd(6)} ${"Name".padEnd(34)} ${"Category".padEnd(20)} ${
      "Level".padEnd(10)
    } ${"SEAL".padEnd(6)}`,
  );
  console.log("-".repeat(80));

  for (let i = 0; i < ALL_TECHNIQUES.length; i++) {
    const technique = ALL_TECHNIQUES[i];
    const meta = getTechniqueMetadata(technique);
    if (!meta) continue;

    console.log(
      `${String(i + 1).padEnd(6)} ` +
        `${meta.name.padEnd(34)} ` +
        `${categoryLabel(meta.category).padEnd(20)} ` +
        `${meta.complexityLevel.padEnd(10)} ` +
        `${meta.minSealQuality.toFixed(1)}`,
    );
  }
  console.log();

  // 3. Group by TechniqueCategory
  console.log("=== Grouped by Category ===\n");

  for (const cat of ALL_CATEGORIES) {
    const members = getTechniquesByCategory(cat);
    const suffix = members.length === 1 ? "" : "s";
    console.log(
      `${categoryLabel(cat)} (${members.length} technique${suffix}):`,
    );
    for (const m of members) {
      console.log(`  - ${m.name} -- ${m.description}`);
    }
    console.log();
  }

  // 4. Show techniques suitable for each ComplexityLevel
  console.log("=== By Complexity Level ===\n");

  for (const level of ALL_COMPLEXITY_LEVELS) {
    const members = getTechniquesByComplexity(level);
    const names = members.map((m: TechniqueMetadata) => m.name).join(", ");
    console.log(`${level.padEnd(10)} (${members.length}): ${names}`);
  }
  console.log();

  // 5. Show techniques best-for specific TaskCharacteristics
  console.log("=== By Task Characteristic ===\n");

  const selectedCharacteristics: TaskCharacteristic[] = [
    "MultiStepReasoning",
    "NumericalCalculation",
    "LogicalDeduction",
    "CreativeGeneration",
    "LongContextSummarization",
    "CodeGeneration",
    "AlgorithmicProblem",
  ];

  for (const tc of selectedCharacteristics) {
    const matching = allMetadata.filter((m: TechniqueMetadata) =>
      m.bestFor.includes(tc)
    );
    const names = matching.map((m: TechniqueMetadata) => m.name).join(", ");
    console.log(`${tc} (${matching.length}): ${names}`);
  }
  console.log();

  // 6. Comparison table -- SEAL thresholds
  console.log("=== SEAL Quality Threshold Table ===\n");
  console.log(
    `${"Technique".padEnd(34)} ${"Level".padEnd(10)} ${"MinSEAL".padEnd(10)} ${
      "BKS?".padEnd(6)
    }`,
  );
  console.log("-".repeat(64));

  const sorted = [...allMetadata].sort((a, b) =>
    a.minSealQuality - b.minSealQuality
  );

  for (const meta of sorted) {
    console.log(
      `${meta.name.padEnd(34)} ` +
        `${meta.complexityLevel.padEnd(10)} ` +
        `${meta.minSealQuality.toFixed(1).padEnd(10)} ` +
        `${meta.bksPromotionEligible ? "yes" : "no"}`,
    );
  }
  console.log();

  // 7. Technique ID conversion
  console.log("=== Technique ID Conversion ===\n");

  const sampleTechniques: PromptingTechnique[] = [
    "ChainOfThought",
    "PlanAndSolve",
    "SkillsInContext",
  ];

  for (const t of sampleTechniques) {
    const id = techniqueToId(t);
    const roundTrip = parseTechniqueId(id);
    console.log(`  ${t} -> "${id}" -> ${roundTrip}`);
  }
  console.log();

  // 8. Filter by SEAL quality
  console.log("=== Techniques Available at SEAL 0.6 ===\n");

  const available = getTechniquesBySealQuality(0.6);
  console.log(`${available.length} techniques available at SEAL >= 0.6:`);
  for (const m of available) {
    console.log(`  - ${m.name} (min SEAL: ${m.minSealQuality})`);
  }

  console.log("\n=== Done ===");
}

await main();
