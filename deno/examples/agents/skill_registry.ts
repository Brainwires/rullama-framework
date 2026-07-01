// Example: Skill Registry
// Demonstrates SKILL.md creation, skill discovery with SkillRegistry,
// query matching with SkillRouter, and lazy-loading full skill instructions.
// Run: deno run --allow-read --allow-write --allow-env deno/examples/skills/skill_registry.ts

import {
  createSkillMetadata,
  type DiscoveryPath,
  executionMode,
  hasToolRestrictions,
  isToolAllowed,
  type Skill,
  type SkillMetadata,
  SkillRegistry,
  SkillRouter,
  type SkillSource,
} from "@rullama/agent";

async function main(): Promise<void> {
  // -----------------------------------------------------------------------
  // 1. Setup -- create temp dir with three SKILL.md files
  // -----------------------------------------------------------------------
  console.log("=== 1. Setup: Creating SKILL.md files ===\n");

  const tempDir = `${
    Deno.env.get("TMPDIR") ?? "/tmp"
  }/rullama-skills-example`;
  try {
    await Deno.mkdir(tempDir, { recursive: true });
  } catch (e) {
    if (!(e instanceof Deno.errors.AlreadyExists)) throw e;
  }

  const skills: Array<[string, string, string]> = [
    [
      "review-pr",
      "Reviews pull requests for code quality and best practices",
      "# PR Review Instructions\n\n" +
      "1. Check for code style violations\n" +
      "2. Look for security issues\n" +
      "3. Verify test coverage\n" +
      "4. Suggest improvements",
    ],
    [
      "deploy",
      "Deploys applications to staging or production environments",
      "# Deploy Instructions\n\n" +
      "1. Verify build passes\n" +
      "2. Run database migrations\n" +
      "3. Deploy to target environment\n" +
      "4. Run smoke tests",
    ],
    [
      "test-gen",
      "Generates unit tests for functions and modules",
      "# Test Generation Instructions\n\n" +
      "1. Analyze function signatures\n" +
      "2. Identify edge cases\n" +
      "3. Generate test stubs\n" +
      "4. Add assertions for expected behavior",
    ],
  ];

  for (const [name, description, instructions] of skills) {
    const content =
      `---\nname: ${name}\ndescription: ${description}\n---\n\n${instructions}\n`;
    const path = `${tempDir}/${name}.md`;
    await Deno.writeTextFile(path, content);
    console.log(`  Created: ${path}`);
  }

  // -----------------------------------------------------------------------
  // 2. Discovery -- create SkillRegistry and discover from temp dir
  // -----------------------------------------------------------------------
  console.log("\n=== 2. Discovery: Scanning for skills ===\n");

  const registry = new SkillRegistry();
  const discoveryPaths: DiscoveryPath[] = [
    { path: tempDir, source: "personal" },
  ];
  registry.discoverFrom(discoveryPaths);

  console.log(`  Discovered ${registry.length} skills`);

  // -----------------------------------------------------------------------
  // 3. Listing -- show all discovered skills with metadata
  // -----------------------------------------------------------------------
  console.log("\n=== 3. Listing: All discovered skills ===\n");

  for (const name of registry.listSkills()) {
    const meta = registry.getMetadata(name);
    if (meta) {
      console.log(`  /${meta.name} -- ${meta.description}`);
      console.log(`    Source: ${meta.source}`);
    }
  }

  // -----------------------------------------------------------------------
  // 4. Routing -- match user queries against skills
  // -----------------------------------------------------------------------
  console.log("\n=== 4. Routing: Matching queries to skills ===\n");

  const router = new SkillRouter(registry);

  const queries = [
    "review my pull request for quality issues",
    "deploy the app to production",
    "generate tests for this module",
    "completely unrelated cooking recipe",
  ];

  for (const query of queries) {
    const matches = router.matchSkills(query);
    console.log(`  Query: "${query}"`);
    if (matches.length === 0) {
      console.log("    No matches found");
    } else {
      for (const m of matches) {
        console.log(
          `    -> ${m.skillName} (confidence: ${
            m.confidence.toFixed(2)
          }, source: ${m.source})`,
        );
      }
    }
    console.log();
  }

  // -----------------------------------------------------------------------
  // 5. Format suggestions the way the CLI would show them
  // -----------------------------------------------------------------------
  console.log("=== 5. Formatted suggestions ===\n");

  const codeReviewMatches = router.matchSkills("review code quality");
  const suggestion = router.formatSuggestions(codeReviewMatches);
  console.log(`  ${suggestion ?? "No suggestions"}`);

  // -----------------------------------------------------------------------
  // 6. Load full skill -- lazy-load instructions from disk
  // -----------------------------------------------------------------------
  console.log("\n=== 6. Loading full skill content ===\n");

  const skill: Skill = registry.getSkill("review-pr");

  console.log(`  Skill: ${skill.metadata.name}`);
  console.log(`  Description: ${skill.metadata.description}`);
  console.log(`  Execution mode: ${skill.executionMode}`);
  console.log("  Instructions:\n");
  for (const line of skill.instructions.split("\n")) {
    console.log(`    ${line}`);
  }

  // -----------------------------------------------------------------------
  // 7. Metadata utility functions
  // -----------------------------------------------------------------------
  console.log("\n=== 7. Metadata Utilities ===\n");

  // Create metadata programmatically
  const customMeta = createSkillMetadata(
    "custom-skill",
    "A programmatically created skill",
  );
  console.log(
    `  Created metadata: ${customMeta.name} -- ${customMeta.description}`,
  );
  console.log(`  Execution mode: ${executionMode(customMeta)}`);
  console.log(`  Has tool restrictions: ${hasToolRestrictions(customMeta)}`);
  console.log(`  Is 'Read' allowed: ${isToolAllowed(customMeta, "Read")}`);

  // Register it
  registry.register(customMeta);
  console.log(`  Registered. Total skills: ${registry.length}`);

  // -----------------------------------------------------------------------
  // 8. Formatted skill list
  // -----------------------------------------------------------------------
  console.log("\n=== 8. Formatted Skill List ===\n");
  console.log(registry.formatSkillList());

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------
  try {
    await Deno.remove(tempDir, { recursive: true });
  } catch {
    // Ignore cleanup errors
  }

  console.log("Done.");
}

await main();
