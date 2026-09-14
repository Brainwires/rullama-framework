# @rullama/skills

SKILL.md markdown-manifest skills system: parser, metadata, `SkillRegistry`
(progressive disclosure -- metadata at startup, full instructions on demand),
`SkillRouter` (keyword / semantic / explicit matching) and `SkillExecutor`
(inline, subagent or script execution modes). No `@rullama/*` dependencies.

Extracted from the old `@rullama/agents` package in v0.11.0 to mirror Rust's
`rullama-skills` crate.

## Install

```sh
deno add jsr:@rullama/skills
```

## Quick Example

A skill is a markdown file with YAML front matter:

```md
---
name: deploy
description: Deploys the application to staging or production
---

# Deploy Instructions

1. Verify the build passes
2. Run database migrations
3. Deploy to the target environment
```

```ts
import {
  parseMetadataFromContent,
  SkillRegistry,
  SkillRouter,
} from "@rullama/skills";

const registry = new SkillRegistry();

// Discover SKILL.md files from disk (metadata only)...
registry.discoverFrom([
  { path: `${Deno.env.get("HOME")}/.rullama/skills`, source: "personal" },
  { path: "./skills", source: "project" },
]);

// ...or register parsed metadata directly
registry.register(parseMetadataFromContent(
  "---\nname: deploy\ndescription: Deploys the application to staging or production\n---\n\n# Deploy Instructions\n",
  "skills/deploy.md",
));

console.log(registry.listSkills(), registry.length);
console.log(registry.formatSkillList());

// Route a user query to matching skills
const router = new SkillRouter(registry).withMinConfidence(0.3);
const matches = router.matchSkills("deploy the app to staging");
console.log(matches.map((m) => `${m.skillName} (${m.confidence.toFixed(2)})`));

// Full instructions are loaded lazily and cached
if (matches.length > 0) {
  const skill = registry.getSkill(matches[0].skillName);
  console.log(skill.metadata.description, skill.instructions.length);
}
```
