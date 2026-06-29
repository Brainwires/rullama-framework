// Example: Agent Card Construction
// Demonstrates building A2A AgentCards with capabilities, skills,
// security schemes, and JSON serialization with round-trip verification.
// Run: deno run deno/examples/a2a/agent_card.ts

import type {
  AgentCapabilities,
  AgentCard,
  AgentProvider,
  AgentSkill,
  HttpAuthSecurityScheme,
  SecurityScheme,
} from "@rullama/a2a";

async function main(): Promise<void> {
  // -----------------------------------------------------------------------
  // 1. Build a full AgentCard with all fields populated
  // -----------------------------------------------------------------------
  console.log("=== Full AgentCard ===\n");

  const securitySchemes: Record<string, SecurityScheme> = {
    bearer_auth: {
      httpAuthSecurityScheme: {
        scheme: "Bearer",
        bearerFormat: "JWT",
        description: "JWT Bearer token authentication",
      },
    },
  };

  const fullCard: AgentCard = {
    name: "code-review-agent",
    description:
      "An autonomous code review agent that analyzes pull requests and provides actionable feedback.",
    version: "1.2.0",
    supportedInterfaces: [],
    capabilities: {
      streaming: true,
      pushNotifications: true,
      extendedAgentCard: false,
    },
    skills: [
      {
        id: "review-pr",
        name: "Pull Request Review",
        description:
          "Analyzes code diffs and produces structured review comments.",
        tags: ["code-review", "static-analysis", "security"],
        examples: [
          "Review this PR for security issues",
          "Check for performance regressions in the diff",
        ],
      },
      {
        id: "suggest-fix",
        name: "Suggest Fix",
        description: "Generates code fix suggestions for identified issues.",
        tags: ["code-generation", "refactoring"],
        inputModes: ["text/plain"],
        outputModes: ["text/plain", "application/json"],
      },
    ],
    defaultInputModes: ["text/plain", "application/json"],
    defaultOutputModes: ["text/plain", "application/json"],
    provider: {
      url: "https://brainwires.dev",
      organization: "Brainwires",
    },
    securitySchemes,
    documentationUrl: "https://docs.brainwires.dev/agents/code-review",
    iconUrl: "https://brainwires.dev/icons/code-review.svg",
  };

  // -----------------------------------------------------------------------
  // 2. Serialize to JSON
  // -----------------------------------------------------------------------
  const json = JSON.stringify(fullCard, null, 2);
  console.log(json);
  console.log();

  // -----------------------------------------------------------------------
  // 3. Round-trip: deserialize and verify
  // -----------------------------------------------------------------------
  console.log("=== Round-Trip Verification ===\n");

  const deserialized: AgentCard = JSON.parse(json);

  console.assert(
    deserialized.name === fullCard.name,
    "round-trip name mismatch",
  );
  console.assert(
    deserialized.skills.length === fullCard.skills.length,
    "round-trip skills count mismatch",
  );
  console.log(
    `Round-trip OK: name = "${deserialized.name}", skills = ${deserialized.skills.length}`,
  );
  console.log();

  // -----------------------------------------------------------------------
  // 4. Build a minimal AgentCard (bare essentials only)
  // -----------------------------------------------------------------------
  console.log("=== Minimal AgentCard ===\n");

  const minimalCard: AgentCard = {
    name: "echo-agent",
    description: "A minimal agent that echoes messages back.",
    version: "0.8.0",
    supportedInterfaces: [],
    capabilities: {},
    skills: [],
    defaultInputModes: ["text/plain"],
    defaultOutputModes: ["text/plain"],
  };

  const minimalJson = JSON.stringify(minimalCard, null, 2);
  console.log(minimalJson);
  console.log();

  // -----------------------------------------------------------------------
  // 5. Side-by-side comparison
  // -----------------------------------------------------------------------
  console.log("=== Comparison ===\n");

  const pad = (s: string, n: number) => s.padEnd(n);

  console.log(
    `${pad("Field", 20)} ${pad("Full Card", 25)} ${pad("Minimal Card", 25)}`,
  );
  console.log("-".repeat(70));
  console.log(
    `${pad("name", 20)} ${pad(fullCard.name, 25)} ${pad(minimalCard.name, 25)}`,
  );
  console.log(
    `${pad("version", 20)} ${pad(fullCard.version, 25)} ${
      pad(minimalCard.version, 25)
    }`,
  );
  console.log(
    `${pad("skills", 20)} ${pad(String(fullCard.skills.length), 25)} ${
      pad(String(minimalCard.skills.length), 25)
    }`,
  );
  console.log(
    `${pad("streaming", 20)} ${
      pad(String(fullCard.capabilities.streaming), 25)
    } ${pad(String(minimalCard.capabilities.streaming), 25)}`,
  );
  console.log(
    `${pad("provider", 20)} ${
      pad(fullCard.provider?.organization ?? "None", 25)
    } ${pad(minimalCard.provider?.organization ?? "None", 25)}`,
  );
  console.log(
    `${pad("securitySchemes", 20)} ${
      pad(String(fullCard.securitySchemes !== undefined), 25)
    } ${pad(String(minimalCard.securitySchemes !== undefined), 25)}`,
  );

  console.log("\nDone.");
}

await main();
