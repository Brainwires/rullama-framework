// Example: Knowledge Graph
// Demonstrates entity extraction, relationship modeling, thought creation, and the BrainClient interface.
// Run: deno run deno/examples/cognition/knowledge_graph.ts

import {
  ALL_THOUGHT_CATEGORIES,
  createThought,
  parseThoughtCategory,
  parseThoughtSource,
} from "@rullama/knowledge";

import type {
  BrainClient,
  CaptureThoughtRequest,
  CaptureThoughtResponse,
  ContradictionEvent,
  Entity,
  EntityType,
  ExtractionResult,
  Relationship,
  SearchMemoryRequest,
  SearchMemoryResponse,
  Thought,
  ThoughtCategory,
} from "@rullama/knowledge";

// ---------------------------------------------------------------------------
// In-memory entity store (simplified demonstration)
// ---------------------------------------------------------------------------

interface EntityEntry {
  name: string;
  entityType: EntityType;
  messageIds: string[];
  mentionCount: number;
  firstSeen: number;
  lastSeen: number;
}

class SimpleEntityStore {
  private entities: Map<string, EntityEntry> = new Map();
  private relationships: Relationship[] = [];
  private contradictions: ContradictionEvent[] = [];

  addExtraction(
    extraction: ExtractionResult,
    messageId: string,
    timestamp: number,
  ): void {
    // Merge entities
    for (const [name, entityType] of extraction.entities) {
      const existing = this.entities.get(name);
      if (existing) {
        existing.mentionCount++;
        existing.lastSeen = timestamp;
        if (!existing.messageIds.includes(messageId)) {
          existing.messageIds.push(messageId);
        }
      } else {
        this.entities.set(name, {
          name,
          entityType,
          messageIds: [messageId],
          mentionCount: 1,
          firstSeen: timestamp,
          lastSeen: timestamp,
        });
      }
    }

    // Check for contradictions before adding relationships
    for (const rel of extraction.relationships) {
      if (rel.kind === "Modifies") {
        const existing = this.relationships.find(
          (r) =>
            r.kind === "Modifies" &&
            r.modifier === rel.modifier &&
            r.modified === rel.modified &&
            r.changeType !== rel.changeType,
        );
        if (existing && existing.kind === "Modifies") {
          this.contradictions.push({
            kind: "ConflictingModification",
            subject: rel.modified,
            existingContext: existing.changeType,
            newContext: rel.changeType,
          });
        }
      }
    }

    this.relationships.push(...extraction.relationships);
  }

  getTopEntities(n: number): EntityEntry[] {
    return [...this.entities.values()]
      .sort((a, b) => b.mentionCount - a.mentionCount)
      .slice(0, n);
  }

  getByType(entityType: EntityType): EntityEntry[] {
    return [...this.entities.values()].filter((e) =>
      e.entityType === entityType
    );
  }

  getRelated(name: string): string[] {
    const related = new Set<string>();
    for (const rel of this.relationships) {
      switch (rel.kind) {
        case "Contains":
          if (rel.container === name) related.add(rel.contained);
          if (rel.contained === name) related.add(rel.container);
          break;
        case "References":
          if (rel.from === name) related.add(rel.to);
          if (rel.to === name) related.add(rel.from);
          break;
        case "DependsOn":
          if (rel.dependent === name) related.add(rel.dependency);
          if (rel.dependency === name) related.add(rel.dependent);
          break;
        case "Modifies":
          if (rel.modifier === name) related.add(rel.modified);
          if (rel.modified === name) related.add(rel.modifier);
          break;
        case "CoOccurs":
          if (rel.entityA === name) related.add(rel.entityB);
          if (rel.entityB === name) related.add(rel.entityA);
          break;
        case "Defines":
          if (rel.definer === name) related.add(rel.defined);
          if (rel.defined === name) related.add(rel.definer);
          break;
      }
    }
    return [...related];
  }

  stats() {
    const byType: Record<string, number> = {};
    for (const e of this.entities.values()) {
      byType[e.entityType] = (byType[e.entityType] ?? 0) + 1;
    }
    return {
      totalEntities: this.entities.size,
      totalRelationships: this.relationships.length,
      entitiesByType: byType,
    };
  }

  pendingContradictions(): ContradictionEvent[] {
    return this.contradictions;
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log("=== rullama Knowledge Graph Example ===\n");

  // 1. Build an entity store from extracted entities
  console.log("--- Step 1: Populate the Entity Store ---\n");

  const store = new SimpleEntityStore();

  const extraction1: ExtractionResult = {
    entities: [
      ["main.rs", "file"],
      ["Config", "type"],
      ["process_request", "function"],
    ],
    relationships: [
      { kind: "Contains", container: "main.rs", contained: "process_request" },
      { kind: "References", from: "process_request", to: "Config" },
    ],
  };

  const extraction2: ExtractionResult = {
    entities: [
      ["server.rs", "file"],
      ["handle_connection", "function"],
      ["Config", "type"],
    ],
    relationships: [
      {
        kind: "Contains",
        container: "server.rs",
        contained: "handle_connection",
      },
      {
        kind: "DependsOn",
        dependent: "handle_connection",
        dependency: "Config",
      },
      {
        kind: "CoOccurs",
        entityA: "handle_connection",
        entityB: "process_request",
        messageId: "msg-2",
      },
    ],
  };

  const extraction3: ExtractionResult = {
    entities: [
      ["DatabaseError", "error"],
      ["process_request", "function"],
    ],
    relationships: [
      {
        kind: "Modifies",
        modifier: "process_request",
        modified: "DatabaseError",
        changeType: "handles",
      },
    ],
  };

  store.addExtraction(extraction1, "msg-1", 1000);
  store.addExtraction(extraction2, "msg-2", 2000);
  store.addExtraction(extraction3, "msg-3", 3000);

  const stats = store.stats();
  console.log(`Total entities: ${stats.totalEntities}`);
  console.log(`Total relationships: ${stats.totalRelationships}`);
  console.log("Entities by type:");
  for (const [entityType, count] of Object.entries(stats.entitiesByType)) {
    console.log(`  ${entityType}: ${count}`);
  }
  console.log();

  // 2. Query the entity store
  console.log("--- Step 2: Query the Entity Store ---\n");

  const top = store.getTopEntities(3);
  console.log("Top 3 entities by mention count:");
  for (const entity of top) {
    console.log(
      `  ${entity.name} (${entity.entityType}) -- ${entity.mentionCount} mentions`,
    );
  }
  console.log();

  const functions = store.getByType("function");
  console.log(`Functions: [${functions.map((e) => `"${e.name}"`).join(", ")}]`);

  const related = store.getRelated("process_request");
  console.log(
    `Entities related to 'process_request': [${
      related.map((r) => `"${r}"`).join(", ")
    }]`,
  );
  console.log();

  // 3. Demonstrate relationship types
  console.log("--- Step 3: Relationship Types ---\n");

  const sampleRelationships: Relationship[] = [
    {
      kind: "Defines",
      definer: "auth.ts",
      defined: "AuthService",
      context: "class definition",
    },
    { kind: "References", from: "handler.ts", to: "AuthService" },
    {
      kind: "Modifies",
      modifier: "migration.sql",
      modified: "users_table",
      changeType: "adds column",
    },
    { kind: "DependsOn", dependent: "api.ts", dependency: "database.ts" },
    { kind: "Contains", container: "src/", contained: "auth.ts" },
    {
      kind: "CoOccurs",
      entityA: "Config",
      entityB: "Logger",
      messageId: "msg-5",
    },
  ];

  for (const rel of sampleRelationships) {
    switch (rel.kind) {
      case "Defines":
        console.log(
          `  ${rel.kind}: ${rel.definer} defines ${rel.defined} (${rel.context})`,
        );
        break;
      case "References":
        console.log(`  ${rel.kind}: ${rel.from} -> ${rel.to}`);
        break;
      case "Modifies":
        console.log(
          `  ${rel.kind}: ${rel.modifier} ${rel.changeType} ${rel.modified}`,
        );
        break;
      case "DependsOn":
        console.log(
          `  ${rel.kind}: ${rel.dependent} depends on ${rel.dependency}`,
        );
        break;
      case "Contains":
        console.log(
          `  ${rel.kind}: ${rel.container} contains ${rel.contained}`,
        );
        break;
      case "CoOccurs":
        console.log(
          `  ${rel.kind}: ${rel.entityA} & ${rel.entityB} in ${rel.messageId}`,
        );
        break;
    }
  }
  console.log();

  // 4. Contradiction detection
  console.log("--- Step 4: Contradiction Detection ---\n");

  const conflicting: ExtractionResult = {
    entities: [],
    relationships: [
      {
        kind: "Modifies",
        modifier: "process_request",
        modified: "DatabaseError",
        changeType: "ignores", // conflicts with "handles"
      },
    ],
  };
  store.addExtraction(conflicting, "msg-4", 4000);

  const contradictions = store.pendingContradictions();
  if (contradictions.length === 0) {
    console.log("No contradictions detected.");
  } else {
    console.log("Contradictions detected:");
    for (const c of contradictions) {
      console.log(
        `  ${c.kind} on '${c.subject}': existing='${c.existingContext}', new='${c.newContext}'`,
      );
    }
  }
  console.log();

  // 5. Thought creation
  console.log("--- Step 5: Thought Construction ---\n");

  const thought = createThought(
    "Decided to use PostgreSQL for the auth service",
  );
  thought.category = parseThoughtCategory("decision");
  thought.tags = ["database", "auth", "architecture"];
  thought.importance = 0.9;
  thought.source = parseThoughtSource("conversation");

  console.log(`Thought: ${thought.content}`);
  console.log(`  ID:         ${thought.id}`);
  console.log(`  Category:   ${thought.category}`);
  console.log(`  Tags:       [${thought.tags.join(", ")}]`);
  console.log(`  Source:     ${thought.source}`);
  console.log(`  Importance: ${thought.importance.toFixed(1)}`);
  console.log(
    `  Created:    ${new Date(thought.createdAt * 1000).toISOString()}`,
  );
  console.log();

  // 6. List all thought categories
  console.log("Available thought categories:");
  for (const cat of ALL_THOUGHT_CATEGORIES) {
    console.log(`  - ${cat}`);
  }
  console.log();

  // 7. Demonstrate category and source parsing
  console.log("--- Step 6: Category & Source Parsing ---\n");

  const testCategories = ["decision", "idea", "todo", "ref", "unknown_value"];
  for (const input of testCategories) {
    const parsed = parseThoughtCategory(input);
    console.log(`  parseThoughtCategory("${input}") -> "${parsed}"`);
  }
  console.log();

  const testSources = ["manual", "conversation_extract", "import", "webhook"];
  for (const input of testSources) {
    const parsed = parseThoughtSource(input);
    console.log(`  parseThoughtSource("${input}") -> "${parsed}"`);
  }

  console.log("\n=== Done ===");
}

await main();
