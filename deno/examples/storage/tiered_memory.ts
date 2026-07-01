// Example: TieredMemory — hot/warm/cold memory hierarchy
// Demonstrates configuring TieredMemory with custom tier thresholds, adding
// messages with importance scores, demoting entries, and inspecting statistics.
// Run: deno run deno/examples/storage/tiered_memory.ts

import {
  computeMultiFactorScore,
  createTierMetadata,
  defaultTieredMemoryConfig,
  demoteTier,
  type MessageMetadata,
  promoteTier,
  recencyFromHours,
  recordAccess,
  retentionScore,
  TieredMemory,
  type TieredMemoryConfig,
} from "@rullama/storage";

async function main() {
  console.log("=== Tiered Memory Example ===\n");

  // 1. Configure tiered memory with custom thresholds
  const config: TieredMemoryConfig = {
    hotRetentionHours: 12,
    warmRetentionHours: 72,
    hotImportanceThreshold: 0.4,
    warmImportanceThreshold: 0.2,
    maxHotMessages: 500,
    maxWarmSummaries: 2000,
  };

  console.log("TieredMemory config:");
  console.log(`  hot retention:  ${config.hotRetentionHours} hours`);
  console.log(`  warm retention: ${config.warmRetentionHours} hours`);
  console.log(`  max hot msgs:   ${config.maxHotMessages}`);
  console.log();

  // Compare with defaults
  const defaults = defaultTieredMemoryConfig();
  console.log("Default config for comparison:");
  console.log(`  hot retention:  ${defaults.hotRetentionHours} hours`);
  console.log(`  warm retention: ${defaults.warmRetentionHours} hours`);
  console.log();

  // 2. Create the tiered memory system
  const tiered = new TieredMemory(config);

  // 3. Add messages with varying importance scores
  const conversationId = "conv-tiered-1";
  const now = Math.floor(Date.now() / 1000);

  const entries: [string, number][] = [
    [
      "Architecture decision: we will use an event-driven design with CQRS.",
      0.95,
    ],
    ["Let me check the test output... all 42 tests pass.", 0.2],
    [
      "The database schema has three main tables: users, projects, and events.",
      0.85,
    ],
    ["Can you add a newline at the end of that file?", 0.05],
    ["We decided to use PostgreSQL with pgvector for production storage.", 0.9],
  ];

  console.log("--- Adding Messages ---");
  for (let i = 0; i < entries.length; i++) {
    const [content, importance] = entries[i];
    const msg: MessageMetadata = {
      messageId: `tmsg-${i + 1}`,
      conversationId,
      role: i % 2 === 0 ? "user" : "assistant",
      content,
      tokenCount: content.split(/\s+/).length,
      createdAt: now + i,
    };
    tiered.addMessage(msg, importance);
    console.log(`  Added (importance=${importance.toFixed(2)}): ${content}`);
  }
  console.log();

  // 4. Check tier statistics
  const stats = tiered.getStats();
  console.log("--- Tier Statistics ---");
  console.log(`  Hot:   ${stats.hotCount} entries`);
  console.log(`  Warm:  ${stats.warmCount} entries`);
  console.log(`  Cold:  ${stats.coldCount} entries`);
  console.log(`  Total: ${stats.totalTracked} tracked`);
  console.log();

  // 5. Identify demotion candidates (lowest retention score)
  console.log("--- Demotion Candidates (lowest retention score) ---");
  const candidates = tiered.getDemotionCandidates("hot", 2);
  for (const id of candidates) {
    const meta = tiered.getTierMetadata(id);
    const msg = tiered.getHotMessage(id);
    if (meta && msg) {
      console.log(
        `  ${id}: score=${retentionScore(meta).toFixed(3)}, ` +
          `importance=${meta.importance.toFixed(2)} -- "${
            msg.content.slice(0, 50)
          }..."`,
      );
    }
  }
  console.log();

  // 6. Demote the weakest message to warm tier
  console.log("--- Demoting to Warm Tier ---");
  if (candidates.length > 0) {
    const demoteId = candidates[0];
    const msg = tiered.getHotMessage(demoteId);
    if (msg) {
      const summary = {
        summaryId: `summary-${demoteId}`,
        originalMessageId: demoteId,
        conversationId,
        role: msg.role,
        summary: tiered.fallbackSummarize(msg.content),
        keyEntities: [],
        createdAt: now,
      };
      tiered.demoteToWarm(demoteId, summary);
      console.log(`  Demoted ${demoteId} to warm tier`);
      console.log(`  Summary: "${summary.summary}"`);
    }
  }

  const statsAfter = tiered.getStats();
  console.log(
    `\n  Stats after demotion: hot=${statsAfter.hotCount}, warm=${statsAfter.warmCount}`,
  );
  console.log();

  // 7. Record access and promote
  console.log("--- Access Recording & Promotion ---");
  tiered.recordAccess("tmsg-1");
  const meta1 = tiered.getTierMetadata("tmsg-1");
  if (meta1) {
    console.log(`  tmsg-1 access count: ${meta1.accessCount}`);
  }
  console.log();

  // 8. Demonstrate tier navigation helpers
  console.log("--- Tier Navigation ---");
  console.log(`  hot  -> demote -> ${demoteTier("hot") ?? "none"}`);
  console.log(`  warm -> demote -> ${demoteTier("warm") ?? "none"}`);
  console.log(
    `  cold -> demote -> ${demoteTier("cold") ?? "none (already coldest)"}`,
  );
  console.log(`  cold -> promote -> ${promoteTier("cold") ?? "none"}`);
  console.log(`  warm -> promote -> ${promoteTier("warm") ?? "none"}`);
  console.log(
    `  hot  -> promote -> ${promoteTier("hot") ?? "none (already hottest)"}`,
  );
  console.log();

  // 9. Multi-factor scoring
  console.log("--- Multi-Factor Scoring ---");
  const scores = [
    { similarity: 0.9, recency: recencyFromHours(1), importance: 0.95 },
    { similarity: 0.7, recency: recencyFromHours(24), importance: 0.5 },
    { similarity: 0.8, recency: recencyFromHours(72), importance: 0.2 },
  ];
  for (const s of scores) {
    const mf = computeMultiFactorScore(s.similarity, s.recency, s.importance);
    console.log(
      `  sim=${s.similarity.toFixed(2)} rec=${s.recency.toFixed(3)} imp=${
        s.importance.toFixed(2)
      } -> combined=${mf.combined.toFixed(3)}`,
    );
  }
  console.log();

  // 10. Tier metadata creation and retention scoring
  console.log("--- Retention Score Comparison ---");
  const highImportance = createTierMetadata("important-msg", 0.95);
  const lowImportance = createTierMetadata("trivial-msg", 0.05);
  console.log(
    `  High importance (0.95): retention=${
      retentionScore(highImportance).toFixed(3)
    }`,
  );
  console.log(
    `  Low importance  (0.05): retention=${
      retentionScore(lowImportance).toFixed(3)
    }`,
  );

  // Simulate access boosting retention
  recordAccess(highImportance);
  recordAccess(highImportance);
  recordAccess(highImportance);
  console.log(
    `  High importance after 3 accesses: retention=${
      retentionScore(highImportance).toFixed(3)
    }`,
  );

  console.log("\nDone.");
}

await main();
