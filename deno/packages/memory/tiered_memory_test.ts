/**
 * Tests for TieredMemory, tier metadata, and multi-factor scoring.
 */

import { assertEquals } from "@std/assert";
import {
  computeMultiFactorScore,
  createTierMetadata,
  defaultTieredMemoryConfig,
  demoteTier,
  type KeyFact,
  type MessageSummary,
  parseMemoryAuthority,
  promoteTier,
  recencyFromHours,
  recordAccess,
  retentionScore,
  TieredMemory,
} from "./tiered_memory.ts";
import type { MessageMetadata } from "@rullama/stores";

function makeMessage(id: string, conversationId = "conv-1"): MessageMetadata {
  const now = Math.floor(Date.now() / 1000);
  return {
    messageId: id,
    conversationId,
    role: "user",
    content: `Message ${id}`,
    createdAt: now,
  };
}

// -- Multi-factor score tests -----------------------------------------------

Deno.test("MultiFactorScore - weights sum to 1", () => {
  const score = computeMultiFactorScore(1.0, 1.0, 1.0);
  assertEquals(Math.abs(score.combined - 1.0) < 1e-6, true);
});

Deno.test("MultiFactorScore - zero inputs", () => {
  const score = computeMultiFactorScore(0.0, 0.0, 0.0);
  assertEquals(score.combined, 0.0);
});

Deno.test("recencyFromHours - fresh entry is ~1.0", () => {
  const r = recencyFromHours(0.0);
  assertEquals(Math.abs(r - 1.0) < 1e-6, true);
});

Deno.test("recencyFromHours - decays over time", () => {
  const rNow = recencyFromHours(0.0);
  const rDay = recencyFromHours(24.0);
  const rWeek = recencyFromHours(168.0);
  assertEquals(rNow > rDay, true);
  assertEquals(rDay > rWeek, true);
  assertEquals(rWeek > 0, true);
});

Deno.test("balanced entry beats stale high-similarity entry", () => {
  const stale = computeMultiFactorScore(0.95, recencyFromHours(168.0), 0.0);
  const fresh = computeMultiFactorScore(0.70, recencyFromHours(1.0), 0.9);
  assertEquals(fresh.combined > stale.combined, true);
});

// -- Tier demotion/promotion tests ------------------------------------------

Deno.test("tier demotion chain", () => {
  assertEquals(demoteTier("hot"), "warm");
  assertEquals(demoteTier("warm"), "cold");
  assertEquals(demoteTier("cold"), undefined);
});

Deno.test("tier promotion chain", () => {
  assertEquals(promoteTier("hot"), undefined);
  assertEquals(promoteTier("warm"), "hot");
  assertEquals(promoteTier("cold"), "warm");
});

// -- TierMetadata tests -----------------------------------------------------

Deno.test("TierMetadata - retention score is positive", () => {
  const meta = createTierMetadata("test-1", 0.8);
  const score = retentionScore(meta);
  assertEquals(score > 0, true);
});

Deno.test("TierMetadata - record access maintains/increases score", () => {
  const meta = createTierMetadata("test-1", 0.8);
  const score1 = retentionScore(meta);
  recordAccess(meta);
  const score2 = retentionScore(meta);
  // Access factor from ln(1+1)*0.1 should add to score
  assertEquals(score2 >= score1 * 0.9, true);
});

Deno.test("TierMetadata - default authority is session", () => {
  const meta = createTierMetadata("m-1", 0.5);
  assertEquals(meta.authority, "session");
});

Deno.test("TierMetadata - with canonical authority", () => {
  const meta = createTierMetadata("m-2", 0.9, "canonical");
  assertEquals(meta.authority, "canonical");
  assertEquals(meta.importance, 0.9);
});

// -- MemoryAuthority tests --------------------------------------------------

Deno.test("MemoryAuthority - round trip", () => {
  assertEquals(parseMemoryAuthority("ephemeral"), "ephemeral");
  assertEquals(parseMemoryAuthority("session"), "session");
  assertEquals(parseMemoryAuthority("canonical"), "canonical");
});

Deno.test("MemoryAuthority - unknown defaults to session", () => {
  assertEquals(parseMemoryAuthority("bogus"), "session");
});

// -- Default config tests ---------------------------------------------------

Deno.test("default config values", () => {
  const config = defaultTieredMemoryConfig();
  assertEquals(config.hotRetentionHours, 24);
  assertEquals(config.warmRetentionHours, 168);
  assertEquals(config.hotImportanceThreshold > 0, true);
  assertEquals(config.sessionTtlSecs, undefined);
});

// -- TieredMemory integration tests -----------------------------------------

Deno.test("TieredMemory - add and retrieve hot message", () => {
  const tm = new TieredMemory();
  const msg = makeMessage("m-1");
  tm.addMessage(msg, 0.8);

  const retrieved = tm.getHotMessage("m-1");
  assertEquals(retrieved?.messageId, "m-1");

  const meta = tm.getTierMetadata("m-1");
  assertEquals(meta?.tier, "hot");
  assertEquals(meta?.importance, 0.8);
  assertEquals(meta?.authority, "session");
});

Deno.test("TieredMemory - canonical message", () => {
  const tm = new TieredMemory();
  const msg = makeMessage("m-canon");
  tm.addCanonicalMessage(msg, 0.95);

  const meta = tm.getTierMetadata("m-canon");
  assertEquals(meta?.authority, "canonical");
});

Deno.test("TieredMemory - demote to warm", () => {
  const tm = new TieredMemory();
  const msg = makeMessage("m-1");
  tm.addMessage(msg, 0.3);

  const summary: MessageSummary = {
    summaryId: "s-1",
    originalMessageId: "m-1",
    conversationId: "conv-1",
    role: "user",
    summary: "Summary of m-1",
    keyEntities: [],
    createdAt: Math.floor(Date.now() / 1000),
  };

  tm.demoteToWarm("m-1", summary);
  const meta = tm.getTierMetadata("m-1");
  assertEquals(meta?.tier, "warm");
});

Deno.test("TieredMemory - demote to cold", () => {
  const tm = new TieredMemory();
  const msg = makeMessage("m-1");
  tm.addMessage(msg, 0.1);

  const summary: MessageSummary = {
    summaryId: "s-1",
    originalMessageId: "m-1",
    conversationId: "conv-1",
    role: "user",
    summary: "Summary",
    keyEntities: [],
    createdAt: Math.floor(Date.now() / 1000),
  };
  tm.demoteToWarm("m-1", summary);

  const fact: KeyFact = {
    factId: "f-1",
    originalMessageIds: ["m-1"],
    conversationId: "conv-1",
    fact: "Key fact",
    factType: "other",
    createdAt: Math.floor(Date.now() / 1000),
  };
  tm.demoteToCold("s-1", fact);

  // Stats should reflect the moves
  const stats = tm.getStats();
  assertEquals(stats.hotCount, 1); // hot message still tracked
  assertEquals(stats.coldCount, 1);
});

Deno.test("TieredMemory - promote to hot", () => {
  const tm = new TieredMemory();
  const msg = makeMessage("m-1");
  tm.addMessage(msg, 0.5);

  const summary: MessageSummary = {
    summaryId: "s-1",
    originalMessageId: "m-1",
    conversationId: "conv-1",
    role: "user",
    summary: "Summary",
    keyEntities: [],
    createdAt: Math.floor(Date.now() / 1000),
  };
  tm.demoteToWarm("m-1", summary);
  assertEquals(tm.getTierMetadata("m-1")?.tier, "warm");

  tm.promoteToHot("m-1");
  assertEquals(tm.getTierMetadata("m-1")?.tier, "hot");
});

Deno.test("TieredMemory - get demotion candidates", () => {
  const tm = new TieredMemory();
  // Add 3 messages with decreasing importance
  tm.addMessage(makeMessage("high"), 0.9);
  tm.addMessage(makeMessage("med"), 0.5);
  tm.addMessage(makeMessage("low"), 0.1);

  const candidates = tm.getDemotionCandidates("hot", 2);
  assertEquals(candidates.length, 2);
  // Lowest retention score first -> "low" should be first candidate
  assertEquals(candidates[0], "low");
});

Deno.test("TieredMemory - evict expired messages", () => {
  const config = defaultTieredMemoryConfig();
  config.sessionTtlSecs = 0; // Expire immediately
  const tm = new TieredMemory(config);

  tm.addMessage(makeMessage("m-expire"), 0.5);

  // The message should already be expired (ttl=0)
  const evicted = tm.evictExpired();
  assertEquals(evicted, 1);
  assertEquals(tm.getHotMessage("m-expire"), undefined);
});

Deno.test("TieredMemory - canonical messages survive eviction", () => {
  const config = defaultTieredMemoryConfig();
  config.sessionTtlSecs = 0;
  const tm = new TieredMemory(config);

  // Session message will get TTL
  tm.addMessage(makeMessage("m-session"), 0.5);
  // Canonical message gets no TTL
  tm.addCanonicalMessage(makeMessage("m-canon"), 0.9);

  const evicted = tm.evictExpired();
  assertEquals(evicted, 1); // Only session message evicted
  assertEquals(tm.getHotMessage("m-canon")?.messageId, "m-canon");
});

Deno.test("TieredMemory - fallback summarize", () => {
  const tm = new TieredMemory();
  const short = "Short message";
  assertEquals(tm.fallbackSummarize(short), short);

  const long = Array.from({ length: 100 }, (_, i) => `word${i}`).join(" ");
  const summarized = tm.fallbackSummarize(long);
  assertEquals(summarized.endsWith("..."), true);
  // 75 words joined by spaces, last word has "..." appended (no extra space)
  assertEquals(summarized.split(" ").length, 75);
});

Deno.test("TieredMemory - stats", () => {
  const tm = new TieredMemory();
  tm.addMessage(makeMessage("m-1"), 0.5);
  tm.addMessage(makeMessage("m-2"), 0.5);

  const stats = tm.getStats();
  assertEquals(stats.hotCount, 2);
  assertEquals(stats.warmCount, 0);
  assertEquals(stats.coldCount, 0);
  assertEquals(stats.totalTracked, 2);
});
