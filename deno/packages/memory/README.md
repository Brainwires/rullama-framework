# @rullama/memory

Tiered memory orchestration for the rullama: a hot/warm/cold hierarchy for
conversation history with retention scoring and multi-factor retrieval ranking.
Equivalent to the Rust `rullama-memory` crate.

- **Hot** holds full messages (`MessageMetadata` from `@rullama/stores`).
- **Warm** holds compressed `MessageSummary` records.
- **Cold** holds distilled `KeyFact` records.

Messages flow hot → warm → cold as they age or lose importance and are promoted
back up when accessed. `TieredMemory` is an **in-process** implementation: it
keeps every tier in `Map`s, enforces the capacity and TTL rules, and ranks
demotion candidates. It does not persist anything — pair it with a
`StorageBackend` from `@rullama/storage` and the domain stores in
`@rullama/stores` when you need durable tiers.

## Install

```sh
deno add jsr:@rullama/memory
```

## Quick Example

```ts
import {
  computeMultiFactorScore,
  type MessageMetadata,
  recencyFromHours,
  TieredMemory,
} from "@rullama/memory";

const memory = new TieredMemory({
  hotRetentionHours: 24,
  warmRetentionHours: 168,
  hotImportanceThreshold: 0.3,
  warmImportanceThreshold: 0.1,
  maxHotMessages: 1000,
  maxWarmSummaries: 5000,
  sessionTtlSecs: 3600, // hot messages expire after an hour unless canonical
});

const now = Math.floor(Date.now() / 1000);
const message: MessageMetadata = {
  messageId: "msg-1",
  conversationId: "conv-1",
  role: "user",
  content: "We decided to ship the parser rewrite in v2.",
  createdAt: now,
};

memory.addMessage(message, 0.8);
memory.recordAccess("msg-1");

// Lowest retention score first -- these are the messages to summarize next.
const [demote] = memory.getDemotionCandidates("hot", 1);
if (demote) {
  memory.demoteToWarm(demote, {
    summaryId: "sum-1",
    originalMessageId: demote,
    conversationId: "conv-1",
    role: "user",
    summary: memory.fallbackSummarize(message.content),
    keyEntities: ["parser rewrite", "v2"],
    createdAt: now,
  });
}

// Rank a retrieval hit by similarity, recency, and importance (0.5/0.3/0.2).
const score = computeMultiFactorScore(0.91, recencyFromHours(2), 0.8);
console.log(score.combined, memory.getStats());
```

## API

| Export                                                 | Description                                                                   |
| ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `TieredMemory`                                         | In-process tier manager: add/promote/demote messages, TTL eviction, stats     |
| `TieredMemoryConfig` / `defaultTieredMemoryConfig`     | Retention hours, importance thresholds, tier capacities, optional session TTL |
| `TierMetadata` / `createTierMetadata` / `recordAccess` | Per-message tier, importance, access count, and authority tracking            |
| `retentionScore`                                       | Importance + recency + access-count score; lowest is demoted first            |
| `MultiFactorScore` / `computeMultiFactorScore`         | Retrieval ranking blend of similarity, recency, and importance                |
| `recencyFromHours`                                     | Exponential recency decay from hours since last access                        |
| `MemoryTier` / `demoteTier` / `promoteTier`            | The `"hot" \| "warm" \| "cold"` ladder and its neighbours                     |
| `MemoryAuthority` / `parseMemoryAuthority`             | `"ephemeral" \| "session" \| "canonical"`; canonical entries survive TTL      |
| `MessageSummary`, `KeyFact`, `FactType`                | Warm- and cold-tier record shapes                                             |
| `TieredSearchResult`                                   | Shape of a cross-tier retrieval hit                                           |
| `MessageMetadata`                                      | Hot-tier message record (alias of `@rullama/stores`'s type)                   |
