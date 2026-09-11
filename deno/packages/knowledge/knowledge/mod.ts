/**
 * Knowledge module -- Brain client, thoughts, entities, relationships.
 *
 * Provides types for the Open Brain knowledge system: thought capture,
 * entity extraction, relationship graphs, and BKS/PKS knowledge stores.
 */

// ---------------------------------------------------------------------------
// ThoughtCategory
// ---------------------------------------------------------------------------

/** Category of a thought, used for filtering and organisation. */
export type ThoughtCategory =
  | "decision"
  | "person"
  | "insight"
  | "meeting_note"
  | "idea"
  | "action_item"
  | "reference"
  | "general";

/** All ThoughtCategory values. */
export const ALL_THOUGHT_CATEGORIES: readonly ThoughtCategory[] = [
  "decision",
  "person",
  "insight",
  "meeting_note",
  "idea",
  "action_item",
  "reference",
  "general",
] as const;

/** Parse a string into a ThoughtCategory, defaulting to "general". */
export function parseThoughtCategory(s: string): ThoughtCategory {
  const lower = s.toLowerCase();
  switch (lower) {
    case "decision":
      return "decision";
    case "person":
      return "person";
    case "insight":
      return "insight";
    case "meeting_note":
    case "meetingnote":
      return "meeting_note";
    case "idea":
      return "idea";
    case "action_item":
    case "actionitem":
    case "todo":
      return "action_item";
    case "reference":
    case "ref":
      return "reference";
    default:
      return "general";
  }
}

// ---------------------------------------------------------------------------
// ThoughtSource
// ---------------------------------------------------------------------------

/** How a thought was captured. */
export type ThoughtSource =
  | "manual"
  | "conversation"
  | "import";

/** Parse a string into a ThoughtSource, defaulting to "manual". */
export function parseThoughtSource(s: string): ThoughtSource {
  const lower = s.toLowerCase();
  switch (lower) {
    case "manual":
    case "manual_capture":
      return "manual";
    case "conversation":
    case "conversation_extract":
      return "conversation";
    case "import":
      return "import";
    default:
      return "manual";
  }
}

// ---------------------------------------------------------------------------
// Thought
// ---------------------------------------------------------------------------

/** A persistent thought stored in the Open Brain. */
export interface Thought {
  /** Unique identifier (UUID). */
  id: string;
  /** The thought content text. */
  content: string;
  /** Category for filtering and organisation. */
  category: ThoughtCategory;
  /** User-provided or auto-extracted tags. */
  tags: string[];
  /** How the thought was captured. */
  source: ThoughtSource;
  /** Importance score in 0.0--1.0. */
  importance: number;
  /** Unix timestamp of creation. */
  createdAt: number;
  /** Unix timestamp of last update. */
  updatedAt: number;
  /** Soft-delete flag. */
  deleted: boolean;
}

/** Create a new Thought with defaults. */
export function createThought(content: string): Thought {
  const now = Math.floor(Date.now() / 1000);
  return {
    id: crypto.randomUUID(),
    content,
    category: "general",
    tags: [],
    source: "manual",
    importance: 0.5,
    createdAt: now,
    updatedAt: now,
    deleted: false,
  };
}

// ---------------------------------------------------------------------------
// EntityType (mirrors rullama-core graph::EntityType)
// ---------------------------------------------------------------------------

/** Entity types for the knowledge graph. */
export type EntityType =
  | "file"
  | "function"
  | "type"
  | "variable"
  | "concept"
  | "error"
  | "command";

// ---------------------------------------------------------------------------
// Entity
// ---------------------------------------------------------------------------

/** A named entity extracted from conversation. */
export interface Entity {
  /** Display name of the entity. */
  name: string;
  /** The kind of entity. */
  entityType: EntityType;
  /** Message IDs where this entity appears. */
  messageIds: string[];
  /** Unix timestamp when first seen. */
  firstSeen: number;
  /** Unix timestamp when last seen. */
  lastSeen: number;
  /** Total number of mentions. */
  mentionCount: number;
}

// ---------------------------------------------------------------------------
// Relationship
// ---------------------------------------------------------------------------

/** Relationship between entities (discriminated union). */
export type Relationship =
  | { kind: "Defines"; definer: string; defined: string; context: string }
  | { kind: "References"; from: string; to: string }
  | {
    kind: "Modifies";
    modifier: string;
    modified: string;
    changeType: string;
  }
  | { kind: "DependsOn"; dependent: string; dependency: string }
  | { kind: "Contains"; container: string; contained: string }
  | {
    kind: "CoOccurs";
    entityA: string;
    entityB: string;
    messageId: string;
  };

// ---------------------------------------------------------------------------
// ExtractionResult
// ---------------------------------------------------------------------------

/** Extraction result from a single message. */
export interface ExtractionResult {
  /** Extracted entities as [name, type] pairs. */
  entities: [string, EntityType][];
  /** Extracted relationships between entities. */
  relationships: Relationship[];
}

// ---------------------------------------------------------------------------
// Contradiction detection
// ---------------------------------------------------------------------------

/** Why two stored facts were flagged as a potential contradiction. */
export type ContradictionKind =
  | "ConflictingDefinition"
  | "ConflictingModification";

/** A potential contradiction detected when inserting a new fact. */
export interface ContradictionEvent {
  /** What kind of contradiction was detected. */
  kind: ContradictionKind;
  /** The entity key involved. */
  subject: string;
  /** Context string from the previously stored relationship. */
  existingContext: string;
  /** Context string from the newly inserted relationship. */
  newContext: string;
}

// ---------------------------------------------------------------------------
// BrainClient interface (stub -- concrete implementations need storage)
// ---------------------------------------------------------------------------

/** Request to capture a new thought. */
export interface CaptureThoughtRequest {
  /** Text of the thought to store. */
  content: string;
  /** Category name; parsed with {@link parseThoughtCategory}, auto-detected when omitted. */
  category?: string;
  /** Tags to attach; implementations may add auto-extracted ones. */
  tags?: string[];
  /** Importance score in 0.0--1.0 (implementations default to 0.5). */
  importance?: number;
  /** Capture source name; parsed with {@link parseThoughtSource}, defaults to `"manual"`. */
  source?: string;
}

/** Response after capturing a thought. */
export interface CaptureThoughtResponse {
  /** UUID assigned to the stored thought. */
  id: string;
  /** Category the thought was filed under (given or detected). */
  category: string;
  /** Final tag list on the stored thought. */
  tags: string[];
  /** Importance score in 0.0--1.0 that was stored. */
  importance: number;
  /** Number of PKS facts extracted from the content during capture. */
  factsExtracted: number;
}

/** Request to search memory. */
export interface SearchMemoryRequest {
  /** Natural-language query to embed and match semantically. */
  query: string;
  /** Maximum number of results to return. */
  limit?: number;
  /** Drop results whose similarity score is below this threshold. */
  minScore?: number;
  /** Restrict to thoughts in this category. */
  category?: string;
  /** Restrict to these result sources (e.g. `"thoughts"`, `"pks"`). */
  sources?: string[];
}

/** Response from memory search. */
export interface SearchMemoryResponse {
  /** Matching entries, best score first. */
  results: MemorySearchResult[];
  /** Number of entries in `results`. */
  total: number;
}

/** A single memory search result. */
export interface MemorySearchResult {
  /** Matched text (thought content or fact value). */
  content: string;
  /** Similarity score, higher is a closer match. */
  score: number;
  /** Which store produced the hit (e.g. `"thoughts"` or `"pks"`). */
  source: string;
  /** UUID of the originating thought, when the hit is a thought. */
  thoughtId?: string;
  /** Category of the originating thought or fact. */
  category?: string;
  /** Tags of the originating thought. */
  tags?: string[];
  /** Unix timestamp (seconds) when the entry was created. */
  createdAt?: number;
}

/** Request to list recent thoughts. */
export interface ListRecentRequest {
  /** Maximum number of thoughts to return. */
  limit?: number;
  /** Only list thoughts in this category. */
  category?: string;
  /** Only list thoughts created after this ISO 8601 timestamp. */
  since?: string;
}

/** Response from listing recent thoughts. */
export interface ListRecentResponse {
  /** Thought summaries, newest first. */
  thoughts: ThoughtSummary[];
  /** Number of entries in `thoughts`. */
  total: number;
}

/** Summary of a thought for listing. */
export interface ThoughtSummary {
  /** Thought UUID. */
  id: string;
  /** Thought text. */
  content: string;
  /** Category name. */
  category: string;
  /** Attached tags. */
  tags: string[];
  /** Importance score in 0.0--1.0. */
  importance: number;
  /** Unix timestamp (seconds) of creation. */
  createdAt: number;
}

/** Request to get a single thought. */
export interface GetThoughtRequest {
  /** UUID of the thought to fetch. */
  id: string;
}

/** Response containing a full thought. */
export interface GetThoughtResponse {
  /** Thought UUID. */
  id: string;
  /** Thought text. */
  content: string;
  /** Category name. */
  category: string;
  /** Attached tags. */
  tags: string[];
  /** How the thought was captured (see {@link ThoughtSource}). */
  source: string;
  /** Importance score in 0.0--1.0. */
  importance: number;
  /** Unix timestamp (seconds) of creation. */
  createdAt: number;
  /** Unix timestamp (seconds) of the last update. */
  updatedAt: number;
}

/** Request to search knowledge (PKS/BKS). */
export interface SearchKnowledgeRequest {
  /** Query text matched against fact keys and values. */
  query: string;
  /** Which store to search: `"pks"`, `"bks"`, or both when omitted. */
  source?: string;
  /** Restrict to facts in this category. */
  category?: string;
  /** Drop facts whose confidence is below this threshold (0.0--1.0). */
  minConfidence?: number;
  /** Maximum number of results to return. */
  limit?: number;
}

/** Response from knowledge search. */
export interface SearchKnowledgeResponse {
  /** Matching facts, highest confidence first. */
  results: KnowledgeResult[];
  /** Number of entries in `results`. */
  total: number;
}

/** A single knowledge search result. */
export interface KnowledgeResult {
  /** Store the fact came from (`"pks"` or `"bks"`). */
  source: string;
  /** Fact category. */
  category: string;
  /** Fact key (the subject, e.g. an entity name). */
  key: string;
  /** Fact value (what is known about the key). */
  value: string;
  /** Confidence in 0.0--1.0 that the fact is correct. */
  confidence: number;
  /** Surrounding text the fact was extracted from. */
  context?: string;
}

/** Request to delete a thought. */
export interface DeleteThoughtRequest {
  /** UUID of the thought to soft-delete. */
  id: string;
}

/** Response after deleting a thought. */
export interface DeleteThoughtResponse {
  /** `true` when the thought existed and was marked deleted. */
  deleted: boolean;
  /** UUID of the thought the request targeted. */
  id: string;
}

/** Memory statistics. */
export interface MemoryStatsResponse {
  /** Counts over the thought store. */
  thoughts: ThoughtStats;
  /** Counts over the Personal Knowledge Store. */
  pks: PksStats;
  /** Counts over the Behavioral Knowledge Store. */
  bks: BksStats;
}

/** Thought store statistics. */
export interface ThoughtStats {
  /** Total non-deleted thoughts. */
  total: number;
  /** Thought count per category name. */
  byCategory: Record<string, number>;
  /** Thoughts created in the last 24 hours. */
  recent24h: number;
  /** Thoughts created in the last 7 days. */
  recent7d: number;
  /** Thoughts created in the last 30 days. */
  recent30d: number;
  /** Most-used tags as `[tag, count]` pairs, most used first. */
  topTags: [string, number][];
}

/** Personal Knowledge Store statistics. */
export interface PksStats {
  /** Total facts stored. */
  totalFacts: number;
  /** Fact count per category name. */
  byCategory: Record<string, number>;
  /** Mean confidence across all facts (0.0--1.0). */
  avgConfidence: number;
}

/** Behavioral Knowledge Store statistics. */
export interface BksStats {
  /** Total behavioral truths stored. */
  totalTruths: number;
  /** Truth count per category name. */
  byCategory: Record<string, number>;
}

/**
 * BrainClient interface -- central orchestrator for all Open Brain storage.
 *
 * Concrete implementations require a storage backend (e.g. LanceDB, Postgres)
 * and an embedding provider. This interface defines the public API contract.
 */
export interface BrainClient {
  /** Capture a new thought, embed it, detect category, extract PKS facts. */
  captureThought(
    req: CaptureThoughtRequest,
  ): Promise<CaptureThoughtResponse>;

  /** Semantic search across thoughts and optionally PKS facts. */
  searchMemory(req: SearchMemoryRequest): Promise<SearchMemoryResponse>;

  /** List recent thoughts, optionally filtered. */
  listRecent(req: ListRecentRequest): Promise<ListRecentResponse>;

  /** Get a single thought by ID. */
  getThought(id: string): Promise<GetThoughtResponse | null>;

  /** Search PKS and/or BKS knowledge stores. */
  searchKnowledge(
    req: SearchKnowledgeRequest,
  ): Promise<SearchKnowledgeResponse>;

  /** Get aggregate statistics across all memory stores. */
  memoryStats(): Promise<MemoryStatsResponse>;

  /** Soft-delete a thought by ID. */
  deleteThought(id: string): Promise<DeleteThoughtResponse>;
}
