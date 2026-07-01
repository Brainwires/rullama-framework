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
  content: string;
  category?: string;
  tags?: string[];
  importance?: number;
  source?: string;
}

/** Response after capturing a thought. */
export interface CaptureThoughtResponse {
  id: string;
  category: string;
  tags: string[];
  importance: number;
  factsExtracted: number;
}

/** Request to search memory. */
export interface SearchMemoryRequest {
  query: string;
  limit?: number;
  minScore?: number;
  category?: string;
  sources?: string[];
}

/** Response from memory search. */
export interface SearchMemoryResponse {
  results: MemorySearchResult[];
  total: number;
}

/** A single memory search result. */
export interface MemorySearchResult {
  content: string;
  score: number;
  source: string;
  thoughtId?: string;
  category?: string;
  tags?: string[];
  createdAt?: number;
}

/** Request to list recent thoughts. */
export interface ListRecentRequest {
  limit?: number;
  category?: string;
  since?: string;
}

/** Response from listing recent thoughts. */
export interface ListRecentResponse {
  thoughts: ThoughtSummary[];
  total: number;
}

/** Summary of a thought for listing. */
export interface ThoughtSummary {
  id: string;
  content: string;
  category: string;
  tags: string[];
  importance: number;
  createdAt: number;
}

/** Request to get a single thought. */
export interface GetThoughtRequest {
  id: string;
}

/** Response containing a full thought. */
export interface GetThoughtResponse {
  id: string;
  content: string;
  category: string;
  tags: string[];
  source: string;
  importance: number;
  createdAt: number;
  updatedAt: number;
}

/** Request to search knowledge (PKS/BKS). */
export interface SearchKnowledgeRequest {
  query: string;
  source?: string;
  category?: string;
  minConfidence?: number;
  limit?: number;
}

/** Response from knowledge search. */
export interface SearchKnowledgeResponse {
  results: KnowledgeResult[];
  total: number;
}

/** A single knowledge search result. */
export interface KnowledgeResult {
  source: string;
  category: string;
  key: string;
  value: string;
  confidence: number;
  context?: string;
}

/** Request to delete a thought. */
export interface DeleteThoughtRequest {
  id: string;
}

/** Response after deleting a thought. */
export interface DeleteThoughtResponse {
  deleted: boolean;
  id: string;
}

/** Memory statistics. */
export interface MemoryStatsResponse {
  thoughts: ThoughtStats;
  pks: PksStats;
  bks: BksStats;
}

/** Thought store statistics. */
export interface ThoughtStats {
  total: number;
  byCategory: Record<string, number>;
  recent24h: number;
  recent7d: number;
  recent30d: number;
  topTags: [string, number][];
}

/** Personal Knowledge Store statistics. */
export interface PksStats {
  totalFacts: number;
  byCategory: Record<string, number>;
  avgConfidence: number;
}

/** Behavioral Knowledge Store statistics. */
export interface BksStats {
  totalTruths: number;
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
