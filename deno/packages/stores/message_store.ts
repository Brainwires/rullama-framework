/**
 * Message Store -- persists conversation messages with semantic search.
 *
 * Equivalent to Rust's `stores/message_store.rs` in rullama-storage.
 * @module
 */

import type { EmbeddingProvider } from "@rullama/core";
import type { StorageBackend } from "@rullama/storage";
import {
  type FieldDef,
  FieldTypes,
  fieldValueAsI32,
  fieldValueAsI64,
  fieldValueAsStr,
  FieldValues,
  type Filter,
  Filters,
  optionalField,
  type Record,
  recordGet,
  requiredField,
} from "@rullama/storage";

const TABLE_NAME = "messages";

/** Metadata for a message. */
export interface MessageMetadata {
  /** Unique message identifier; the key `get` and `delete` use. */
  messageId: string;
  /** Id of the conversation the message belongs to. */
  conversationId: string;
  /** Author role of the message (e.g. `"user"`, `"assistant"`, `"system"`). */
  role: string;
  /** Message text; this is what gets embedded for semantic search. */
  content: string;
  /** Token count of `content`, if the caller measured it. */
  tokenCount?: number;
  /** Identifier of the model that produced the message, if any. */
  modelId?: string;
  /** Image attachments serialized by the caller into one string (stored opaquely), if any. */
  images?: string;
  /** Creation time as a Unix timestamp in seconds. */
  createdAt: number;
  /** Expiry time as a Unix timestamp in seconds; once reached, `deleteExpired` removes the message. */
  expiresAt?: number;
}

function tableSchema(embeddingDim: number): FieldDef[] {
  return [
    requiredField("vector", FieldTypes.Vector(embeddingDim)),
    requiredField("message_id", FieldTypes.Utf8),
    requiredField("conversation_id", FieldTypes.Utf8),
    requiredField("role", FieldTypes.Utf8),
    requiredField("content", FieldTypes.Utf8),
    optionalField("token_count", FieldTypes.Int32),
    optionalField("model_id", FieldTypes.Utf8),
    optionalField("images", FieldTypes.Utf8),
    requiredField("created_at", FieldTypes.Int64),
    optionalField("expires_at", FieldTypes.Int64),
  ];
}

function toRecord(m: MessageMetadata, embedding: number[]): Record {
  return [
    ["vector", FieldValues.Vector(embedding)],
    ["message_id", FieldValues.Utf8(m.messageId)],
    ["conversation_id", FieldValues.Utf8(m.conversationId)],
    ["role", FieldValues.Utf8(m.role)],
    ["content", FieldValues.Utf8(m.content)],
    ["token_count", FieldValues.Int32(m.tokenCount ?? null)],
    ["model_id", FieldValues.Utf8(m.modelId ?? null)],
    ["images", FieldValues.Utf8(m.images ?? null)],
    ["created_at", FieldValues.Int64(m.createdAt)],
    ["expires_at", FieldValues.Int64(m.expiresAt ?? null)],
  ];
}

function fromRecord(r: Record): MessageMetadata {
  const messageId = recordGet(r, "message_id");
  const conversationId = recordGet(r, "conversation_id");
  const role = recordGet(r, "role");
  const content = recordGet(r, "content");
  const createdAt = recordGet(r, "created_at");

  if (!messageId || !conversationId || !role || !content || !createdAt) {
    throw new Error("Missing required fields in message record");
  }

  return {
    messageId: fieldValueAsStr(messageId)!,
    conversationId: fieldValueAsStr(conversationId)!,
    role: fieldValueAsStr(role)!,
    content: fieldValueAsStr(content)!,
    tokenCount: recordGet(r, "token_count")
      ? fieldValueAsI32(recordGet(r, "token_count")!)
      : undefined,
    modelId: recordGet(r, "model_id")
      ? fieldValueAsStr(recordGet(r, "model_id")!)
      : undefined,
    images: recordGet(r, "images")
      ? fieldValueAsStr(recordGet(r, "images")!)
      : undefined,
    createdAt: fieldValueAsI64(createdAt)!,
    expiresAt: recordGet(r, "expires_at")
      ? fieldValueAsI64(recordGet(r, "expires_at")!)
      : undefined,
  };
}

/** Interface for message store operations. */
export interface MessageStoreI {
  /** Create the backing `messages` table (including its embedding vector column) if it does not already exist. */
  ensureTable(): Promise<void>;
  /** Persist one message, embedding its `content` for semantic search. */
  add(message: MessageMetadata): Promise<void>;
  /** Persist several messages in one insert, embedding their contents in a single batch; a no-op for an empty array. */
  addBatch(messages: MessageMetadata[]): Promise<void>;
  /** Look up a message by id; `undefined` when none has that id. */
  get(messageId: string): Promise<MessageMetadata | undefined>;
  /** Every message in a conversation, in the order the backend returns them (no sorting applied). */
  getByConversation(conversationId: string): Promise<MessageMetadata[]>;
  /**
   * Semantic search over all messages.
   * @param query Text that is embedded and compared against stored message vectors.
   * @param limit Maximum number of hits requested from the backend.
   * @param minScore Hits scoring below this similarity are dropped.
   * @returns `[message, similarityScore]` pairs in the backend's ranking order.
   */
  search(
    query: string,
    limit: number,
    minScore: number,
  ): Promise<[MessageMetadata, number][]>;
  /**
   * Semantic search restricted to one conversation.
   * @param conversationId Only messages of this conversation are candidates.
   * @param query Text that is embedded and compared against stored message vectors.
   * @param limit Maximum number of hits requested from the backend.
   * @param minScore Hits scoring below this similarity are dropped.
   * @returns `[message, similarityScore]` pairs in the backend's ranking order.
   */
  searchConversation(
    conversationId: string,
    query: string,
    limit: number,
    minScore: number,
  ): Promise<[MessageMetadata, number][]>;
  /** Delete every message belonging to a conversation. */
  deleteByConversation(conversationId: string): Promise<void>;
  /** Delete one message by id (no-op when absent). */
  delete(messageId: string): Promise<void>;
  /**
   * Delete every message whose `expiresAt` is set and not later than now.
   * @returns The number of messages removed.
   */
  deleteExpired(): Promise<number>;
}

/** Store for managing messages with semantic search. */
export class MessageStore implements MessageStoreI {
  /**
   * Create a store over `backend`; call {@link ensureTable} before use.
   * @param backend Storage backend that holds the `messages` table.
   * @param embeddings Provider used to embed message content and search
   *   queries; its `dimension` fixes the width of the table's vector column.
   */
  constructor(
    private readonly backend: StorageBackend,
    private readonly embeddings: EmbeddingProvider,
  ) {}

  /** Create the `messages` table on the backend, sized to the embedding provider's `dimension`. */
  async ensureTable(): Promise<void> {
    await this.backend.ensureTable(
      TABLE_NAME,
      tableSchema(this.embeddings.dimension),
    );
  }

  /** Embed `message.content` and insert the message as one record. */
  async add(message: MessageMetadata): Promise<void> {
    const embedding = await this.embeddings.embed(message.content);
    const record = toRecord(message, embedding);
    await this.backend.insert(TABLE_NAME, [record]);
  }

  /** Embed all contents with one `embedBatch` call and insert the records together; returns early for an empty array. */
  async addBatch(messages: MessageMetadata[]): Promise<void> {
    if (messages.length === 0) return;
    const contents = messages.map((m) => m.content);
    const embeddings = await this.embeddings.embedBatch(contents);
    const records = messages.map((m, i) => toRecord(m, embeddings[i]));
    await this.backend.insert(TABLE_NAME, records);
  }

  /** Query the backend for the single record whose `message_id` matches. */
  async get(messageId: string): Promise<MessageMetadata | undefined> {
    const filter = Filters.Eq("message_id", FieldValues.Utf8(messageId));
    const records = await this.backend.query(TABLE_NAME, filter, 1);
    return records.length > 0 ? fromRecord(records[0]) : undefined;
  }

  /** Query every record whose `conversation_id` matches, in backend order. */
  async getByConversation(conversationId: string): Promise<MessageMetadata[]> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    const records = await this.backend.query(TABLE_NAME, filter);
    return records.map(fromRecord);
  }

  /** Vector search over the whole table; see {@link searchWithFilter}. */
  // deno-lint-ignore require-await
  async search(
    query: string,
    limit: number,
    minScore: number,
  ): Promise<[MessageMetadata, number][]> {
    return this.searchWithFilter(query, limit, minScore);
  }

  /** Vector search filtered to one `conversation_id`; see {@link searchWithFilter}. */
  // deno-lint-ignore require-await
  async searchConversation(
    conversationId: string,
    query: string,
    limit: number,
    minScore: number,
  ): Promise<[MessageMetadata, number][]> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    return this.searchWithFilter(query, limit, minScore, filter);
  }

  /**
   * Embed `query`, run the backend's vector search on the `vector` column
   * (optionally narrowed by `filter`), and drop hits scoring below `minScore`.
   */
  private async searchWithFilter(
    query: string,
    limit: number,
    minScore: number,
    filter?: Filter,
  ): Promise<[MessageMetadata, number][]> {
    const queryEmbedding = await this.embeddings.embed(query);
    const scored = await this.backend.vectorSearch(
      TABLE_NAME,
      "vector",
      queryEmbedding,
      limit,
      filter,
    );

    return scored
      .filter((sr) => sr.score >= minScore)
      .map((sr) => [fromRecord(sr.record), sr.score]);
  }

  /** Delete every record whose `conversation_id` matches. */
  async deleteByConversation(conversationId: string): Promise<void> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(TABLE_NAME, filter);
  }

  /** Delete the record whose `message_id` matches. */
  async delete(messageId: string): Promise<void> {
    const filter = Filters.Eq("message_id", FieldValues.Utf8(messageId));
    await this.backend.delete(TABLE_NAME, filter);
  }

  /**
   * Count records with a non-null `expires_at` <= now, delete them if there
   * are any, and return that count.
   */
  async deleteExpired(): Promise<number> {
    const now = Math.floor(Date.now() / 1000);
    const filter = Filters.And([
      Filters.NotNull("expires_at"),
      Filters.Lte("expires_at", FieldValues.Int64(now)),
    ]);

    const count = await this.backend.count(TABLE_NAME, filter);
    if (count > 0) {
      await this.backend.delete(TABLE_NAME, filter);
    }
    return count;
  }
}

/**
 * In-memory MessageStore for testing (does not require an embedding provider
 * -- stores a zero vector).
 */
export class InMemoryMessageStore implements MessageStoreI {
  private messages: Map<string, MessageMetadata> = new Map();

  /** No-op: there is no table to create in memory. */
  async ensureTable(): Promise<void> {
    await Promise.resolve();
  }

  /** Store a shallow copy of the message keyed by `messageId`, replacing any existing one. */
  async add(message: MessageMetadata): Promise<void> {
    this.messages.set(message.messageId, { ...message });
    await Promise.resolve();
  }

  /** Store a shallow copy of each message keyed by `messageId`. */
  async addBatch(messages: MessageMetadata[]): Promise<void> {
    for (const m of messages) {
      this.messages.set(m.messageId, { ...m });
    }
    await Promise.resolve();
  }

  /** Return the stored message for `messageId`, if any. */
  async get(messageId: string): Promise<MessageMetadata | undefined> {
    return await Promise.resolve(this.messages.get(messageId));
  }

  /** Stored messages whose `conversationId` matches, in insertion order. */
  async getByConversation(conversationId: string): Promise<MessageMetadata[]> {
    return await Promise.resolve(
      [...this.messages.values()].filter((m) =>
        m.conversationId === conversationId
      ),
    );
  }

  /**
   * Returns the first `limit` stored messages, each with a score of 1.0; the
   * query and `minScore` are ignored because nothing is embedded in memory.
   */
  async search(
    _query: string,
    limit: number,
    _minScore: number,
  ): Promise<[MessageMetadata, number][]> {
    // Simple: return first N messages with score 1.0
    return await Promise.resolve(
      [...this.messages.values()].slice(0, limit).map((m) =>
        [m, 1.0] as [MessageMetadata, number]
      ),
    );
  }

  /**
   * Returns the first `limit` messages of the conversation, each with a score
   * of 1.0; the query and `minScore` are ignored.
   */
  async searchConversation(
    conversationId: string,
    _query: string,
    limit: number,
    _minScore: number,
  ): Promise<[MessageMetadata, number][]> {
    return await Promise.resolve(
      [...this.messages.values()]
        .filter((m) => m.conversationId === conversationId)
        .slice(0, limit)
        .map((m) => [m, 1.0] as [MessageMetadata, number]),
    );
  }

  /** Remove every stored message whose `conversationId` matches. */
  async deleteByConversation(conversationId: string): Promise<void> {
    for (const [id, m] of this.messages) {
      if (m.conversationId === conversationId) this.messages.delete(id);
    }
    await Promise.resolve();
  }

  /** Remove the message from the map (no-op when absent). */
  async delete(messageId: string): Promise<void> {
    this.messages.delete(messageId);
    await Promise.resolve();
  }

  /** Remove every message whose `expiresAt` is set and <= now; returns how many were removed. */
  async deleteExpired(): Promise<number> {
    const now = Math.floor(Date.now() / 1000);
    let count = 0;
    for (const [id, m] of this.messages) {
      if (m.expiresAt !== undefined && m.expiresAt <= now) {
        this.messages.delete(id);
        count++;
      }
    }
    return await Promise.resolve(count);
  }
}
