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
  messageId: string;
  conversationId: string;
  role: string;
  content: string;
  tokenCount?: number;
  modelId?: string;
  images?: string;
  createdAt: number;
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
  ensureTable(): Promise<void>;
  add(message: MessageMetadata): Promise<void>;
  addBatch(messages: MessageMetadata[]): Promise<void>;
  get(messageId: string): Promise<MessageMetadata | undefined>;
  getByConversation(conversationId: string): Promise<MessageMetadata[]>;
  search(
    query: string,
    limit: number,
    minScore: number,
  ): Promise<[MessageMetadata, number][]>;
  searchConversation(
    conversationId: string,
    query: string,
    limit: number,
    minScore: number,
  ): Promise<[MessageMetadata, number][]>;
  deleteByConversation(conversationId: string): Promise<void>;
  delete(messageId: string): Promise<void>;
  deleteExpired(): Promise<number>;
}

/** Store for managing messages with semantic search. */
export class MessageStore implements MessageStoreI {
  constructor(
    private readonly backend: StorageBackend,
    private readonly embeddings: EmbeddingProvider,
  ) {}

  async ensureTable(): Promise<void> {
    await this.backend.ensureTable(
      TABLE_NAME,
      tableSchema(this.embeddings.dimension),
    );
  }

  async add(message: MessageMetadata): Promise<void> {
    const embedding = await this.embeddings.embed(message.content);
    const record = toRecord(message, embedding);
    await this.backend.insert(TABLE_NAME, [record]);
  }

  async addBatch(messages: MessageMetadata[]): Promise<void> {
    if (messages.length === 0) return;
    const contents = messages.map((m) => m.content);
    const embeddings = await this.embeddings.embedBatch(contents);
    const records = messages.map((m, i) => toRecord(m, embeddings[i]));
    await this.backend.insert(TABLE_NAME, records);
  }

  async get(messageId: string): Promise<MessageMetadata | undefined> {
    const filter = Filters.Eq("message_id", FieldValues.Utf8(messageId));
    const records = await this.backend.query(TABLE_NAME, filter, 1);
    return records.length > 0 ? fromRecord(records[0]) : undefined;
  }

  async getByConversation(conversationId: string): Promise<MessageMetadata[]> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    const records = await this.backend.query(TABLE_NAME, filter);
    return records.map(fromRecord);
  }

  // deno-lint-ignore require-await
  async search(
    query: string,
    limit: number,
    minScore: number,
  ): Promise<[MessageMetadata, number][]> {
    return this.searchWithFilter(query, limit, minScore);
  }

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

  async deleteByConversation(conversationId: string): Promise<void> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(TABLE_NAME, filter);
  }

  async delete(messageId: string): Promise<void> {
    const filter = Filters.Eq("message_id", FieldValues.Utf8(messageId));
    await this.backend.delete(TABLE_NAME, filter);
  }

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

  async ensureTable(): Promise<void> {
    await Promise.resolve();
  }

  async add(message: MessageMetadata): Promise<void> {
    this.messages.set(message.messageId, { ...message });
    await Promise.resolve();
  }

  async addBatch(messages: MessageMetadata[]): Promise<void> {
    for (const m of messages) {
      this.messages.set(m.messageId, { ...m });
    }
    await Promise.resolve();
  }

  async get(messageId: string): Promise<MessageMetadata | undefined> {
    return await Promise.resolve(this.messages.get(messageId));
  }

  async getByConversation(conversationId: string): Promise<MessageMetadata[]> {
    return await Promise.resolve(
      [...this.messages.values()].filter((m) =>
        m.conversationId === conversationId
      ),
    );
  }

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

  async deleteByConversation(conversationId: string): Promise<void> {
    for (const [id, m] of this.messages) {
      if (m.conversationId === conversationId) this.messages.delete(id);
    }
    await Promise.resolve();
  }

  async delete(messageId: string): Promise<void> {
    this.messages.delete(messageId);
    await Promise.resolve();
  }

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
