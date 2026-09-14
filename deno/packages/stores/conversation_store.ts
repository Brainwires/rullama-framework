/**
 * Conversation Store -- persists conversation metadata.
 *
 * Equivalent to Rust's `stores/conversation_store.rs` in rullama-storage.
 * @module
 */

import type { StorageBackend } from "@rullama/storage";
import {
  type FieldDef,
  FieldTypes,
  fieldValueAsI32,
  fieldValueAsI64,
  fieldValueAsStr,
  FieldValues,
  Filters,
  optionalField,
  type Record,
  recordGet,
  requiredField,
} from "@rullama/storage";

const TABLE_NAME = "conversations";

/** Metadata for a conversation. */
export interface ConversationMetadata {
  /** Unique conversation identifier; the key every lookup and delete uses. */
  conversationId: string;
  /** Human-readable title, if one has been set. */
  title?: string;
  /** Identifier of the model the conversation runs on, if recorded at creation. */
  modelId?: string;
  /** Creation time as a Unix timestamp in seconds. */
  createdAt: number;
  /** Last-modification time as a Unix timestamp in seconds; bumped by every `update`. */
  updatedAt: number;
  /** Number of messages in the conversation, as reported by the caller (not derived from the message store). */
  messageCount: number;
}

function tableSchema(): FieldDef[] {
  return [
    requiredField("conversation_id", FieldTypes.Utf8),
    optionalField("title", FieldTypes.Utf8),
    optionalField("model_id", FieldTypes.Utf8),
    requiredField("created_at", FieldTypes.Int64),
    requiredField("updated_at", FieldTypes.Int64),
    requiredField("message_count", FieldTypes.Int32),
  ];
}

function toRecord(m: ConversationMetadata): Record {
  return [
    ["conversation_id", FieldValues.Utf8(m.conversationId)],
    ["title", FieldValues.Utf8(m.title ?? null)],
    ["model_id", FieldValues.Utf8(m.modelId ?? null)],
    ["created_at", FieldValues.Int64(m.createdAt)],
    ["updated_at", FieldValues.Int64(m.updatedAt)],
    ["message_count", FieldValues.Int32(m.messageCount)],
  ];
}

function fromRecord(r: Record): ConversationMetadata {
  const conversationId = recordGet(r, "conversation_id");
  const createdAt = recordGet(r, "created_at");
  const updatedAt = recordGet(r, "updated_at");
  const messageCount = recordGet(r, "message_count");

  if (!conversationId || !createdAt || !updatedAt || !messageCount) {
    throw new Error("Missing required fields in conversation record");
  }

  return {
    conversationId: fieldValueAsStr(conversationId)!,
    title: recordGet(r, "title")
      ? fieldValueAsStr(recordGet(r, "title")!)
      : undefined,
    modelId: recordGet(r, "model_id")
      ? fieldValueAsStr(recordGet(r, "model_id")!)
      : undefined,
    createdAt: fieldValueAsI64(createdAt)!,
    updatedAt: fieldValueAsI64(updatedAt)!,
    messageCount: fieldValueAsI32(messageCount)!,
  };
}

/** Interface for conversation store operations. */
export interface ConversationStoreI {
  /** Create the backing `conversations` table if it does not already exist. */
  ensureTable(): Promise<void>;
  /**
   * Create a conversation, or — when one with this id already exists — update
   * its `title` / `messageCount` instead (the existing `modelId` is kept).
   * @param conversationId Id of the conversation to create.
   * @param title Optional title.
   * @param modelId Optional model identifier (only stored on a fresh create).
   * @param messageCount Initial message count (default 0).
   * @returns The metadata as stored after the create or update.
   */
  create(
    conversationId: string,
    title?: string,
    modelId?: string,
    messageCount?: number,
  ): Promise<ConversationMetadata>;
  /** Look up a conversation by id; `undefined` when none has that id. */
  get(conversationId: string): Promise<ConversationMetadata | undefined>;
  /**
   * List conversations, most recently updated first.
   * @param limit Maximum number to return; omit for all.
   */
  list(limit?: number): Promise<ConversationMetadata[]>;
  /**
   * Change `title` and/or `messageCount` of an existing conversation and set
   * `updatedAt` to now; an omitted argument keeps the current value.
   * Rejects with `"Conversation not found"` when the id is unknown.
   */
  update(
    conversationId: string,
    title?: string,
    messageCount?: number,
  ): Promise<void>;
  /** Delete the conversation with the given id (no-op when absent). */
  delete(conversationId: string): Promise<void>;
}

/** Store for managing conversations. */
export class ConversationStore implements ConversationStoreI {
  /**
   * Create a store over `backend`; call {@link ensureTable} before use.
   * @param backend Storage backend that holds the `conversations` table.
   */
  constructor(private readonly backend: StorageBackend) {}

  /** Create the `conversations` table on the backend if it is missing. */
  async ensureTable(): Promise<void> {
    await this.backend.ensureTable(TABLE_NAME, tableSchema());
  }

  /**
   * Insert a new conversation stamped with the current time, or, if the id
   * already exists, delegate to {@link update} and return the re-read record.
   */
  async create(
    conversationId: string,
    title?: string,
    modelId?: string,
    messageCount?: number,
  ): Promise<ConversationMetadata> {
    // Check if conversation already exists - if so, just update
    const existing = await this.get(conversationId);
    if (existing) {
      await this.update(conversationId, title ?? existing.title, messageCount);
      const updated = await this.get(conversationId);
      if (!updated) throw new Error("Conversation should exist after update");
      return updated;
    }

    const now = Math.floor(Date.now() / 1000);
    const metadata: ConversationMetadata = {
      conversationId,
      title,
      modelId,
      createdAt: now,
      updatedAt: now,
      messageCount: messageCount ?? 0,
    };

    await this.backend.insert(TABLE_NAME, [toRecord(metadata)]);
    return metadata;
  }

  /** Query the backend for the single record whose `conversation_id` matches. */
  async get(conversationId: string): Promise<ConversationMetadata | undefined> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    const records = await this.backend.query(TABLE_NAME, filter, 1);
    return records.length > 0 ? fromRecord(records[0]) : undefined;
  }

  /**
   * Load every conversation from the backend, sort by `updatedAt` descending,
   * then truncate to `limit` when given.
   */
  async list(limit?: number): Promise<ConversationMetadata[]> {
    const records = await this.backend.query(TABLE_NAME);
    let conversations = records.map(fromRecord);
    conversations.sort((a, b) => b.updatedAt - a.updatedAt);
    if (limit !== undefined) {
      conversations = conversations.slice(0, limit);
    }
    return conversations;
  }

  /**
   * Rewrite the conversation's record with the new `title` / `messageCount`
   * and a fresh `updatedAt`. Implemented as a delete followed by an insert,
   * so the two steps are not atomic on the backend.
   */
  async update(
    conversationId: string,
    title?: string,
    messageCount?: number,
  ): Promise<void> {
    const current = await this.get(conversationId);
    if (!current) throw new Error("Conversation not found");

    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(TABLE_NAME, filter);

    const updated: ConversationMetadata = {
      conversationId,
      title: title ?? current.title,
      modelId: current.modelId,
      createdAt: current.createdAt,
      updatedAt: Math.floor(Date.now() / 1000),
      messageCount: messageCount ?? current.messageCount,
    };

    await this.backend.insert(TABLE_NAME, [toRecord(updated)]);
  }

  /** Delete every record whose `conversation_id` matches. */
  async delete(conversationId: string): Promise<void> {
    const filter = Filters.Eq(
      "conversation_id",
      FieldValues.Utf8(conversationId),
    );
    await this.backend.delete(TABLE_NAME, filter);
  }
}

/** In-memory conversation store for testing. */
export class InMemoryConversationStore implements ConversationStoreI {
  private conversations: Map<string, ConversationMetadata> = new Map();

  /** No-op: there is no table to create in memory. */
  async ensureTable(): Promise<void> {
    await Promise.resolve();
  }

  /**
   * Store a new conversation stamped with the current time, or, if the id
   * already exists, apply {@link update} and return the stored record.
   */
  async create(
    conversationId: string,
    title?: string,
    modelId?: string,
    messageCount?: number,
  ): Promise<ConversationMetadata> {
    const existing = this.conversations.get(conversationId);
    if (existing) {
      await this.update(conversationId, title ?? existing.title, messageCount);
      return this.conversations.get(conversationId)!;
    }

    const now = Math.floor(Date.now() / 1000);
    const metadata: ConversationMetadata = {
      conversationId,
      title,
      modelId,
      createdAt: now,
      updatedAt: now,
      messageCount: messageCount ?? 0,
    };
    this.conversations.set(conversationId, metadata);
    return await Promise.resolve(metadata);
  }

  /** Return the stored record for `conversationId`, if any. */
  async get(conversationId: string): Promise<ConversationMetadata | undefined> {
    return await Promise.resolve(this.conversations.get(conversationId));
  }

  /** All stored conversations sorted by `updatedAt` descending, truncated to `limit` when given. */
  async list(limit?: number): Promise<ConversationMetadata[]> {
    let result = [...this.conversations.values()];
    result.sort((a, b) => b.updatedAt - a.updatedAt);
    if (limit !== undefined) result = result.slice(0, limit);
    return await Promise.resolve(result);
  }

  /**
   * Replace the stored record with a copy carrying the new `title` /
   * `messageCount` and a fresh `updatedAt`; throws when the id is unknown.
   */
  async update(
    conversationId: string,
    title?: string,
    messageCount?: number,
  ): Promise<void> {
    const current = this.conversations.get(conversationId);
    if (!current) throw new Error("Conversation not found");

    this.conversations.set(conversationId, {
      ...current,
      title: title ?? current.title,
      messageCount: messageCount ?? current.messageCount,
      updatedAt: Math.floor(Date.now() / 1000),
    });
    await Promise.resolve();
  }

  /** Remove the conversation from the map (no-op when absent). */
  async delete(conversationId: string): Promise<void> {
    this.conversations.delete(conversationId);
    await Promise.resolve();
  }
}
