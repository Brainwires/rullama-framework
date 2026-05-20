/**
 * Response caching decorator.
 *
 * Wraps a Provider in a content-addressed cache so deterministic eval runs
 * are byte-reproducible and local development stops burning real tokens.
 * The cache key is a SHA-256 over the serialised inputs (tools sorted by
 * name, so reordering doesn't break hits).
 *
 * Streaming bypasses the cache — reconstructing a replayable event stream
 * from a single recorded response would fabricate data a caller cannot
 * distinguish from real model output.
 *
 * Equivalent to Rust's `brainwires_resilience::cache` module.
 */

import type {
  ChatOptions,
  ChatResponse,
  Message,
  Provider,
  Role,
  StreamChunk,
  Tool,
  Usage,
} from "@brainwires/core";
import { Message as MessageClass } from "@brainwires/core";

/** Key used to address a cached response. */
export interface CacheKey {
  value: string;
}

/** Wire representation of a cached response. */
export interface CachedResponse {
  role: Role;
  /** Message payload as plain text (block messages are rendered to a string). */
  text: string;
  usage: Usage;
  finish_reason?: string;
}

function cachedResponseFromChat(resp: ChatResponse): CachedResponse {
  const text = typeof resp.message.content === "string"
    ? resp.message.content
    : (resp.message.text() ?? "");
  return {
    role: resp.message.role,
    text,
    usage: { ...resp.usage },
    finish_reason: resp.finish_reason,
  };
}

function cachedResponseToChat(cr: CachedResponse): ChatResponse {
  const msg = cr.role === "assistant"
    ? MessageClass.assistant(cr.text)
    : cr.role === "system"
    ? MessageClass.system(cr.text)
    : MessageClass.user(cr.text);
  return { message: msg, usage: { ...cr.usage }, finish_reason: cr.finish_reason };
}

/** Pluggable storage backend. */
export interface CacheBackend {
  get(key: CacheKey): Promise<CachedResponse | null>;
  put(key: CacheKey, resp: CachedResponse): Promise<void>;
}

/** In-memory cache — the default backend. */
export class MemoryCache implements CacheBackend {
  private readonly entries = new Map<string, CachedResponse>();

  get(key: CacheKey): Promise<CachedResponse | null> {
    return Promise.resolve(this.entries.get(key.value) ?? null);
  }

  put(key: CacheKey, resp: CachedResponse): Promise<void> {
    this.entries.set(key.value, resp);
    return Promise.resolve();
  }

  size(): number {
    return this.entries.size;
  }

  isEmpty(): boolean {
    return this.entries.size === 0;
  }
}

function hexEncode(bytes: Uint8Array): string {
  const HEX = "0123456789abcdef";
  let out = "";
  for (const b of bytes) {
    out += HEX[(b >> 4) & 0xf] + HEX[b & 0xf];
  }
  return out;
}

/** Compute a stable cache key from the inputs to a chat() call. */
export async function cacheKeyFor(
  messages: Message[],
  tools: Tool[] | undefined,
  options: ChatOptions,
): Promise<CacheKey> {
  const enc = new TextEncoder();
  const parts: Uint8Array[] = [];

  // Serialise messages via Message.toJSON (skips undefined fields).
  const msgs_json = JSON.stringify(messages.map((m) => m.toJSON()));
  parts.push(enc.encode(msgs_json));

  if (tools && tools.length > 0) {
    const names = tools.map((t) => t.name).slice().sort();
    for (const n of names) {
      parts.push(enc.encode("\x00tool:"));
      parts.push(enc.encode(n));
    }
  }

  parts.push(enc.encode("\x00opts:"));
  parts.push(enc.encode(JSON.stringify(options.toJSON())));

  // Concatenate then hash.
  const total = parts.reduce((sum, p) => sum + p.length, 0);
  const buf = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    buf.set(p, off);
    off += p.length;
  }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", buf));
  return { value: hexEncode(digest) };
}

/** A Provider decorator that deduplicates identical chat() calls. */
export class CachedProvider implements Provider {
  readonly inner: Provider;
  readonly backend: CacheBackend;

  constructor(inner: Provider, backend: CacheBackend) {
    this.inner = inner;
    this.backend = backend;
  }

  /** Convenience constructor using an in-memory backend. */
  static withMemoryCache(inner: Provider): { provider: CachedProvider; cache: MemoryCache } {
    const cache = new MemoryCache();
    return { provider: new CachedProvider(inner, cache), cache };
  }

  get name(): string {
    return this.inner.name;
  }

  maxOutputTokens(): number {
    return this.inner.maxOutputTokens?.() ?? Infinity;
  }

  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    const key = await cacheKeyFor(messages, tools, options);
    const hit = await this.backend.get(key);
    if (hit !== null) {
      return cachedResponseToChat(hit);
    }
    const resp = await this.inner.chat(messages, tools, options);
    try {
      await this.backend.put(key, cachedResponseFromChat(resp));
    } catch {
      // Caching failures are non-fatal.
    }
    return resp;
  }

  streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    return this.inner.streamChat(messages, tools, options);
  }
}
