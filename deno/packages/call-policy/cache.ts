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
 * Equivalent to Rust's `rullama_resilience::cache` module.
 */

import { ProviderDecorator } from "./decorator.ts";
import type {
  ChatOptions,
  ChatResponse,
  Message,
  Provider,
  Role,
  StreamChunk,
  Tool,
  Usage,
} from "@rullama/core";
import { Message as MessageClass } from "@rullama/core";

/** Key used to address a cached response. */
export interface CacheKey {
  /** Lower-case hex SHA-256 digest of the serialised call inputs. */
  value: string;
}

/** Wire representation of a cached response. */
export interface CachedResponse {
  /** Role of the cached message (normally `"assistant"`). */
  role: Role;
  /** Message payload as plain text (block messages are rendered to a string). */
  text: string;
  /** Token usage reported by the provider for the original call. */
  usage: Usage;
  /** Provider's finish reason for the original call, when it reported one. */
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
  return {
    message: msg,
    usage: { ...cr.usage },
    finish_reason: cr.finish_reason,
  };
}

/** Pluggable storage backend. */
export interface CacheBackend {
  /** Look up a cached response; resolves `null` on a miss. */
  get(key: CacheKey): Promise<CachedResponse | null>;
  /** Store (or overwrite) the response for `key`. Failures are swallowed by {@link CachedProvider}. */
  put(key: CacheKey, resp: CachedResponse): Promise<void>;
}

/** Default {@link MemoryCache} capacity. */
export const DEFAULT_MEMORY_CACHE_ENTRIES = 1000;

/**
 * In-memory cache — the default backend. Bounded: least-recently-used entries
 * are evicted once `maxEntries` is exceeded, so a long-running process with
 * many distinct prompts cannot grow without limit.
 */
export class MemoryCache implements CacheBackend {
  /** Entries in recency order — oldest first, so eviction pops from the front. */
  private readonly entries = new Map<string, CachedResponse>();
  /** Maximum number of entries retained before LRU eviction. */
  readonly maxEntries: number;

  /**
   * Create an empty LRU cache.
   *
   * @param maxEntries Capacity; must be a positive integer (throws otherwise).
   */
  constructor(maxEntries: number = DEFAULT_MEMORY_CACHE_ENTRIES) {
    if (!Number.isInteger(maxEntries) || maxEntries < 1) {
      throw new Error(
        `maxEntries must be a positive integer (got ${maxEntries})`,
      );
    }
    this.maxEntries = maxEntries;
  }

  /** Look up `key`, marking the entry most-recently-used on a hit. */
  get(key: CacheKey): Promise<CachedResponse | null> {
    const hit = this.entries.get(key.value);
    if (hit !== undefined) {
      // Refresh recency: Map iteration order is insertion order.
      this.entries.delete(key.value);
      this.entries.set(key.value, hit);
    }
    return Promise.resolve(hit ?? null);
  }

  /** Insert or refresh `key`, evicting least-recently-used entries past `maxEntries`. */
  put(key: CacheKey, resp: CachedResponse): Promise<void> {
    this.entries.delete(key.value);
    this.entries.set(key.value, resp);
    while (this.entries.size > this.maxEntries) {
      this.entries.delete(this.entries.keys().next().value as string);
    }
    return Promise.resolve();
  }

  /** Number of entries currently held. */
  size(): number {
    return this.entries.size;
  }

  /** `true` when no entries are held. */
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
  scope?: string,
): Promise<CacheKey> {
  const enc = new TextEncoder();
  const parts: Uint8Array[] = [];

  // A scope (tenant, user, session) keeps one caller's cached completions
  // from being served to another who sends the same prompt.
  if (scope !== undefined) parts.push(enc.encode(`\x00scope:${scope}`));

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
export class CachedProvider extends ProviderDecorator {
  /** Storage the responses are read from and written to. */
  readonly backend: CacheBackend;
  /** Cache-key scope (tenant / user / session); see {@link cacheKeyFor}. */
  readonly scope: string | undefined;

  /**
   * Wrap `inner` with a content-addressed response cache.
   *
   * @param inner Provider whose `chat` responses are cached.
   * @param backend Where responses are stored ({@link MemoryCache} or your own).
   * @param scope Partition the cache: entries written under one scope are
   *   never returned for another. Set it per tenant or per user whenever one
   *   provider instance serves more than one principal.
   */
  constructor(inner: Provider, backend: CacheBackend, scope?: string) {
    super(inner);
    this.backend = backend;
    this.scope = scope;
  }

  /** Convenience constructor using an in-memory backend. */
  static withMemoryCache(
    inner: Provider,
  ): { provider: CachedProvider; cache: MemoryCache } {
    const cache = new MemoryCache();
    return { provider: new CachedProvider(inner, cache), cache };
  }

  /**
   * Serve the response from the backend when the {@link cacheKeyFor} key hits;
   * otherwise forward to the wrapped provider and store the result. A failing
   * `backend.put` is ignored — the live response is still returned.
   */
  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    const key = await cacheKeyFor(messages, tools, options, this.scope);
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

  /** Pass-through: streaming is never cached (see the module docs for why). */
  streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    return this.inner.streamChat(messages, tools, options);
  }
}
