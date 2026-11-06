/**
 * @module transport/admission
 *
 * Connection admission shared by every listener (socket listeners and
 * {@link StreamServerListener}) and by `McpServer.serveConnection`: the
 * first-line pre-auth protocol and the accepted-connection queue.
 */

import type { ConnectionInfo, ServerTransport } from "./traits.ts";

/**
 * Outcome of a {@link PreAuthHook}: admit the connection (optionally with an
 * identity that becomes `ConnectionInfo.identity`), or reject it.
 */
export type PreAuthResult =
  | { ok: true; identity?: unknown }
  | { ok: false; reason?: string };

/**
 * Pre-authentication hook run on the first line a client sends, before MCP
 * starts on the connection (e.g. a bearer token for TCP loopback). The line is
 * consumed; it is not passed to the MCP server. `null` means the client closed
 * (or timed out) before sending a line — reject it.
 */
export type PreAuthHook = (
  firstLine: string | null,
  info: ConnectionInfo,
) => PreAuthResult | Promise<PreAuthResult>;

let connectionCounter = 0;

/** A fresh process-unique connection id with the given prefix. */
export function nextConnectionId(prefix: string): string {
  connectionCounter += 1;
  return `${prefix}-${connectionCounter}`;
}

/** Default time a client has to send its pre-auth line. */
export const DEFAULT_PRE_AUTH_TIMEOUT_MS = 10_000;

/** JSON-RPC error line sent to a client whose pre-auth was rejected. */
export function unauthorizedLine(reason?: string): string {
  return JSON.stringify({
    jsonrpc: "2.0",
    id: null,
    error: {
      code: -32003,
      message: `Unauthorized${reason ? `: ${reason}` : ""}`,
    },
  });
}

/** The first line `transport` delivers, or `null` on EOF / after `ms`. */
export async function readFirstLine(
  transport: ServerTransport,
  ms: number,
): Promise<string | null> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), ms);
  });
  try {
    return await Promise.race([transport.readRequest(), timeout]);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Run `preAuth` on the connection's first line. On success the hook's
 * identity replaces `info.identity`; on rejection (or a throwing hook) the
 * client gets one Unauthorized line. Returns whether the connection is
 * admitted; the caller closes a rejected transport.
 */
export async function runPreAuth(
  transport: ServerTransport,
  info: ConnectionInfo,
  preAuth: PreAuthHook,
  timeoutMs: number = DEFAULT_PRE_AUTH_TIMEOUT_MS,
): Promise<boolean> {
  let verdict: PreAuthResult;
  try {
    verdict = await preAuth(await readFirstLine(transport, timeoutMs), info);
  } catch (e) {
    verdict = { ok: false, reason: String(e) };
  }
  if (verdict.ok) {
    if (verdict.identity !== undefined) info.identity = verdict.identity;
    return true;
  }
  await transport.writeResponse(unauthorizedLine(verdict.reason)).catch(
    () => {},
  );
  return false;
}

/**
 * Unbounded push/pull queue behind a listener's async iterator: `push` hands
 * an item to a waiting `next()` or buffers it; `end` finishes iteration and
 * returns the items nobody took (so the caller can close them).
 */
export class AcceptQueue<T> {
  #items: T[] = [];
  #waiters: ((v: IteratorResult<T>) => void)[] = [];
  #ended = false;

  /** Whether {@link end} was called. */
  get ended(): boolean {
    return this.#ended;
  }

  /** Deliver one item (ignored, and returned as `false`, after {@link end}). */
  push(item: T): boolean {
    if (this.#ended) return false;
    const waiter = this.#waiters.shift();
    if (waiter) waiter({ value: item, done: false });
    else this.#items.push(item);
    return true;
  }

  /** The next item, or done once ended and drained. */
  next(): Promise<IteratorResult<T>> {
    const item = this.#items.shift();
    if (item !== undefined) {
      return Promise.resolve({ value: item, done: false });
    }
    if (this.#ended) return Promise.resolve({ value: undefined, done: true });
    return new Promise((resolve) => this.#waiters.push(resolve));
  }

  /**
   * End iteration after the buffered items have been taken (nothing new is
   * accepted). Idempotent.
   */
  finish(): void {
    this.#ended = true;
    for (const w of this.#waiters) w({ value: undefined, done: true });
    this.#waiters = [];
  }

  /** End iteration now; returns the undelivered items. Idempotent. */
  end(): T[] {
    this.finish();
    return this.#items.splice(0);
  }
}
