/**
 * @module transport/stream
 *
 * Embeddable transports for hosts that bring their own sockets: a
 * {@link StreamServerTransport} built from any byte stream plus a write
 * function, and a {@link StreamServerListener} that turns any async iterable
 * of such connections into the listener `McpServer.serve` consumes.
 *
 * This is the least-privilege path. `UnixSocketServerTransport.listen` needs
 * `--allow-read`/`--allow-write` on the socket path (it stats, chmods and
 * removes the file); a desktop app that runs without `--allow-write` can open
 * its sockets another way (libc via `Deno.dlopen`, a pre-opened `Deno.Conn`, a
 * pipe from a parent process) and still get the same session handling —
 * newline framing, concurrent dispatch, cancellation, pre-auth, identity.
 * Nothing in this module touches the file system, the network or the
 * environment, so it needs no permissions at all.
 */

import type { ConnectionInfo, ServerTransport } from "./traits.ts";
import { LineReader, MAX_LINE_BYTES } from "./framing.ts";
import {
  AcceptQueue,
  nextConnectionId,
  type PreAuthHook,
  runPreAuth,
} from "./admission.ts";

/**
 * The two halves of one byte-oriented connection, as a host provides them.
 *
 * `write` must deliver the bytes it is given. It may resolve to a byte count:
 * a count smaller than `bytes.length` is treated as a partial write and the
 * rest is written by further calls — so `conn.write.bind(conn)` from a
 * `Deno.Conn` (or an FFI `write(2)` wrapper) can be passed as-is. Writes are
 * never overlapped: the next call starts after the previous one settled.
 */
export interface ByteStreamEndpoint {
  /** Bytes from the client: a web stream or any async iterable of chunks. */
  readable: ReadableStream<Uint8Array> | AsyncIterable<Uint8Array>;
  /** Write bytes to the client (see the interface docs for partial writes). */
  write(bytes: Uint8Array): Promise<void | number>;
  /** Release the connection (close the fd/socket). Called at most once. */
  close?(): void | Promise<void>;
}

/** Options for {@link StreamServerTransport.from}. */
export interface StreamTransportOptions {
  /** Longest JSON-RPC line accepted (default 16 MiB). */
  maxLineBytes?: number;
}

/**
 * A server transport over a host-provided byte stream: newline-delimited
 * JSON-RPC read from `readable` (partial lines, several messages per chunk and
 * split UTF-8 sequences are handled), responses written through `write` in
 * call order. Hand it to `McpServer.serveConnection`.
 *
 * @example Serve one connection from an FFI socket
 * ```ts ignore
 * const transport = StreamServerTransport.from({
 *   readable: fdReadable(fd),            // ReadableStream or async generator
 *   write: (bytes) => fdWrite(fd, bytes), // may return a partial count
 *   close: () => libc.close(fd),
 * });
 * server.serveConnection(transport, { connection: { id: "tile-1", transport: "unix" } });
 * ```
 */
export class StreamServerTransport implements ServerTransport {
  #endpoint: ByteStreamEndpoint;
  #lines: LineReader;
  #encoder = new TextEncoder();
  #writing: Promise<void> = Promise.resolve();
  #closed = false;

  /**
   * Wrap a host connection; reading starts immediately.
   * @param endpoint Readable side, write function and optional close.
   * @param maxLineBytes Longest line accepted before a read throws.
   */
  constructor(
    endpoint: ByteStreamEndpoint,
    maxLineBytes: number = MAX_LINE_BYTES,
  ) {
    this.#endpoint = endpoint;
    const readable = endpoint.readable instanceof ReadableStream
      ? endpoint.readable
      : ReadableStream.from(endpoint.readable);
    this.#lines = new LineReader(readable, maxLineBytes);
  }

  /**
   * Build a transport from a host connection (same as the constructor, with
   * an options object).
   * @param endpoint Readable side, write function and optional close.
   * @param options Line-length limit.
   */
  static from(
    endpoint: ByteStreamEndpoint,
    options: StreamTransportOptions = {},
  ): StreamServerTransport {
    return new StreamServerTransport(endpoint, options.maxLineBytes);
  }

  /** Whether {@link close} was called. */
  get closed(): boolean {
    return this.#closed;
  }

  readRequest(): Promise<string | null> {
    if (this.#closed) return Promise.resolve(null);
    return this.#lines.read();
  }

  writeResponse(response: string): Promise<void> {
    const data = this.#encoder.encode(response + "\n");
    const next = this.#writing.then(() => this.#writeAll(data));
    // Keep the chain alive after a failed write; the caller still sees the error.
    this.#writing = next.catch(() => {});
    return next;
  }

  /** Write every byte of `data`, continuing after partial writes. */
  async #writeAll(data: Uint8Array): Promise<void> {
    if (this.#closed) throw new Error("stream transport is closed");
    for (let off = 0; off < data.length;) {
      const n = await this.#endpoint.write(data.subarray(off));
      if (typeof n !== "number") return; // void: the endpoint wrote it all
      if (n <= 0) throw new Error(`stream write made no progress (${n})`);
      off += n;
    }
  }

  /**
   * Close: a pending read resolves to `null`, later writes fail, and the
   * endpoint's `close()` runs once. Idempotent.
   */
  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    await this.#lines.cancel();
    try {
      await this.#endpoint.close?.();
    } catch { /* already closed by the stream cancel */ }
  }
}

/**
 * One connection from a host-implemented listener: the byte endpoint plus
 * what the host knows about the peer (merged into its `ConnectionInfo`).
 */
export interface StreamConnection extends ByteStreamEndpoint {
  /** Connection details (`id` defaults to a fresh one, `transport` to the listener's). */
  info?: Partial<ConnectionInfo>;
}

/** Options for {@link StreamServerListener}. */
export interface StreamListenOptions {
  /** `ConnectionInfo.transport` for connections that do not set one (default `"stream"`). */
  transport?: string;
  /** Identity stamped on connections whose `info` carries none. */
  identity?: unknown;
  /** First-line check run before a connection is yielded (see {@link PreAuthHook}). */
  preAuth?: PreAuthHook;
  /** How long a client has to send its pre-auth line (default 10 000 ms). */
  preAuthTimeoutMs?: number;
  /** Longest JSON-RPC line accepted per connection (default 16 MiB). */
  maxLineBytes?: number;
}

/** One connection yielded by a {@link StreamServerListener}. */
export interface AcceptedStreamConnection {
  /** The per-connection transport to hand to `McpServer.serveConnection`. */
  transport: StreamServerTransport;
  /** Who connected (id, transport kind, identity, host-provided fields). */
  info: ConnectionInfo;
}

/**
 * Adapts a host's own accept loop — any `AsyncIterable<StreamConnection>`
 * (FFI `accept(2)`, `Deno.Listener` you opened yourself, test pipes) — into a
 * listener for `McpServer.serve`. Each connection becomes a
 * {@link StreamServerTransport}; the optional pre-auth hook runs per
 * connection concurrently, exactly as for `UnixSocketServerTransport`.
 * Iterate it once; iteration ends when the source ends or after {@link close}.
 *
 * @example
 * ```ts ignore
 * async function* accepted() {
 *   for (;;) {
 *     const fd = await ffiAccept(listenFd);           // your FFI code
 *     yield { readable: fdReadable(fd), write: (b) => fdWrite(fd, b),
 *             close: () => libc.close(fd), info: { transport: "unix", path } };
 *   }
 * }
 * await server.serve(new StreamServerListener(accepted(), { identity: { tile: "1" } }));
 * ```
 */
export class StreamServerListener
  implements AsyncIterable<AcceptedStreamConnection> {
  #source: AsyncIterable<StreamConnection>;
  #iterator: AsyncIterator<StreamConnection> | null = null;
  #options: StreamListenOptions;
  #queue = new AcceptQueue<AcceptedStreamConnection>();
  #pending = new Set<StreamServerTransport>();
  #admitting = new Set<Promise<void>>();
  #closed = false;
  #closeCalled = false;
  #pumpDone: Promise<void>;

  /**
   * Start pulling connections from `source`.
   * @param source The host's accept loop; its iterator is `return()`ed on {@link close}.
   * @param options Transport name, identity, pre-auth hook and limits.
   */
  constructor(
    source: AsyncIterable<StreamConnection>,
    options: StreamListenOptions = {},
  ) {
    this.#source = source;
    this.#options = options;
    this.#pumpDone = this.#pump();
  }

  /** Whether {@link close} was called (or the source ended). */
  get closed(): boolean {
    return this.#closed;
  }

  /** Resolves once the source is no longer being read. */
  get finished(): Promise<void> {
    return this.#pumpDone;
  }

  /**
   * Stop accepting: the source iterator is returned, connections still in
   * pre-auth or not yet taken are closed, iteration ends. (When the source
   * ends by itself, connections already admitted are still yielded.) Connections already
   * yielded stay open. Idempotent.
   */
  close(): void {
    if (this.#closeCalled) return;
    this.#closeCalled = true;
    this.#closed = true;
    this.#iterator?.return?.().catch(() => {});
    for (const t of this.#pending) t.close();
    this.#pending.clear();
    for (const { transport } of this.#queue.end()) transport.close();
  }

  /** Yield admitted connections; `break` closes the listener. */
  [Symbol.asyncIterator](): AsyncIterator<AcceptedStreamConnection> {
    return {
      next: () => this.#queue.next(),
      return: () => {
        this.close();
        return Promise.resolve({ value: undefined, done: true });
      },
    };
  }

  async #pump(): Promise<void> {
    try {
      this.#iterator = this.#source[Symbol.asyncIterator]();
      while (!this.#closed) {
        const { value, done } = await this.#iterator.next();
        if (done) break;
        if (this.#closed) {
          await value.close?.();
          break;
        }
        const p: Promise<void> = this.#admit(value)
          .catch((e) => console.error(`stream connection setup failed: ${e}`))
          .finally(() => this.#admitting.delete(p));
        this.#admitting.add(p);
      }
    } catch (e) {
      if (!this.#closed) console.error(`stream listener source failed: ${e}`);
    }
    // Source exhausted: let connections still in pre-auth finish first.
    await Promise.allSettled([...this.#admitting]);
    // Admitted connections not taken yet are still delivered.
    this.#closed = true;
    this.#queue.finish();
  }

  async #admit(conn: StreamConnection): Promise<void> {
    const opts = this.#options;
    const transport = new StreamServerTransport(conn, opts.maxLineBytes);
    const kind = conn.info?.transport ?? opts.transport ?? "stream";
    const info: ConnectionInfo = {
      ...conn.info,
      id: conn.info?.id ?? nextConnectionId(kind),
      transport: kind,
    };
    if (info.identity === undefined && opts.identity !== undefined) {
      info.identity = opts.identity;
    }
    let admitted = true;
    if (opts.preAuth) {
      this.#pending.add(transport);
      try {
        admitted = await runPreAuth(
          transport,
          info,
          opts.preAuth,
          opts.preAuthTimeoutMs,
        );
      } finally {
        this.#pending.delete(transport);
      }
    }
    if (!admitted || !this.#queue.push({ transport, info })) {
      await transport.close();
    }
  }
}
