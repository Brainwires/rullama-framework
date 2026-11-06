/**
 * @module transport/socket
 *
 * Socket server transports: a listener on a unix domain socket path
 * ({@link UnixSocketServerTransport}) or a TCP hostname/port
 * ({@link TcpServerTransport}) that yields one {@link SocketServerTransport}
 * per accepted connection. Framing is the same newline-delimited JSON-RPC as
 * `StdioServerTransport`, so `nc -U <path>` is a working stdio shim for
 * any MCP client that only speaks stdio.
 */

import type { ConnectionInfo, ServerTransport } from "./traits.ts";
import { LineReader, MAX_LINE_BYTES } from "./framing.ts";

/**
 * One accepted socket connection, framed as newline-delimited JSON-RPC.
 * Reads tolerate partial lines and several messages per chunk; writes are
 * written fully, in call order.
 */
export class SocketServerTransport implements ServerTransport {
  /** The accepted connection. */
  readonly conn: Deno.Conn;
  #lines: LineReader;
  #encoder = new TextEncoder();
  #writing: Promise<void> = Promise.resolve();
  #closed = false;

  /**
   * Wrap an accepted connection.
   * @param conn The connection from `Deno.Listener.accept()` (or `Deno.connect`).
   * @param maxLineBytes Longest line accepted before a read throws.
   */
  constructor(conn: Deno.Conn, maxLineBytes: number = MAX_LINE_BYTES) {
    this.conn = conn;
    this.#lines = new LineReader(conn.readable, maxLineBytes);
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

  /** Write every byte of `data` (a socket write may be partial). */
  async #writeAll(data: Uint8Array): Promise<void> {
    if (this.#closed) throw new Error("socket transport is closed");
    for (let off = 0; off < data.length;) {
      off += await this.conn.write(data.subarray(off));
    }
  }

  /** Close the connection; a pending read resolves to `null`. Idempotent. */
  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    await this.#lines.cancel();
    try {
      this.conn.close();
    } catch { /* already closed by the stream cancel */ }
  }
}

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

/** Options shared by {@link UnixSocketServerTransport} and {@link TcpServerTransport}. */
export interface SocketListenOptions {
  /** Identity stamped on every connection's `ConnectionInfo.identity`. */
  identity?: unknown;
  /** First-line check run before a connection is yielded (see {@link PreAuthHook}). */
  preAuth?: PreAuthHook;
  /** How long a client has to send its pre-auth line (default 10 000 ms). */
  preAuthTimeoutMs?: number;
  /** Longest JSON-RPC line accepted per connection (default 16 MiB). */
  maxLineBytes?: number;
}

/** Options for {@link UnixSocketServerTransport.listen}. */
export interface UnixSocketListenOptions extends SocketListenOptions {
  /**
   * Permission bits applied to the socket file right after binding
   * (default `0o600`: only the owning user may connect). `null` keeps the
   * process umask.
   */
  mode?: number | null;
}

/** Options for {@link TcpServerTransport.listen}. */
export interface TcpListenOptions extends SocketListenOptions {
  /** Interface to bind (default `"127.0.0.1"`: loopback only). */
  hostname?: string;
  /** Port to bind; `0` (the default) picks a free one — read it from `port`. */
  port?: number;
}

/** One connection yielded by a socket listener. */
export interface AcceptedConnection {
  /** The per-connection transport to hand to `McpServer.serveConnection`. */
  transport: SocketServerTransport;
  /** Who connected (id, addresses, identity). */
  info: ConnectionInfo;
}

let connectionCounter = 0;

/** A fresh process-unique connection id with the given prefix. */
export function nextConnectionId(prefix: string): string {
  connectionCounter += 1;
  return `${prefix}-${connectionCounter}`;
}

/** JSON-RPC error line sent to a client whose pre-auth was rejected. */
function unauthorizedLine(reason?: string): string {
  return JSON.stringify({
    jsonrpc: "2.0",
    id: null,
    error: {
      code: -32003,
      message: `Unauthorized${reason ? `: ${reason}` : ""}`,
    },
  });
}

/**
 * Base class of the socket listeners: accepts connections in the background,
 * runs the optional pre-auth hook per connection (concurrently, so a slow
 * client never blocks the accept loop) and yields admitted connections from
 * its async iterator. Iterate it once; iteration ends after {@link close}.
 */
export abstract class SocketServerListener
  implements AsyncIterable<AcceptedConnection> {
  #listener: Deno.Listener;
  #options: SocketListenOptions;
  #queue: AcceptedConnection[] = [];
  #waiters: ((v: IteratorResult<AcceptedConnection>) => void)[] = [];
  #pending = new Set<SocketServerTransport>();
  #closed = false;
  #acceptDone: Promise<void>;

  /**
   * Start accepting on `listener`.
   * @param listener A bound listener; owned (and closed) by this object.
   * @param options Identity, pre-auth hook and limits.
   */
  protected constructor(listener: Deno.Listener, options: SocketListenOptions) {
    this.#listener = listener;
    this.#options = options;
    this.#acceptDone = this.#acceptLoop();
  }

  /** Transport kind used in {@link ConnectionInfo.transport} and connection ids. */
  protected abstract get kind(): string;

  /** Extra fields for each connection's {@link ConnectionInfo}. */
  protected connectionExtras(_conn: Deno.Conn): Partial<ConnectionInfo> {
    return {};
  }

  /** The bound local address. */
  get addr(): Deno.Addr {
    return this.#listener.addr;
  }

  /** Whether {@link close} was called. */
  get closed(): boolean {
    return this.#closed;
  }

  /**
   * Stop accepting. Connections already yielded stay open (close them through
   * their session or transport); connections still in pre-auth are dropped.
   * Iteration ends. Idempotent.
   */
  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    try {
      this.#listener.close();
    } catch { /* already closed */ }
    for (const t of this.#pending) t.close();
    this.#pending.clear();
    for (const { transport } of this.#queue) transport.close();
    this.#queue = [];
    for (const w of this.#waiters) w({ value: undefined, done: true });
    this.#waiters = [];
    this.onClose();
  }

  /** Hook for subclasses (socket-file cleanup); called once from {@link close}. */
  protected onClose(): void {}

  /** Resolves once the accept loop has stopped (after {@link close}). */
  get finished(): Promise<void> {
    return this.#acceptDone;
  }

  /** Yield admitted connections until {@link close}; `break` closes the listener. */
  [Symbol.asyncIterator](): AsyncIterator<AcceptedConnection> {
    return {
      next: () => {
        const item = this.#queue.shift();
        if (item) return Promise.resolve({ value: item, done: false });
        if (this.#closed) {
          return Promise.resolve({ value: undefined, done: true });
        }
        return new Promise((resolve) => this.#waiters.push(resolve));
      },
      return: () => {
        this.close();
        return Promise.resolve({ value: undefined, done: true });
      },
    };
  }

  async #acceptLoop(): Promise<void> {
    while (!this.#closed) {
      let conn: Deno.Conn;
      try {
        conn = await this.#listener.accept();
      } catch (e) {
        if (this.#closed) break;
        // A transient accept error (EMFILE, aborted handshake) must not kill
        // the listener; a closed listener surfaces as BadResource/Interrupted.
        if (
          e instanceof Deno.errors.BadResource ||
          e instanceof Deno.errors.Interrupted
        ) {
          break;
        }
        console.error(`socket accept error: ${e}`);
        await new Promise((r) => setTimeout(r, 50));
        continue;
      }
      this.#admit(conn).catch((e) =>
        console.error(`socket connection setup failed: ${e}`)
      );
    }
    this.close();
  }

  async #admit(conn: Deno.Conn): Promise<void> {
    const transport = new SocketServerTransport(
      conn,
      this.#options.maxLineBytes,
    );
    const info: ConnectionInfo = {
      id: nextConnectionId(this.kind),
      transport: this.kind,
      localAddr: conn.localAddr,
      remoteAddr: conn.remoteAddr,
      ...this.connectionExtras(conn),
      ...(this.#options.identity !== undefined
        ? { identity: this.#options.identity }
        : {}),
    };
    const admitted = !this.#closed && await this.#preAuthenticate(
      transport,
      info,
    );
    if (!admitted || this.#closed) {
      await transport.close();
      return;
    }
    const accepted = { transport, info };
    const waiter = this.#waiters.shift();
    if (waiter) waiter({ value: accepted, done: false });
    else this.#queue.push(accepted);
  }

  /**
   * Run the pre-auth hook (if any) on the first line. On success the hook's
   * identity replaces `info.identity`; on rejection the client gets one
   * Unauthorized line. Returns whether the connection is admitted.
   */
  async #preAuthenticate(
    transport: SocketServerTransport,
    info: ConnectionInfo,
  ): Promise<boolean> {
    const preAuth = this.#options.preAuth;
    if (!preAuth) return true;
    this.#pending.add(transport);
    let verdict: PreAuthResult;
    try {
      verdict = await preAuth(await this.#firstLine(transport), info);
    } catch (e) {
      verdict = { ok: false, reason: String(e) };
    } finally {
      this.#pending.delete(transport);
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

  async #firstLine(transport: SocketServerTransport): Promise<string | null> {
    const ms = this.#options.preAuthTimeoutMs ?? 10_000;
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
}

/**
 * Listener on a unix domain socket path. Each accepted connection is one MCP
 * client; serve it with `McpServer.serve(listener)` (or
 * `serveConnection` per {@link AcceptedConnection}).
 *
 * On bind, a stale socket file (nothing listening — a previous process
 * crashed) is removed; a live one fails with `Deno.errors.AddrInUse`, and a
 * non-socket file at the path fails with `Deno.errors.AlreadyExists`. On
 * {@link close} the socket file is removed if it is still the one this
 * listener created. Requires `--allow-read` and `--allow-write` for the path.
 */
export class UnixSocketServerTransport extends SocketServerListener {
  /** The socket file path. */
  readonly path: string;
  #ino: number | null;

  private constructor(
    listener: Deno.Listener,
    path: string,
    ino: number | null,
    options: UnixSocketListenOptions,
  ) {
    super(listener, options);
    this.path = path;
    this.#ino = ino;
  }

  /**
   * Bind `path` (clearing a stale socket file first) and start accepting.
   * @param path Socket file path (keep it short: ~104 bytes on macOS).
   * @param options Identity, pre-auth hook, file mode and limits.
   */
  static async listen(
    path: string,
    options: UnixSocketListenOptions = {},
  ): Promise<UnixSocketServerTransport> {
    await removeStaleSocket(path);
    const listener = Deno.listen({ transport: "unix", path });
    let ino: number | null = null;
    try {
      const mode = options.mode === undefined ? 0o600 : options.mode;
      if (mode !== null) await Deno.chmod(path, mode);
      ino = (await Deno.lstat(path)).ino;
    } catch (e) {
      listener.close();
      throw e;
    }
    return new UnixSocketServerTransport(listener, path, ino, options);
  }

  /** `"unix"`. */
  protected get kind(): string {
    return "unix";
  }

  /** Adds the socket `path` to each connection's info. */
  protected override connectionExtras(): Partial<ConnectionInfo> {
    return { path: this.path };
  }

  /** Remove the socket file if it is still the one this listener bound. */
  protected override onClose(): void {
    try {
      const st = Deno.lstatSync(this.path);
      if (st.isSocket && (this.#ino === null || st.ino === this.#ino)) {
        Deno.removeSync(this.path);
      }
    } catch { /* already gone */ }
  }
}

/**
 * Remove a socket file nobody is listening on. Throws `AddrInUse` if a server
 * answers, `AlreadyExists` if the path is not a socket.
 */
async function removeStaleSocket(path: string): Promise<void> {
  let st: Deno.FileInfo;
  try {
    st = await Deno.lstat(path);
  } catch (e) {
    if (e instanceof Deno.errors.NotFound) return;
    throw e;
  }
  if (!st.isSocket) {
    throw new Deno.errors.AlreadyExists(
      `${path} exists and is not a socket; refusing to replace it`,
    );
  }
  let probe: Deno.Conn | null = null;
  try {
    probe = await Deno.connect({ transport: "unix", path });
  } catch {
    // ECONNREFUSED / ENOENT: stale.
    await Deno.remove(path).catch(() => {});
    return;
  }
  probe.close();
  throw new Deno.errors.AddrInUse(`a server is already listening on ${path}`);
}

/**
 * Listener on a TCP hostname/port (loopback by default). Pair it with a
 * {@link PreAuthHook} — any local process can reach a loopback port.
 */
export class TcpServerTransport extends SocketServerListener {
  private constructor(listener: Deno.Listener, options: TcpListenOptions) {
    super(listener, options);
  }

  /**
   * Bind and start accepting.
   * @param options Hostname (default `127.0.0.1`), port (default `0` = any
   *   free port), identity, pre-auth hook and limits.
   */
  static listen(options: TcpListenOptions = {}): TcpServerTransport {
    const listener = Deno.listen({
      transport: "tcp",
      hostname: options.hostname ?? "127.0.0.1",
      port: options.port ?? 0,
    });
    return new TcpServerTransport(listener, options);
  }

  /** The bound port (useful with `port: 0`). */
  get port(): number {
    return (this.addr as Deno.NetAddr).port;
  }

  /** The bound hostname. */
  get hostname(): string {
    return (this.addr as Deno.NetAddr).hostname;
  }

  /** `"tcp"`. */
  protected get kind(): string {
    return "tcp";
  }
}
