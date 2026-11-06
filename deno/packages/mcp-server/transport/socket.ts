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
import { MAX_LINE_BYTES } from "./framing.ts";
import { StreamServerTransport } from "./stream.ts";
import {
  AcceptQueue,
  nextConnectionId,
  type PreAuthHook,
  runPreAuth,
} from "./admission.ts";

export {
  nextConnectionId,
  type PreAuthHook,
  type PreAuthResult,
} from "./admission.ts";

/**
 * One accepted socket connection, framed as newline-delimited JSON-RPC.
 * Reads tolerate partial lines and several messages per chunk; writes are
 * written fully, in call order. A {@link StreamServerTransport} over a
 * `Deno.Conn`.
 */
export class SocketServerTransport extends StreamServerTransport {
  /** The accepted connection. */
  readonly conn: Deno.Conn;

  /**
   * Wrap an accepted connection.
   * @param conn The connection from `Deno.Listener.accept()` (or `Deno.connect`).
   * @param maxLineBytes Longest line accepted before a read throws.
   */
  constructor(conn: Deno.Conn, maxLineBytes: number = MAX_LINE_BYTES) {
    super({
      readable: conn.readable,
      write: (bytes) => conn.write(bytes),
      close: () => conn.close(),
    }, maxLineBytes);
    this.conn = conn;
  }
}

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

/**
 * One connection yielded by a listener. Socket listeners yield
 * `AcceptedConnection<SocketServerTransport>` (the default); `McpServer.serve`
 * accepts any `AsyncIterable<AcceptedConnection<ServerTransport>>`, so a host
 * can feed it connections built on its own transport.
 */
export interface AcceptedConnection<
  T extends ServerTransport = SocketServerTransport,
> {
  /** The per-connection transport to hand to `McpServer.serveConnection`. */
  transport: T;
  /** Who connected (id, addresses, identity). */
  info: ConnectionInfo;
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
  #queue = new AcceptQueue<AcceptedConnection>();
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
    for (const { transport } of this.#queue.end()) transport.close();
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
      next: () => this.#queue.next(),
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
    if (!this.#queue.push({ transport, info })) await transport.close();
  }

  /**
   * Run the pre-auth hook (if any) on the first line; see `runPreAuth`.
   * Returns whether the connection is admitted.
   */
  async #preAuthenticate(
    transport: SocketServerTransport,
    info: ConnectionInfo,
  ): Promise<boolean> {
    const preAuth = this.#options.preAuth;
    if (!preAuth) return true;
    this.#pending.add(transport);
    try {
      return await runPreAuth(
        transport,
        info,
        preAuth,
        this.#options.preAuthTimeoutMs,
      );
    } finally {
      this.#pending.delete(transport);
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
