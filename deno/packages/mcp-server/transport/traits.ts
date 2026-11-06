/**
 * @module transport/traits
 *
 * Server transport interface for MCP communication.
 * Equivalent to Rust's `ServerTransport` trait.
 */

/**
 * Interface for server-side MCP transport: one bidirectional
 * newline-delimited JSON-RPC stream (one client).
 * Equivalent to Rust `ServerTransport` trait.
 */
export interface ServerTransport {
  /** Read the next JSON-RPC request, or null on EOF. */
  readRequest(): Promise<string | null>;

  /** Write a JSON-RPC response (or a server-initiated notification). */
  writeResponse(response: string): Promise<void>;

  /**
   * Optional: close the underlying stream. After this a pending
   * {@link readRequest} resolves to `null` (or rejects) and writes fail.
   * Transports without it (stdio) simply end when their input does.
   */
  close?(): Promise<void>;
}

/**
 * Who is on the other end of one server connection. The server copies it onto
 * every `RequestContext` (`ctx.connection`) so handlers know which
 * socket — and so which host-side entity — a request came from.
 */
export interface ConnectionInfo {
  /** Process-unique connection id, e.g. `"unix-3"`. */
  id: string;
  /** Transport kind: `"stdio"`, `"unix"`, `"tcp"`, or a custom name. */
  transport: string;
  /** Socket path, for unix-socket connections. */
  path?: string;
  /** Local address of the accepted socket (socket transports). */
  localAddr?: Deno.Addr;
  /** Peer address of the accepted socket (socket transports). */
  remoteAddr?: Deno.Addr;
  /**
   * Host-defined identity: the listener's `identity` option, replaced by
   * whatever a pre-auth hook returned for this connection.
   */
  identity?: unknown;
}
