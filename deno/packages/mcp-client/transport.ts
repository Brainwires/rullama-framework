/**
 * @module transport
 *
 * Transport layer for MCP communication.
 * Equivalent to Rust's `rullama-mcp/src/transport.rs`.
 *
 * Uses `Deno.Command` for subprocess management (stdio transport) and
 * `Deno.connect` for unix-socket / TCP connections ({@link SocketTransport}).
 */

import type {
  JsonRpcMessage,
  JsonRpcNotification,
  JsonRpcRequest,
  JsonRpcResponse,
} from "./types.ts";
import { parseJsonRpcMessage } from "./types.ts";

/**
 * Split a text stream into non-empty lines, keeping a trailing partial line
 * until the next chunk (or flush) completes it.
 */
export function lineSplitter(): TransformStream<string, string> {
  let lineBuffer = "";
  return new TransformStream<string, string>({
    transform(chunk, controller) {
      lineBuffer += chunk;
      const lines = lineBuffer.split("\n");
      lineBuffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.trim().length > 0) controller.enqueue(line);
      }
    },
    flush(controller) {
      if (lineBuffer.trim().length > 0) controller.enqueue(lineBuffer);
    },
  });
}

/** Any message a client writes: a request, a notification, or a reply to a server request. */
export type OutgoingMessage =
  | JsonRpcRequest
  | JsonRpcNotification
  | JsonRpcResponse;

/**
 * A client-side MCP connection carrying newline-delimited JSON-RPC.
 * Implemented by {@link StdioTransport} and {@link SocketTransport}; implement
 * it to plug any other byte stream into `McpClient.connectTransport`.
 */
export interface ClientTransport {
  /** Write one message (a line of JSON). Concurrent calls must not interleave. */
  sendRequest(message: OutgoingMessage): Promise<void>;
  /** Read the next message; throws at end of stream. */
  receiveMessage(): Promise<JsonRpcMessage>;
  /** Read the next message, which must be a response. */
  receiveResponse(): Promise<JsonRpcResponse>;
  /** Close the connection. A pending {@link receiveMessage} rejects. */
  close(): Promise<void>;
}

/**
 * Stdio transport for communicating with MCP servers via stdin/stdout.
 * Equivalent to Rust `StdioTransport`.
 *
 * Spawns a child process and communicates over newline-delimited JSON on
 * stdin (write) and stdout (read).
 */
export class StdioTransport implements ClientTransport {
  #child: Deno.ChildProcess;
  #writer: WritableStreamDefaultWriter<Uint8Array>;
  #reader: ReadableStreamDefaultReader<string>;
  #encoder: TextEncoder;
  #closed = false;

  private constructor(
    child: Deno.ChildProcess,
    writer: WritableStreamDefaultWriter<Uint8Array>,
    reader: ReadableStreamDefaultReader<string>,
  ) {
    this.#child = child;
    this.#writer = writer;
    this.#reader = reader;
    this.#encoder = new TextEncoder();
  }

  /**
   * Create a new stdio transport by spawning a command.
   * Equivalent to Rust `StdioTransport::new`.
   */
  // deno-lint-ignore require-await
  static async create(
    command: string,
    args: string[],
    env?: Record<string, string>,
  ): Promise<StdioTransport> {
    const cmd = new Deno.Command(command, {
      args,
      stdin: "piped",
      stdout: "piped",
      stderr: "inherit",
      env,
    });

    const child = cmd.spawn();
    const writer = child.stdin.getWriter();

    // Create a line reader from stdout
    const reader = child.stdout
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(lineSplitter())
      .getReader();

    // We return immediately; the process is now running.
    // Suppress the "unhandled promise" on status since we manage lifecycle via close().
    child.status.catch(() => {});

    return new StdioTransport(child, writer, reader);
  }

  /**
   * Send a JSON-RPC request via stdin.
   * Equivalent to Rust `StdioTransport::send_request`.
   */
  async sendRequest(request: OutgoingMessage): Promise<void> {
    if (this.#closed) {
      throw new Error("Transport is closed");
    }
    const json = JSON.stringify(request) + "\n";
    await this.#writer.write(this.#encoder.encode(json));
  }

  /**
   * Receive a JSON-RPC response from stdout.
   * Equivalent to Rust `StdioTransport::receive_response`.
   */
  async receiveResponse(): Promise<JsonRpcResponse> {
    const msg = await this.receiveMessage();
    if (msg.type === "response") {
      return msg.response;
    }
    throw new Error(
      `Expected response but received notification: ${msg.notification.method}`,
    );
  }

  /**
   * Receive any JSON-RPC message (response or notification).
   * Discriminates based on presence of a non-null "id" field.
   * Equivalent to Rust `StdioTransport::receive_message`.
   */
  async receiveMessage(): Promise<JsonRpcMessage> {
    if (this.#closed) {
      throw new Error("Transport is closed");
    }
    const { value, done } = await this.#reader.read();
    if (done || value === undefined) {
      throw new Error("MCP server closed connection (EOF on stdout)");
    }
    return parseJsonRpcMessage(value);
  }

  /**
   * Close the transport and kill the child process.
   * Equivalent to Rust `StdioTransport::close`.
   */
  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;

    try {
      this.#reader.releaseLock();
    } catch { /* ignore */ }

    try {
      await this.#writer.close();
    } catch { /* ignore */ }

    try {
      this.#child.kill("SIGTERM");
    } catch { /* ignore — process may have already exited */ }

    // Wait briefly for the process to exit
    try {
      await this.#child.status;
    } catch { /* ignore */ }
  }

  /** Whether the transport has been closed. */
  get closed(): boolean {
    return this.#closed;
  }
}

/**
 * One connected socket (unix domain or TCP) speaking newline-delimited
 * JSON-RPC — the client side of `@rullama/mcp-server`'s
 * `UnixSocketServerTransport` / `TcpServerTransport`.
 */
export class SocketTransport implements ClientTransport {
  /** The underlying connection. */
  readonly conn: Deno.Conn;
  #reader: ReadableStreamDefaultReader<string>;
  #encoder = new TextEncoder();
  #writing: Promise<void> = Promise.resolve();
  #closed = false;

  /**
   * Wrap an open connection.
   * @param conn A connection from `Deno.connect`.
   */
  constructor(conn: Deno.Conn) {
    this.conn = conn;
    this.#reader = conn.readable
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(lineSplitter())
      .getReader();
  }

  /**
   * Connect to a unix domain socket.
   * @param path Socket file path.
   * @param options `preAuthLine`: a raw line (e.g. a token) sent before any
   *   JSON-RPC, for servers with a pre-auth hook.
   */
  static async connectUnix(
    path: string,
    options: { preAuthLine?: string } = {},
  ): Promise<SocketTransport> {
    const t = new SocketTransport(
      await Deno.connect({ transport: "unix", path }),
    );
    if (options.preAuthLine !== undefined) {
      await t.sendLine(options.preAuthLine);
    }
    return t;
  }

  /**
   * Connect over TCP.
   * @param hostname Host to connect to (e.g. `127.0.0.1`).
   * @param port Port to connect to.
   * @param options `preAuthLine`: a raw line (e.g. a token) sent before any
   *   JSON-RPC, for servers with a pre-auth hook.
   */
  static async connectTcp(
    hostname: string,
    port: number,
    options: { preAuthLine?: string } = {},
  ): Promise<SocketTransport> {
    const t = new SocketTransport(
      await Deno.connect({ transport: "tcp", hostname, port }),
    );
    if (options.preAuthLine !== undefined) {
      await t.sendLine(options.preAuthLine);
    }
    return t;
  }

  /** Write one raw line (a newline is appended). Writes never interleave. */
  sendLine(line: string): Promise<void> {
    const data = this.#encoder.encode(line + "\n");
    const next = this.#writing.then(async () => {
      if (this.#closed) throw new Error("Transport is closed");
      let written = 0;
      while (written < data.length) {
        written += await this.conn.write(data.subarray(written));
      }
    });
    this.#writing = next.catch(() => {});
    return next;
  }

  /** Send one JSON-RPC message. */
  sendRequest(message: OutgoingMessage): Promise<void> {
    return this.sendLine(JSON.stringify(message));
  }

  /** Read the next message; throws when the server closes the connection. */
  async receiveMessage(): Promise<JsonRpcMessage> {
    if (this.#closed) throw new Error("Transport is closed");
    const { value, done } = await this.#reader.read();
    if (done || value === undefined) {
      throw new Error("MCP server closed connection (EOF on socket)");
    }
    return parseJsonRpcMessage(value);
  }

  /** Read the next message, which must be a response. */
  async receiveResponse(): Promise<JsonRpcResponse> {
    const msg = await this.receiveMessage();
    if (msg.type === "response") return msg.response;
    throw new Error(
      `Expected response but received notification: ${msg.notification.method}`,
    );
  }

  /** Close the connection. Idempotent. */
  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    try {
      await this.#reader.cancel();
    } catch { /* ignore */ }
    try {
      this.conn.close();
    } catch { /* already closed by the stream cancel */ }
  }

  /** Whether the transport has been closed. */
  get closed(): boolean {
    return this.#closed;
  }
}

/**
 * Transport layer for MCP communication.
 * Equivalent to Rust `Transport` enum.
 *
 * Wraps any {@link ClientTransport}: {@link StdioTransport} (subprocess) or
 * {@link SocketTransport} (unix socket / TCP).
 */
export class Transport implements ClientTransport {
  #inner: ClientTransport;

  /**
   * Wrap a connected transport; every send/receive call on this instance is
   * forwarded to it unchanged.
   *
   * @param transport E.g. the stdio transport produced by `StdioTransport.create`.
   */
  constructor(transport: ClientTransport) {
    this.#inner = transport;
  }

  /**
   * Send a JSON-RPC request.
   * Equivalent to Rust `Transport::send_request`.
   */
  async sendRequest(request: OutgoingMessage): Promise<void> {
    await this.#inner.sendRequest(request);
  }

  /**
   * Receive a JSON-RPC response.
   * Equivalent to Rust `Transport::receive_response`.
   */
  async receiveResponse(): Promise<JsonRpcResponse> {
    return await this.#inner.receiveResponse();
  }

  /**
   * Receive any JSON-RPC message (response or notification).
   * Equivalent to Rust `Transport::receive_message`.
   */
  async receiveMessage(): Promise<JsonRpcMessage> {
    return await this.#inner.receiveMessage();
  }

  /**
   * Close the transport.
   * Equivalent to Rust `Transport::close`.
   */
  async close(): Promise<void> {
    await this.#inner.close();
  }
}
