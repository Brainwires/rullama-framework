/**
 * @module transport
 *
 * Transport layer for MCP communication.
 * Equivalent to Rust's `rullama-mcp/src/transport.rs`.
 *
 * Uses `Deno.Command` for subprocess management (stdio transport).
 */

import type {
  JsonRpcMessage,
  JsonRpcRequest,
  JsonRpcResponse,
} from "./types.ts";
import { parseJsonRpcMessage } from "./types.ts";

/**
 * Stdio transport for communicating with MCP servers via stdin/stdout.
 * Equivalent to Rust `StdioTransport`.
 *
 * Spawns a child process and communicates over newline-delimited JSON on
 * stdin (write) and stdout (read).
 */
export class StdioTransport {
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
    let lineBuffer = "";
    const reader = child.stdout
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(
        new TransformStream<string, string>({
          transform(chunk, controller) {
            lineBuffer += chunk;
            const lines = lineBuffer.split("\n");
            // Keep the last (possibly incomplete) line in the buffer
            lineBuffer = lines.pop() ?? "";
            for (const line of lines) {
              if (line.trim().length > 0) {
                controller.enqueue(line);
              }
            }
          },
          flush(controller) {
            if (lineBuffer.trim().length > 0) {
              controller.enqueue(lineBuffer);
            }
          },
        }),
      )
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
  async sendRequest(request: JsonRpcRequest): Promise<void> {
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
 * Transport layer for MCP communication.
 * Equivalent to Rust `Transport` enum.
 *
 * Currently only supports stdio transport. Additional transport types
 * (e.g., SSE, WebSocket) can be added in the future.
 */
export class Transport {
  #inner: StdioTransport;

  constructor(transport: StdioTransport) {
    this.#inner = transport;
  }

  /**
   * Send a JSON-RPC request.
   * Equivalent to Rust `Transport::send_request`.
   */
  async sendRequest(request: JsonRpcRequest): Promise<void> {
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
