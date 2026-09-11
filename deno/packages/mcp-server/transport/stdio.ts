/**
 * @module transport/stdio
 *
 * Stdio-based server transport for MCP communication.
 * Equivalent to Rust's `StdioServerTransport`.
 */

import type { ServerTransport } from "./traits.ts";

/** Longest single JSON-RPC line accepted (16 MiB); a longer one is a protocol error. */
export const MAX_LINE_BYTES = 16 * 1024 * 1024;

/**
 * Stdio-based server transport (stdin/stdout).
 * Reads newline-delimited JSON from stdin and writes to stdout.
 * Equivalent to Rust `StdioServerTransport`.
 */
export class StdioServerTransport implements ServerTransport {
  private reader: ReadableStreamDefaultReader<string>;
  private buffer = "";
  private encoder = new TextEncoder();
  private done = false;

  /** @param input Byte stream to read requests from (default: stdin). */
  constructor(input: ReadableStream<Uint8Array> = Deno.stdin.readable) {
    const decoder = new TextDecoderStream();
    input.pipeTo(decoder.writable).catch(() => {
      this.done = true;
    });
    this.reader = decoder.readable.getReader();
  }

  async readRequest(): Promise<string | null> {
    // Check buffer for a complete line
    while (true) {
      const newlineIndex = this.buffer.indexOf("\n");
      if (newlineIndex !== -1) {
        const line = this.buffer.slice(0, newlineIndex).trim();
        this.buffer = this.buffer.slice(newlineIndex + 1);
        // A blank line is not a message and not EOF: keep reading.
        if (line.length === 0) continue;
        return line;
      }
      if (this.buffer.length > MAX_LINE_BYTES) {
        this.buffer = "";
        throw new Error(
          `line exceeds ${MAX_LINE_BYTES} bytes without a newline; dropping input`,
        );
      }

      if (this.done) {
        // Drain remaining buffer
        if (this.buffer.trim().length > 0) {
          const line = this.buffer.trim();
          this.buffer = "";
          return line;
        }
        return null;
      }

      const { value, done } = await this.reader.read();
      if (done) {
        this.done = true;
        // Process remaining buffer
        if (this.buffer.trim().length > 0) {
          const line = this.buffer.trim();
          this.buffer = "";
          return line;
        }
        return null;
      }
      this.buffer += value;
    }
  }

  async writeResponse(response: string): Promise<void> {
    const data = this.encoder.encode(response + "\n");
    await Deno.stdout.write(data);
  }
}
