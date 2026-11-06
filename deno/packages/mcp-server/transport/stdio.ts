/**
 * @module transport/stdio
 *
 * Stdio-based server transport for MCP communication.
 * Equivalent to Rust's `StdioServerTransport`.
 */

import type { ServerTransport } from "./traits.ts";
import { LineReader } from "./framing.ts";

/**
 * Stdio-based server transport (stdin/stdout).
 * Reads newline-delimited JSON from stdin and writes to stdout.
 * Equivalent to Rust `StdioServerTransport`.
 */
export class StdioServerTransport implements ServerTransport {
  private lines: LineReader;
  private encoder = new TextEncoder();

  /**
   * Start decoding `input` as UTF-8 immediately; a stream error marks the
   * transport done so {@link readRequest} drains and returns `null`.
   * @param input Byte stream to read requests from (default: stdin).
   */
  constructor(input: ReadableStream<Uint8Array> = Deno.stdin.readable) {
    this.lines = new LineReader(input);
  }

  readRequest(): Promise<string | null> {
    return this.lines.read();
  }

  async writeResponse(response: string): Promise<void> {
    const data = this.encoder.encode(response + "\n");
    let written = 0;
    while (written < data.length) {
      written += await Deno.stdout.write(data.subarray(written));
    }
  }
}
