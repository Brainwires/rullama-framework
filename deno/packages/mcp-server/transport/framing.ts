/**
 * @module transport/framing
 *
 * Newline-delimited JSON-RPC framing shared by every server transport
 * (stdio, unix socket, TCP): one JSON value per line, UTF-8.
 */

/** Longest single JSON-RPC line accepted (16 MiB); a longer one is a protocol error. */
export const MAX_LINE_BYTES = 16 * 1024 * 1024;

/**
 * Reads newline-delimited messages from a byte stream.
 *
 * A chunk may carry part of a line, exactly one line, or several lines; a
 * multi-byte UTF-8 sequence may be split across chunks. Blank lines are
 * skipped, surrounding whitespace is trimmed, and an unterminated last line is
 * returned when the stream ends.
 */
export class LineReader {
  #reader: ReadableStreamDefaultReader<string>;
  #buffer = "";
  #done = false;
  #maxLineBytes: number;

  /**
   * Start decoding `input` as UTF-8 immediately; a stream error marks the
   * reader done so {@link read} drains the buffer and then returns `null`.
   * @param input Byte stream to read lines from.
   * @param maxLineBytes Longest line accepted before {@link read} throws.
   */
  constructor(
    input: ReadableStream<Uint8Array>,
    maxLineBytes: number = MAX_LINE_BYTES,
  ) {
    this.#maxLineBytes = maxLineBytes;
    const decoder = new TextDecoderStream();
    input.pipeTo(decoder.writable).catch(() => {
      this.#done = true;
    });
    this.#reader = decoder.readable.getReader();
  }

  /**
   * The next non-blank line (without its newline), or `null` at end of stream.
   * Throws when more than `maxLineBytes` arrive without a newline (the
   * buffered input is dropped).
   */
  async read(): Promise<string | null> {
    while (true) {
      const newlineIndex = this.#buffer.indexOf("\n");
      if (newlineIndex !== -1) {
        const line = this.#buffer.slice(0, newlineIndex).trim();
        this.#buffer = this.#buffer.slice(newlineIndex + 1);
        // A blank line is not a message and not EOF: keep reading.
        if (line.length === 0) continue;
        return line;
      }
      if (this.#buffer.length > this.#maxLineBytes) {
        this.#buffer = "";
        throw new Error(
          `line exceeds ${this.#maxLineBytes} bytes without a newline; dropping input`,
        );
      }
      if (this.#done) return this.#drain();

      let chunk: ReadableStreamReadResult<string>;
      try {
        chunk = await this.#reader.read();
      } catch {
        this.#done = true;
        return this.#drain();
      }
      if (chunk.done) {
        this.#done = true;
        return this.#drain();
      }
      this.#buffer += chunk.value;
    }
  }

  /** Stop reading: a pending {@link read} resolves to `null`. */
  async cancel(): Promise<void> {
    this.#done = true;
    this.#buffer = "";
    try {
      await this.#reader.cancel();
    } catch { /* already closed */ }
  }

  #drain(): string | null {
    const line = this.#buffer.trim();
    this.#buffer = "";
    return line.length > 0 ? line : null;
  }
}
