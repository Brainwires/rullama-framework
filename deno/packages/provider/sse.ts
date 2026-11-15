/**
 * Shared SSE (Server-Sent Events) stream parser utilities.
 *
 * Parses `text/event-stream` format from fetch() response bodies into
 * individual data strings. Used by Anthropic and OpenAI providers.
 */

/**
 * Parse a `text/event-stream` response body into an async iterable of
 * `data:` payloads. Yields each data string (with `data: ` prefix stripped).
 * Stops when `[DONE]` is encountered or the stream ends.
 */
export async function* parseSSEStream(
  body: ReadableStream<Uint8Array>,
): AsyncIterable<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete events (delimited by \n\n)
      let pos: number;
      while ((pos = buffer.indexOf("\n\n")) !== -1) {
        const eventBlock = buffer.slice(0, pos);
        buffer = buffer.slice(pos + 2);

        // Extract `data:` lines from the event block
        for (const line of eventBlock.split("\n")) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") return;
            yield data;
          }
        }
      }
    }
  } finally {
    // Leaving early (consumer stopped, `[DONE]`, abort) cancels the body so
    // the HTTP connection closes instead of idling until the server ends it.
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

/**
 * Parse a newline-delimited JSON stream (NDJSON) from a fetch() response body.
 * Used by Gemini and Ollama which stream JSON objects separated by newlines.
 */
export async function* parseNDJSONStream(
  body: ReadableStream<Uint8Array>,
): AsyncIterable<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let pos: number;
      while ((pos = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, pos).trim();
        buffer = buffer.slice(pos + 1);

        if (line.length > 0) {
          yield line;
        }
      }
    }
  } finally {
    // Leaving early (consumer stopped, `[DONE]`, abort) cancels the body so
    // the HTTP connection closes instead of idling until the server ends it.
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
