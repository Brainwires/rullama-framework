/**
 * Shared test helpers for the provider tests: a fake `fetch` and a streamed
 * reply that hangs until aborted. Not published (see deno.json `exclude`).
 *
 * @module
 */

import { assertRejects } from "@std/assert";
import {
  ChatOptions,
  Message,
  type Provider,
  type StreamChunk,
} from "@rullama/core";

/**
 * Stream a reply from `provider` with a `ChatOptions.signal`, aborting it on
 * the first chunk; asserts the stream rejects. The chunks received and the
 * signal passed are returned.
 */
export async function abortOnFirstChunk(
  provider: Provider,
): Promise<{ got: StreamChunk[]; signal: AbortSignal }> {
  const controller = new AbortController();
  const got: StreamChunk[] = [];
  await assertRejects(async () => {
    for await (
      const c of provider.streamChat(
        [Message.user("hi")],
        undefined,
        new ChatOptions({ signal: controller.signal }),
      )
    ) {
      got.push(c);
      controller.abort();
    }
  });
  return { got, signal: controller.signal };
}

/** Run `fn` with `fetch` replaced by `handler`; restores it afterwards. */
export async function withFetch(
  handler: (init: RequestInit) => Response | Promise<Response>,
  fn: () => Promise<void>,
): Promise<void> {
  const original = globalThis.fetch;
  globalThis.fetch = (_input, init) => Promise.resolve(handler(init ?? {}));
  try {
    await fn();
  } finally {
    globalThis.fetch = original;
  }
}

/** Every chunk of a stream. */
export async function collect(
  it: AsyncIterable<StreamChunk>,
): Promise<StreamChunk[]> {
  const out: StreamChunk[] = [];
  for await (const c of it) out.push(c);
  return out;
}

/**
 * A fake streaming reply that sends `text`, then hangs until the request
 * signal aborts (erroring the body, as real fetch does) or the reader cancels
 * it. Pushes `"aborted"` / `"cancelled"` to `events`.
 */
export function hangingStream(
  init: RequestInit,
  events: string[],
  text: string,
): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      const signal = init.signal;
      signal?.addEventListener("abort", () => {
        events.push("aborted");
        controller.error(signal.reason);
      });
    },
    cancel() {
      events.push("cancelled");
    },
  });
  return new Response(body);
}
