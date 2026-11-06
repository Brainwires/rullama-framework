// Test helpers for @rullama/mcp-server (excluded from the published package).

import type { ServerTransport } from "./transport/traits.ts";

/** A JSON-RPC message as written by the server, parsed. */
// deno-lint-ignore no-explicit-any
export type Msg = Record<string, any>;

/**
 * In-memory duplex transport: the test pushes client lines in over time and
 * waits for the server's writes.
 */
export class MemoryTransport implements ServerTransport {
  written: Msg[] = [];
  closed = false;
  #incoming: string[] = [];
  #readers: ((line: string | null) => void)[] = [];
  #ended = false;
  #watchers: (() => void)[] = [];

  /** Deliver one client message (object or raw line). */
  push(message: object | string): void {
    const line = typeof message === "string"
      ? message
      : JSON.stringify(message);
    const reader = this.#readers.shift();
    if (reader) reader(line);
    else this.#incoming.push(line);
  }

  /** Client input EOF. */
  end(): void {
    this.#ended = true;
    for (const r of this.#readers.splice(0)) r(null);
  }

  readRequest(): Promise<string | null> {
    const line = this.#incoming.shift();
    if (line !== undefined) return Promise.resolve(line);
    if (this.#ended) return Promise.resolve(null);
    return new Promise((resolve) => this.#readers.push(resolve));
  }

  writeResponse(response: string): Promise<void> {
    if (this.closed) return Promise.reject(new Error("closed"));
    this.written.push(JSON.parse(response));
    for (const w of this.#watchers.splice(0)) w();
    return Promise.resolve();
  }

  close(): Promise<void> {
    this.closed = true;
    this.end();
    return Promise.resolve();
  }

  /** Resolve with the first written message matching `pred` (fails after `ms`). */
  async waitFor(pred: (m: Msg) => boolean, ms = 2000): Promise<Msg> {
    const deadline = Date.now() + ms;
    while (true) {
      const hit = this.written.find(pred);
      if (hit) return hit;
      const left = deadline - Date.now();
      if (left <= 0) {
        throw new Error(
          `timed out; written so far: ${JSON.stringify(this.written)}`,
        );
      }
      let timer: ReturnType<typeof setTimeout> | undefined;
      await new Promise<void>((resolve) => {
        this.#watchers.push(resolve);
        timer = setTimeout(resolve, left);
      });
      clearTimeout(timer);
    }
  }

  /** The response with `id`. */
  response(id: number | string, ms?: number): Promise<Msg> {
    return this.waitFor((m) => m.id === id && !("method" in m), ms);
  }
}

/** Resolve after `ms` milliseconds. */
export const delay = (ms: number): Promise<void> =>
  new Promise((r) => setTimeout(r, ms));
