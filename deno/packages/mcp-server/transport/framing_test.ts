import { assertEquals, assertRejects } from "@std/assert";
import { LineReader } from "./framing.ts";

function streamOf(chunks: (string | Uint8Array)[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(c) {
      for (const ch of chunks) {
        c.enqueue(typeof ch === "string" ? enc.encode(ch) : ch);
      }
      c.close();
    },
  });
}

async function readAll(r: LineReader): Promise<string[]> {
  const out: string[] = [];
  for (let line = await r.read(); line !== null; line = await r.read()) {
    out.push(line);
  }
  return out;
}

Deno.test("LineReader joins a message split across chunks", async () => {
  const r = new LineReader(streamOf(['{"jsonrpc":"2.0",', '"id":1', "}\n"]));
  assertEquals(await readAll(r), ['{"jsonrpc":"2.0","id":1}']);
});

Deno.test("LineReader splits several messages delivered in one chunk", async () => {
  const r = new LineReader(
    streamOf(['{"a":1}\n{"b":2}\r\n\n{"c":3}\n{"d"', ":4}"]),
  );
  assertEquals(await readAll(r), ['{"a":1}', '{"b":2}', '{"c":3}', '{"d":4}']);
});

Deno.test("LineReader keeps a multi-byte UTF-8 character split across chunks", async () => {
  const bytes = new TextEncoder().encode('{"t":"héllo ✓"}\n');
  const cut = bytes.indexOf(0xc3) + 1; // inside the two-byte "é"
  const r = new LineReader(streamOf([bytes.slice(0, cut), bytes.slice(cut)]));
  assertEquals(await readAll(r), ['{"t":"héllo ✓"}']);
});

Deno.test("LineReader rejects an over-long line without a newline", async () => {
  const r = new LineReader(streamOf(["x".repeat(64)]), 16);
  await assertRejects(() => r.read(), Error, "exceeds 16 bytes");
});

Deno.test("LineReader.cancel ends a pending read with null", async () => {
  const r = new LineReader(new ReadableStream<Uint8Array>()); // never yields
  const pending = r.read();
  await r.cancel();
  assertEquals(await pending, null);
});
