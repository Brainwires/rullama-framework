import { assertEquals } from "@std/assert";
import { lineSplitter } from "./transport.ts";

async function split(chunks: string[]): Promise<string[]> {
  const src = new ReadableStream<string>({
    start(c) {
      for (const ch of chunks) c.enqueue(ch);
      c.close();
    },
  });
  const out: string[] = [];
  for await (const line of src.pipeThrough(lineSplitter())) out.push(line);
  return out;
}

Deno.test("lineSplitter joins partial lines across chunks and flushes the tail", async () => {
  assertEquals(await split(['{"a":', '1}\n{"b":2}\n\n', '{"c":3}']), [
    '{"a":1}',
    '{"b":2}',
    '{"c":3}',
  ]);
  assertEquals(await split(["\n\n"]), []);
  assertEquals(await split(["one\ntwo\nthree\n"]), ["one", "two", "three"]);
});
