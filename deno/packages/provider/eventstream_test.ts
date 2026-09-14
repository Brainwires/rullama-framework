import { assertEquals, assertRejects } from "@std/assert";
import {
  decodeFrame,
  encodeFrame,
  parseBedrockEvents,
  parseEventStream,
} from "./eventstream.ts";

const enc = new TextEncoder();
const chunk = (event: unknown) =>
  encodeFrame(
    {
      ":message-type": "event",
      ":event-type": "chunk",
      ":content-type": "application/json",
    },
    enc.encode(JSON.stringify({ bytes: btoa(JSON.stringify(event)) })),
  );
const stream = (...parts: Uint8Array[]) =>
  new ReadableStream<Uint8Array>({
    start(c) {
      for (const p of parts) c.enqueue(p);
      c.close();
    },
  });

Deno.test("decodeFrame round-trips headers and payload, and reports incomplete input", () => {
  const frame = encodeFrame({ ":event-type": "chunk" }, enc.encode("hi"));
  const decoded = decodeFrame(frame)!;
  assertEquals(decoded.consumed, frame.byteLength);
  assertEquals(decoded.frame.headers[":event-type"], "chunk");
  assertEquals(new TextDecoder().decode(decoded.frame.payload), "hi");
  assertEquals(decodeFrame(frame.subarray(0, 10)), undefined);
  assertEquals(decodeFrame(frame.subarray(0, frame.byteLength - 1)), undefined);
});

Deno.test("parseEventStream reassembles frames split across reads", async () => {
  const a = encodeFrame({ ":event-type": "chunk" }, enc.encode("one"));
  const b = encodeFrame({ ":event-type": "chunk" }, enc.encode("two"));
  const all = new Uint8Array(a.byteLength + b.byteLength);
  all.set(a);
  all.set(b, a.byteLength);
  const frames = [];
  for await (
    const f of parseEventStream(
      stream(all.subarray(0, 7), all.subarray(7, 30), all.subarray(30)),
    )
  ) {
    frames.push(new TextDecoder().decode(f.payload));
  }
  assertEquals(frames, ["one", "two"]);
});

Deno.test("parseBedrockEvents decodes chunk payloads and throws on exception frames", async () => {
  const events = [];
  for await (
    const e of parseBedrockEvents(
      stream(chunk({ type: "message_start" }), chunk({ type: "message_stop" })),
    )
  ) {
    events.push(e.type);
  }
  assertEquals(events, ["message_start", "message_stop"]);
  const boom = encodeFrame(
    { ":message-type": "exception", ":exception-type": "throttlingException" },
    enc.encode('{"message":"slow down"}'),
  );
  await assertRejects(
    async () => {
      for await (const _ of parseBedrockEvents(stream(boom))) { /* drain */ }
    },
    Error,
    "throttlingException",
  );
});
