import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  type StreamConnection,
  StreamServerListener,
  StreamServerTransport,
} from "./stream.ts";

const enc = new TextEncoder();
const dec = new TextDecoder();

/** A readable the test feeds by hand. */
function feed() {
  let ctl!: ReadableStreamDefaultController<Uint8Array>;
  const readable = new ReadableStream<Uint8Array>({
    start(c) {
      ctl = c;
    },
  });
  return {
    readable,
    send: (s: string) => ctl.enqueue(enc.encode(s)),
    end: () => ctl.close(),
  };
}

Deno.test("StreamServerTransport reads lines from a ReadableStream and writes whole lines", async () => {
  const input = feed();
  const out: string[] = [];
  let closes = 0;
  const t = StreamServerTransport.from({
    readable: input.readable,
    write: (b) => {
      out.push(dec.decode(b));
      return Promise.resolve();
    },
    close: () => {
      closes += 1;
    },
  });
  input.send('{"a":1}\n{"b"');
  input.send(":2}\n");
  assertEquals(await t.readRequest(), '{"a":1}');
  assertEquals(await t.readRequest(), '{"b":2}');
  await t.writeResponse('{"ok":true}');
  assertEquals(out.join(""), '{"ok":true}\n');

  await t.close();
  await t.close();
  assertEquals(closes, 1);
  assert(t.closed);
  assertEquals(await t.readRequest(), null);
  await assertRejects(() => t.writeResponse("{}"), Error, "closed");
});

Deno.test("StreamServerTransport accepts an async iterable and ends at its end", async () => {
  async function* chunks() {
    yield enc.encode('{"x":"h');
    yield enc.encode('é"}\n{"y":1}');
  }
  const t = new StreamServerTransport({
    readable: chunks(),
    write: () => Promise.resolve(),
  });
  assertEquals(await t.readRequest(), '{"x":"hé"}');
  assertEquals(await t.readRequest(), '{"y":1}');
  assertEquals(await t.readRequest(), null);
});

Deno.test("StreamServerTransport continues partial writes and serializes writers", async () => {
  const out: string[] = [];
  let active = 0;
  const t = StreamServerTransport.from({
    readable: new ReadableStream(),
    // Like write(2): at most 3 bytes per call.
    write: async (b) => {
      active += 1;
      assertEquals(active, 1, "writes must not overlap");
      await Promise.resolve();
      const n = Math.min(3, b.length);
      out.push(dec.decode(b.subarray(0, n)));
      active -= 1;
      return n;
    },
  });
  await Promise.all([t.writeResponse('{"first":1}'), t.writeResponse("[2]")]);
  assertEquals(out.join(""), '{"first":1}\n[2]\n');
  await t.close();
});

Deno.test("StreamServerTransport rejects a write that makes no progress", async () => {
  const t = StreamServerTransport.from({
    readable: new ReadableStream(),
    write: () => Promise.resolve(0),
  });
  await assertRejects(() => t.writeResponse("{}"), Error, "no progress");
  await t.close();
});

/** A StreamConnection whose writes are collected as text. */
function memConn(info?: StreamConnection["info"]) {
  const input = feed();
  const written: string[] = [];
  let closed = false;
  const conn: StreamConnection = {
    readable: input.readable,
    write: (b) => {
      written.push(dec.decode(b));
      return Promise.resolve();
    },
    close: () => {
      closed = true;
    },
    info,
  };
  return { conn, input, written, isClosed: () => closed };
}

Deno.test("StreamServerListener yields host connections with info and identity", async () => {
  const a = memConn({ id: "tile-a", path: "/run/a.sock" });
  const b = memConn({ identity: { tile: "b" } });
  async function* source() {
    yield a.conn;
    yield b.conn;
  }
  const listener = new StreamServerListener(source(), {
    transport: "unix-ffi",
    identity: { tile: "default" },
  });
  const got = [];
  for await (const c of listener) got.push(c);
  assertEquals(got.length, 2);
  assertEquals(got[0].info.id, "tile-a");
  assertEquals(got[0].info.transport, "unix-ffi");
  assertEquals(got[0].info.path, "/run/a.sock");
  assertEquals(got[0].info.identity, { tile: "default" });
  assert(got[1].info.id.startsWith("unix-ffi-"));
  assertEquals(got[1].info.identity, { tile: "b" });
  assert(listener.closed);
  for (const c of got) await c.transport.close();
});

Deno.test("StreamServerListener runs preAuth per connection", async () => {
  const good = memConn();
  const bad = memConn();
  async function* source() {
    yield bad.conn;
    yield good.conn;
  }
  const listener = new StreamServerListener(source(), {
    preAuth: (line) =>
      line === "token-1"
        ? { ok: true, identity: { user: "u1" } }
        : { ok: false, reason: "bad token" },
  });
  bad.input.send("nope\n");
  good.input.send('token-1\n{"jsonrpc":"2.0"}\n');
  const got = [];
  for await (const c of listener) got.push(c);
  assertEquals(got.length, 1);
  assertEquals(got[0].info.identity, { user: "u1" });
  // The pre-auth line is consumed; MCP sees the next one.
  assertEquals(await got[0].transport.readRequest(), '{"jsonrpc":"2.0"}');
  assert(bad.isClosed());
  const reply = JSON.parse(bad.written.join(""));
  assertEquals(reply.error.code, -32003);
  assertEquals(reply.error.message, "Unauthorized: bad token");
  await got[0].transport.close();
});

Deno.test("StreamServerListener.close returns the source and closes untaken connections", async () => {
  const c = memConn();
  let returned = false;
  let release!: () => void;
  const source: AsyncIterable<StreamConnection> = {
    [Symbol.asyncIterator]() {
      let sent = false;
      return {
        next: async () => {
          if (!sent) {
            sent = true;
            return { value: c.conn, done: false };
          }
          await new Promise<void>((r) => (release = r));
          return { value: undefined, done: true };
        },
        return: () => {
          returned = true;
          release?.();
          return Promise.resolve({ value: undefined, done: true });
        },
      };
    },
  };
  const listener = new StreamServerListener(source);
  await new Promise((r) => setTimeout(r, 10));
  listener.close();
  await listener.finished;
  assert(returned);
  await new Promise((r) => setTimeout(r, 10)); // transport.close() is async
  assert(c.isClosed(), "queued connection is closed");
  assertEquals(await listener[Symbol.asyncIterator]().next(), {
    value: undefined,
    done: true,
  });
});
