import { assert, assertEquals } from "@std/assert";
import { defaultListOptions, SessionId } from "./types.ts";
import { SessionError } from "./error.ts";

Deno.test("SessionId stringifies to its value", () => {
  const id = SessionId.new("user-42");
  assertEquals(id.asStr(), "user-42");
  assertEquals(id.toString(), "user-42");
  assertEquals(String(id), "user-42");
});

Deno.test("SessionId equality via class and string", () => {
  const a = new SessionId("x");
  assert(a.equals(new SessionId("x")));
  assert(a.equals("x"));
  assert(!a.equals("y"));
});

Deno.test("defaultListOptions is unbounded", () => {
  const opts = defaultListOptions();
  assertEquals(opts.offset, 0);
  assertEquals(opts.limit, null);
});

Deno.test("SessionError tags kind and formats message", () => {
  const ser = SessionError.serialization("bad json");
  assertEquals(ser.kind, "serialization");
  assert(ser.message.includes("bad json"));

  const sto = SessionError.storage("disk full");
  assertEquals(sto.kind, "storage");
  assert(sto.message.includes("disk full"));
});
