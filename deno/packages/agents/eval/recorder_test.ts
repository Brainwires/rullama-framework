import { assert, assertEquals, assertNotEquals } from "@std/assert";
import {
  _toolCallRecord,
  computeSequenceDiff,
  isExactMatch,
  levenshtein,
  ToolSequenceRecorder,
} from "./recorder.ts";

Deno.test("record and retrieve", () => {
  const r = new ToolSequenceRecorder();
  r.record("read_file", { path: "a.rs" });
  r.record("write_file", { path: "b.rs" });

  const calls = r.calls();
  assertEquals(calls.length, 2);
  assertEquals(calls[0].name, "read_file");
  assertEquals(calls[1].name, "write_file");
});

Deno.test("call names", () => {
  const r = new ToolSequenceRecorder();
  r.record("bash", {});
  r.record("read_file", {});
  assertEquals(r.callNames(), ["bash", "read_file"]);
});

Deno.test("diff exact match", () => {
  const r = new ToolSequenceRecorder();
  r.record("a", {});
  r.record("b", {});
  r.record("c", {});

  const diff = r.diffAgainst(["a", "b", "c"]);
  assert(isExactMatch(diff));
  assert(Math.abs(diff.similarity - 1.0) < 1e-9);
});

Deno.test("diff partial match", () => {
  const r = new ToolSequenceRecorder();
  r.record("a", {});
  r.record("x", {}); // unexpected
  r.record("c", {});

  const diff = r.diffAgainst(["a", "b", "c"]);
  assert(!isExactMatch(diff));
  assertEquals(diff.edit_distance, 1);
  assert(diff.similarity > 0.5);
});

Deno.test("diff empty vs expected", () => {
  const r = new ToolSequenceRecorder();
  const diff = r.diffAgainst(["a", "b"]);
  assertEquals(diff.edit_distance, 2);
  assertEquals(diff.similarity, 0.0);
});

Deno.test("diff both empty", () => {
  const r = new ToolSequenceRecorder();
  const diff = r.diffAgainst([]);
  assert(isExactMatch(diff));
  assert(Math.abs(diff.similarity - 1.0) < 1e-9);
});

Deno.test("reset clears calls", () => {
  const r = new ToolSequenceRecorder();
  r.record("a", {});
  r.reset();
  assert(r.isEmpty());
});

Deno.test("args fingerprint differs for different args", () => {
  const r1 = _toolCallRecord("tool", { a: 1 });
  const r2 = _toolCallRecord("tool", { a: 2 });
  assertNotEquals(r1.args_fingerprint, r2.args_fingerprint);
});

Deno.test("args fingerprint same for same args", () => {
  const r1 = _toolCallRecord("tool", { x: "hello" });
  const r2 = _toolCallRecord("tool", { x: "hello" });
  assertEquals(r1.args_fingerprint, r2.args_fingerprint);
});

Deno.test("levenshtein identical", () => {
  assertEquals(levenshtein(["a", "b", "c"], ["a", "b", "c"]), 0);
});

Deno.test("levenshtein single substitution", () => {
  assertEquals(levenshtein(["a", "b", "c"], ["a", "x", "c"]), 1);
});

Deno.test("levenshtein insert delete", () => {
  assertEquals(levenshtein(["a", "b"], ["a", "b", "c"]), 1);
  assertEquals(levenshtein(["a", "b", "c"], ["a", "b"]), 1);
});

Deno.test("compute sequence diff helper", () => {
  const d = computeSequenceDiff(["a"], ["a"]);
  assert(isExactMatch(d));
});
