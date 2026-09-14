import { assert } from "@std/assert";
import { PathPattern } from "./types.ts";

Deno.test("PathPattern escapes brackets and refuses to match on an invalid pattern", () => {
  assert(new PathPattern("src/[id].ts").matches("src/[id].ts"));
  assert(!new PathPattern("src/[id].ts").matches("src/i.ts"));
  // `**/.env*` must keep matching what the default policies rely on.
  assert(new PathPattern("**/.env*").matches("/app/.env.local"));
  // Every metacharacter is escaped, so `(` is a literal: it matches itself and
  // nothing else. (The old code fell back to substring matching on a regex
  // error, which made a broken deny rule a no-op and an allow rule over-broad.)
  assert(new PathPattern("(").matches("("));
  assert(!new PathPattern("(").matches("anything containing ( here"));
});
