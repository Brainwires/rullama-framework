import { assert, assertEquals } from "@std/assert";
import { parseCategories, routeFromFallback, routeFromLocal } from "./router.ts";

Deno.test("route fallback has 0.5 confidence", () => {
  const r = routeFromFallback(["FileOps", "Search"]);
  assert(!r.used_local_llm);
  assertEquals(r.confidence, 0.5);
  assertEquals(r.categories.length, 2);
});

Deno.test("route from local keeps confidence", () => {
  const r = routeFromLocal(["Git"], 0.9);
  assert(r.used_local_llm);
  assertEquals(r.confidence, 0.9);
});

Deno.test("parseCategories picks what's mentioned", () => {
  const cats = parseCategories("FileOps, Git, Bash");
  assert(cats.includes("FileOps"));
  assert(cats.includes("Git"));
  assert(cats.includes("Bash"));
});
