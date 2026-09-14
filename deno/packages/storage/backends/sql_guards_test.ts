import { assertEquals, assertThrows } from "@std/assert";
import { Filters } from "../types.ts";
import {
  buildSelect as pgSelect,
  filterToSql as pgFilter,
} from "./postgres.ts";
import { buildSelect as mySelect, filterToSql as myFilter } from "./mysql.ts";
import { filterToSurrealQL } from "./surrealdb.ts";
import { assertIdentifier, assertLimit } from "./sql_guards.ts";

Deno.test("identifiers: plain names pass, quote/space/semicolon smuggling fails", () => {
  assertEquals(assertIdentifier("user_id"), "user_id");
  for (
    const bad of [
      'x" = 1 OR "1',
      "a b",
      "drop;",
      "",
      "1abc",
      "`x`",
      "x".repeat(64),
    ]
  ) {
    assertThrows(() => assertIdentifier(bad), Error, "invalid SQL");
  }
  assertEquals(assertLimit(10), 10);
  assertThrows(() => assertLimit(1.5), Error, "invalid LIMIT");
  assertThrows(() => assertLimit(-1), Error, "invalid LIMIT");
});

Deno.test("filter builders refuse unsafe column names and Raw without opt-in", () => {
  const bad = Filters.Eq('name" = 1 OR "1', { kind: "Utf8", value: "x" });
  assertThrows(() => pgFilter(bad, 1), Error, "invalid SQL column");
  assertThrows(() => myFilter(bad), Error, "invalid SQL column");
  assertThrows(
    () => filterToSurrealQL(bad, { value: 0 }),
    Error,
    "invalid SQL field",
  );
  const raw = Filters.Raw("1 = 1; DROP TABLE t");
  assertThrows(() => pgFilter(raw, 1), Error, "Raw filters are disabled");
  assertThrows(() => myFilter(raw), Error, "Raw filters are disabled");
  assertThrows(
    () => filterToSurrealQL(raw, { value: 0 }),
    Error,
    "Raw filters are disabled",
  );
  assertThrows(
    () => pgSelect('t"; DROP TABLE x; --'),
    Error,
    "invalid SQL table name",
  );
  assertThrows(() => mySelect("t", undefined, 2.5), Error, "invalid LIMIT");
});
