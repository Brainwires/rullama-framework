/**
 * Tests for PostgreSQL backend SQL generation, filter conversion, and type mapping.
 *
 * These tests exercise the pure functions (no live database required).
 */

import { assertEquals } from "@std/assert";
import {
  buildCount,
  buildCreateTable,
  buildDelete,
  buildInsert,
  buildSelect,
  fieldValueToParam,
  filterToSql,
  parseVectorText,
} from "./postgres.ts";
import {
  type FieldDef,
  FieldTypes,
  FieldValues,
  Filters,
  optionalField,
  requiredField,
} from "../types.ts";

// ---------------------------------------------------------------------------
// filterToSql
// ---------------------------------------------------------------------------

Deno.test("filterToSql - Eq", () => {
  const [sql, vals] = filterToSql(
    Filters.Eq("name", FieldValues.Utf8("Alice")),
    1,
  );
  assertEquals(sql, `"name" = $1`);
  assertEquals(vals.length, 1);
});

Deno.test("filterToSql - Ne", () => {
  const [sql, vals] = filterToSql(
    Filters.Ne("status", FieldValues.Utf8("deleted")),
    1,
  );
  assertEquals(sql, `"status" != $1`);
  assertEquals(vals.length, 1);
});

Deno.test("filterToSql - Lt / Gt / Lte / Gte", () => {
  const [ltSql] = filterToSql(Filters.Lt("x", FieldValues.Int32(5)), 1);
  assertEquals(ltSql, `"x" < $1`);

  const [gtSql] = filterToSql(Filters.Gt("x", FieldValues.Int32(5)), 1);
  assertEquals(gtSql, `"x" > $1`);

  const [lteSql] = filterToSql(Filters.Lte("x", FieldValues.Int32(5)), 1);
  assertEquals(lteSql, `"x" <= $1`);

  const [gteSql] = filterToSql(Filters.Gte("x", FieldValues.Int32(5)), 1);
  assertEquals(gteSql, `"x" >= $1`);
});

Deno.test("filterToSql - IsNull / NotNull", () => {
  const [isNullSql, isNullVals] = filterToSql(Filters.IsNull("email"), 1);
  assertEquals(isNullSql, `"email" IS NULL`);
  assertEquals(isNullVals.length, 0);

  const [notNullSql, notNullVals] = filterToSql(Filters.NotNull("email"), 1);
  assertEquals(notNullSql, `"email" IS NOT NULL`);
  assertEquals(notNullVals.length, 0);
});

Deno.test("filterToSql - In", () => {
  const [sql, vals] = filterToSql(
    Filters.In("id", [
      FieldValues.Int64(1),
      FieldValues.Int64(2),
      FieldValues.Int64(3),
    ]),
    1,
  );
  assertEquals(sql, `"id" IN ($1, $2, $3)`);
  assertEquals(vals.length, 3);
});

Deno.test("filterToSql - empty In", () => {
  const [sql, vals] = filterToSql(Filters.In("id", []), 1);
  assertEquals(sql, "1 = 0");
  assertEquals(vals.length, 0);
});

Deno.test("filterToSql - And compound", () => {
  const filter = Filters.And([
    Filters.Eq("name", FieldValues.Utf8("Alice")),
    Filters.Gt("age", FieldValues.Int32(21)),
  ]);
  const [sql, vals] = filterToSql(filter, 1);
  assertEquals(sql, `("name" = $1 AND "age" > $2)`);
  assertEquals(vals.length, 2);
});

Deno.test("filterToSql - Or compound", () => {
  const filter = Filters.Or([
    Filters.Eq("status", FieldValues.Utf8("active")),
    Filters.Eq("status", FieldValues.Utf8("pending")),
  ]);
  const [sql, vals] = filterToSql(filter, 1);
  assertEquals(sql, `("status" = $1 OR "status" = $2)`);
  assertEquals(vals.length, 2);
});

Deno.test("filterToSql - empty And / Or", () => {
  const [andSql, andVals] = filterToSql(Filters.And([]), 1);
  assertEquals(andSql, "1 = 1");
  assertEquals(andVals.length, 0);

  const [orSql, orVals] = filterToSql(Filters.Or([]), 1);
  assertEquals(orSql, "1 = 0");
  assertEquals(orVals.length, 0);
});

Deno.test("filterToSql - Raw expression", () => {
  const [sql, vals] = filterToSql(Filters.Raw("custom_fn(col) > 0"), 1, {
    allowRaw: true,
  });
  assertEquals(sql, "custom_fn(col) > 0");
  assertEquals(vals.length, 0);
});

Deno.test("filterToSql - nested And/Or", () => {
  const filter = Filters.And([
    Filters.Eq("a", FieldValues.Int32(1)),
    Filters.Or([
      Filters.Eq("b", FieldValues.Int32(2)),
      Filters.Eq("c", FieldValues.Int32(3)),
    ]),
  ]);
  const [sql, vals] = filterToSql(filter, 1);
  assertEquals(sql, `("a" = $1 AND ("b" = $2 OR "c" = $3))`);
  assertEquals(vals.length, 3);
});

// ---------------------------------------------------------------------------
// buildCreateTable
// ---------------------------------------------------------------------------

Deno.test("buildCreateTable - basic schema", () => {
  const schema: FieldDef[] = [
    requiredField("id", FieldTypes.Utf8),
    requiredField("count", FieldTypes.Int64),
    optionalField("embedding", FieldTypes.Vector(384)),
  ];
  const sql = buildCreateTable("my_table", schema);
  assertEquals(
    sql,
    `CREATE TABLE IF NOT EXISTS "my_table" ("id" TEXT NOT NULL PRIMARY KEY, "count" BIGINT NOT NULL, "embedding" vector(384))`,
  );
});

// ---------------------------------------------------------------------------
// buildInsert
// ---------------------------------------------------------------------------

Deno.test("buildInsert - two rows", () => {
  const records: import("../types.ts").Record[] = [
    [
      ["id", FieldValues.Utf8("1")],
      ["name", FieldValues.Utf8("Alice")],
    ],
    [
      ["id", FieldValues.Utf8("2")],
      ["name", FieldValues.Utf8("Bob")],
    ],
  ];
  const [sql, params] = buildInsert("users", records);
  assertEquals(
    sql,
    `INSERT INTO "users" ("id", "name") VALUES ($1, $2), ($3, $4)`,
  );
  assertEquals(params.length, 4);
});

// ---------------------------------------------------------------------------
// buildSelect
// ---------------------------------------------------------------------------

Deno.test("buildSelect - no filter with limit", () => {
  const [sql, params] = buildSelect("messages", undefined, 10);
  assertEquals(sql, `SELECT * FROM "messages" LIMIT 10`);
  assertEquals(params.length, 0);
});

Deno.test("buildSelect - with filter", () => {
  const [sql, params] = buildSelect(
    "messages",
    Filters.Eq("user_id", FieldValues.Utf8("u1")),
  );
  assertEquals(sql, `SELECT * FROM "messages" WHERE "user_id" = $1`);
  assertEquals(params.length, 1);
});

// ---------------------------------------------------------------------------
// buildDelete
// ---------------------------------------------------------------------------

Deno.test("buildDelete - basic", () => {
  const [sql, params] = buildDelete(
    "tasks",
    Filters.Eq("id", FieldValues.Utf8("123")),
  );
  assertEquals(sql, `DELETE FROM "tasks" WHERE "id" = $1`);
  assertEquals(params.length, 1);
});

// ---------------------------------------------------------------------------
// buildCount
// ---------------------------------------------------------------------------

Deno.test("buildCount - no filter", () => {
  const [sql, params] = buildCount("events");
  assertEquals(sql, `SELECT COUNT(*) FROM "events"`);
  assertEquals(params.length, 0);
});

Deno.test("buildCount - with filter", () => {
  const [sql, params] = buildCount(
    "events",
    Filters.Gt("score", FieldValues.Float64(0.5)),
  );
  assertEquals(sql, `SELECT COUNT(*) FROM "events" WHERE "score" > $1`);
  assertEquals(params.length, 1);
});

// ---------------------------------------------------------------------------
// fieldValueToParam
// ---------------------------------------------------------------------------

Deno.test("fieldValueToParam - string", () => {
  assertEquals(fieldValueToParam(FieldValues.Utf8("hello")), "hello");
});

Deno.test("fieldValueToParam - null string", () => {
  assertEquals(fieldValueToParam(FieldValues.Utf8(null)), null);
});

Deno.test("fieldValueToParam - int", () => {
  assertEquals(fieldValueToParam(FieldValues.Int32(42)), 42);
});

Deno.test("fieldValueToParam - boolean", () => {
  assertEquals(fieldValueToParam(FieldValues.Boolean(true)), true);
});

Deno.test("fieldValueToParam - vector becomes pg text format", () => {
  assertEquals(fieldValueToParam(FieldValues.Vector([1, 2, 3])), "[1,2,3]");
});

Deno.test("parseVectorText - pgvector text form", () => {
  assertEquals(parseVectorText("[1,2.5,-3]"), [1, 2.5, -3]);
  assertEquals(parseVectorText(""), []);
  assertEquals(parseVectorText("not a vector"), []);
});
