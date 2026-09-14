/**
 * Tests for MySQL backend: SQL generation, filter conversion, type mapping.
 *
 * These tests exercise the pure helper functions (no live MySQL server required).
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  buildCount,
  buildCreateTable,
  buildDelete,
  buildInsert,
  buildSelect,
  cosineSimilarity,
  fieldValueToParam,
  filterToSql,
  mapFieldType,
} from "./mysql.ts";
import {
  type FieldDef,
  FieldTypes,
  FieldValues,
  Filters,
  optionalField,
  requiredField,
} from "../types.ts";

// ---------------------------------------------------------------------------
// mapFieldType
// ---------------------------------------------------------------------------

Deno.test("mapFieldType - standard types", () => {
  assertEquals(mapFieldType(FieldTypes.Utf8), "TEXT");
  assertEquals(mapFieldType(FieldTypes.Int32), "INT");
  assertEquals(mapFieldType(FieldTypes.Int64), "BIGINT");
  assertEquals(mapFieldType(FieldTypes.UInt32), "INT UNSIGNED");
  assertEquals(mapFieldType(FieldTypes.UInt64), "BIGINT UNSIGNED");
  assertEquals(mapFieldType(FieldTypes.Float32), "FLOAT");
  assertEquals(mapFieldType(FieldTypes.Float64), "DOUBLE");
  assertEquals(mapFieldType(FieldTypes.Boolean), "BOOLEAN");
});

Deno.test("mapFieldType - vector becomes JSON", () => {
  assertEquals(mapFieldType(FieldTypes.Vector(384)), "JSON");
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

Deno.test("fieldValueToParam - vector becomes JSON string", () => {
  assertEquals(fieldValueToParam(FieldValues.Vector([1, 2, 3])), "[1,2,3]");
});

// ---------------------------------------------------------------------------
// filterToSql
// ---------------------------------------------------------------------------

Deno.test("filterToSql - Eq", () => {
  const [sql, vals] = filterToSql(
    Filters.Eq("name", FieldValues.Utf8("Alice")),
  );
  assertEquals(sql, "`name` = ?");
  assertEquals(vals.length, 1);
});

Deno.test("filterToSql - Ne", () => {
  const [sql, vals] = filterToSql(
    Filters.Ne("status", FieldValues.Utf8("deleted")),
  );
  assertEquals(sql, "`status` != ?");
  assertEquals(vals.length, 1);
});

Deno.test("filterToSql - Lt / Gt / Lte / Gte", () => {
  const [ltSql] = filterToSql(Filters.Lt("x", FieldValues.Int32(5)));
  assertEquals(ltSql, "`x` < ?");

  const [gtSql] = filterToSql(Filters.Gt("x", FieldValues.Int32(5)));
  assertEquals(gtSql, "`x` > ?");

  const [lteSql] = filterToSql(Filters.Lte("x", FieldValues.Int32(5)));
  assertEquals(lteSql, "`x` <= ?");

  const [gteSql] = filterToSql(Filters.Gte("x", FieldValues.Int32(5)));
  assertEquals(gteSql, "`x` >= ?");
});

Deno.test("filterToSql - IsNull / NotNull", () => {
  const [isNullSql, isNullVals] = filterToSql(Filters.IsNull("email"));
  assertEquals(isNullSql, "`email` IS NULL");
  assertEquals(isNullVals.length, 0);

  const [notNullSql, notNullVals] = filterToSql(Filters.NotNull("email"));
  assertEquals(notNullSql, "`email` IS NOT NULL");
  assertEquals(notNullVals.length, 0);
});

Deno.test("filterToSql - In", () => {
  const [sql, vals] = filterToSql(
    Filters.In("id", [
      FieldValues.Int64(1),
      FieldValues.Int64(2),
      FieldValues.Int64(3),
    ]),
  );
  assertEquals(sql, "`id` IN (?, ?, ?)");
  assertEquals(vals.length, 3);
});

Deno.test("filterToSql - empty In", () => {
  const [sql, vals] = filterToSql(Filters.In("id", []));
  assertEquals(sql, "1 = 0");
  assertEquals(vals.length, 0);
});

Deno.test("filterToSql - And compound", () => {
  const filter = Filters.And([
    Filters.Eq("name", FieldValues.Utf8("Alice")),
    Filters.Gt("age", FieldValues.Int32(21)),
  ]);
  const [sql, vals] = filterToSql(filter);
  assertEquals(sql, "(`name` = ? AND `age` > ?)");
  assertEquals(vals.length, 2);
});

Deno.test("filterToSql - Or compound", () => {
  const filter = Filters.Or([
    Filters.Eq("status", FieldValues.Utf8("active")),
    Filters.Eq("status", FieldValues.Utf8("pending")),
  ]);
  const [sql, vals] = filterToSql(filter);
  assertEquals(sql, "(`status` = ? OR `status` = ?)");
  assertEquals(vals.length, 2);
});

Deno.test("filterToSql - empty And / Or", () => {
  const [andSql, andVals] = filterToSql(Filters.And([]));
  assertEquals(andSql, "1 = 1");
  assertEquals(andVals.length, 0);

  const [orSql, orVals] = filterToSql(Filters.Or([]));
  assertEquals(orSql, "1 = 0");
  assertEquals(orVals.length, 0);
});

Deno.test("filterToSql - Raw expression", () => {
  // Disabled unless explicitly allowed.
  assertThrows(() => filterToSql(Filters.Raw("custom_fn(col) > 0")));
  const [sql, vals] = filterToSql(Filters.Raw("custom_fn(col) > 0"), {
    allowRaw: true,
  });
  assertEquals(sql, "custom_fn(col) > 0");
  assertEquals(vals.length, 0);
});

Deno.test("filterToSql - nested Or/And/In uses positional placeholders throughout", () => {
  const filter = Filters.Or([
    Filters.And([
      Filters.Eq("a", FieldValues.Int32(1)),
      Filters.In("b", [FieldValues.Int32(2), FieldValues.Int32(3)]),
    ]),
    Filters.IsNull("c"),
  ]);
  const [sql, vals] = filterToSql(filter);
  assertEquals(sql, "((`a` = ? AND `b` IN (?, ?)) OR `c` IS NULL)");
  assertEquals(vals.map((v) => v.value), [1, 2, 3]);
});

// ---------------------------------------------------------------------------
// buildCreateTable
// ---------------------------------------------------------------------------

Deno.test("buildCreateTable - basic schema with MySQL syntax", () => {
  const schema: FieldDef[] = [
    requiredField("id", FieldTypes.Utf8),
    requiredField("count", FieldTypes.Int64),
    optionalField("embedding", FieldTypes.Vector(384)),
  ];
  const sql = buildCreateTable("my_table", schema);
  assertEquals(
    sql,
    "CREATE TABLE IF NOT EXISTS `my_table` (`id` TEXT NOT NULL PRIMARY KEY, `count` BIGINT NOT NULL, `embedding` JSON)",
  );
});

// ---------------------------------------------------------------------------
// buildInsert
// ---------------------------------------------------------------------------

Deno.test("buildInsert - two rows with ? placeholders", () => {
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
  assertEquals(sql, "INSERT INTO `users` (`id`, `name`) VALUES (?, ?), (?, ?)");
  assertEquals(params.length, 4);
});

// ---------------------------------------------------------------------------
// buildSelect
// ---------------------------------------------------------------------------

Deno.test("buildSelect - no filter with limit", () => {
  const [sql, params] = buildSelect("messages", undefined, 10);
  assertEquals(sql, "SELECT * FROM `messages` LIMIT 10");
  assertEquals(params.length, 0);
});

Deno.test("buildSelect - with filter", () => {
  const [sql, params] = buildSelect(
    "messages",
    Filters.Eq("user_id", FieldValues.Utf8("u1")),
  );
  assertEquals(sql, "SELECT * FROM `messages` WHERE `user_id` = ?");
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
  assertEquals(sql, "DELETE FROM `tasks` WHERE `id` = ?");
  assertEquals(params.length, 1);
});

// ---------------------------------------------------------------------------
// buildCount
// ---------------------------------------------------------------------------

Deno.test("buildCount - no filter", () => {
  const [sql, params] = buildCount("events");
  assertEquals(sql, "SELECT COUNT(*) AS cnt FROM `events`");
  assertEquals(params.length, 0);
});

Deno.test("buildCount - with filter", () => {
  const [sql, params] = buildCount(
    "events",
    Filters.Gt("score", FieldValues.Float64(0.5)),
  );
  assertEquals(sql, "SELECT COUNT(*) AS cnt FROM `events` WHERE `score` > ?");
  assertEquals(params.length, 1);
});

// ---------------------------------------------------------------------------
// cosineSimilarity
// ---------------------------------------------------------------------------

Deno.test("cosineSimilarity - identical vectors", () => {
  const v = [1, 0, 0];
  const sim = cosineSimilarity(v, v);
  assertEquals(Math.abs(sim - 1.0) < 1e-6, true);
});

Deno.test("cosineSimilarity - orthogonal vectors", () => {
  const a = [1, 0];
  const b = [0, 1];
  assertEquals(Math.abs(cosineSimilarity(a, b)) < 1e-6, true);
});

Deno.test("cosineSimilarity - empty vectors", () => {
  assertEquals(cosineSimilarity([], []), 0);
});

Deno.test("cosineSimilarity - mismatched lengths", () => {
  assertEquals(cosineSimilarity([1], [1, 2]), 0);
});
