/**
 * Tests for SurrealDB backend: query building, type conversion, filter generation.
 *
 * These tests exercise the pure helper functions (no live SurrealDB required).
 */

import { assertEquals } from "@std/assert";
import {
  fieldTypeToSurrealQL,
  fieldValueToJson,
  filterToSurrealQL,
  jsonRowToRecord,
} from "./surrealdb.ts";
import { FieldTypes, FieldValues, Filters } from "../types.ts";

// ---------------------------------------------------------------------------
// fieldTypeToSurrealQL
// ---------------------------------------------------------------------------

Deno.test("fieldTypeToSurrealQL - Utf8", () => {
  assertEquals(fieldTypeToSurrealQL(FieldTypes.Utf8), "string");
});

Deno.test("fieldTypeToSurrealQL - Int32 / Int64", () => {
  assertEquals(fieldTypeToSurrealQL(FieldTypes.Int32), "int");
  assertEquals(fieldTypeToSurrealQL(FieldTypes.Int64), "int");
});

Deno.test("fieldTypeToSurrealQL - UInt32 / UInt64", () => {
  assertEquals(fieldTypeToSurrealQL(FieldTypes.UInt32), "int");
  assertEquals(fieldTypeToSurrealQL(FieldTypes.UInt64), "int");
});

Deno.test("fieldTypeToSurrealQL - Float32 / Float64", () => {
  assertEquals(fieldTypeToSurrealQL(FieldTypes.Float32), "float");
  assertEquals(fieldTypeToSurrealQL(FieldTypes.Float64), "float");
});

Deno.test("fieldTypeToSurrealQL - Boolean", () => {
  assertEquals(fieldTypeToSurrealQL(FieldTypes.Boolean), "bool");
});

Deno.test("fieldTypeToSurrealQL - Vector", () => {
  assertEquals(
    fieldTypeToSurrealQL(FieldTypes.Vector(384)),
    "array<float, 384>",
  );
});

// ---------------------------------------------------------------------------
// fieldValueToJson
// ---------------------------------------------------------------------------

Deno.test("fieldValueToJson - string", () => {
  assertEquals(fieldValueToJson(FieldValues.Utf8("hello")), "hello");
});

Deno.test("fieldValueToJson - null string", () => {
  assertEquals(fieldValueToJson(FieldValues.Utf8(null)), null);
});

Deno.test("fieldValueToJson - int", () => {
  assertEquals(fieldValueToJson(FieldValues.Int32(42)), 42);
  assertEquals(fieldValueToJson(FieldValues.Int64(999)), 999);
});

Deno.test("fieldValueToJson - float", () => {
  assertEquals(fieldValueToJson(FieldValues.Float64(3.14)), 3.14);
});

Deno.test("fieldValueToJson - boolean", () => {
  assertEquals(fieldValueToJson(FieldValues.Boolean(true)), true);
  assertEquals(fieldValueToJson(FieldValues.Boolean(false)), false);
});

Deno.test("fieldValueToJson - vector", () => {
  assertEquals(fieldValueToJson(FieldValues.Vector([1, 2, 3])), [1, 2, 3]);
});

// ---------------------------------------------------------------------------
// filterToSurrealQL
// ---------------------------------------------------------------------------

Deno.test("filterToSurrealQL - Eq", () => {
  const offset = { value: 0 };
  const [sql, binds] = filterToSurrealQL(
    Filters.Eq("name", FieldValues.Utf8("Alice")),
    offset,
  );
  assertEquals(sql, "name = $p0");
  assertEquals(binds.length, 1);
  assertEquals(binds[0][0], "p0");
  assertEquals(binds[0][1], "Alice");
});

Deno.test("filterToSurrealQL - Ne", () => {
  const offset = { value: 0 };
  const [sql] = filterToSurrealQL(
    Filters.Ne("status", FieldValues.Utf8("deleted")),
    offset,
  );
  assertEquals(sql, "status != $p0");
});

Deno.test("filterToSurrealQL - Lt / Gt / Lte / Gte", () => {
  let offset = { value: 0 };
  const [ltSql] = filterToSurrealQL(
    Filters.Lt("x", FieldValues.Int32(5)),
    offset,
  );
  assertEquals(ltSql, "x < $p0");

  offset = { value: 0 };
  const [gtSql] = filterToSurrealQL(
    Filters.Gt("x", FieldValues.Int32(5)),
    offset,
  );
  assertEquals(gtSql, "x > $p0");

  offset = { value: 0 };
  const [lteSql] = filterToSurrealQL(
    Filters.Lte("x", FieldValues.Int32(5)),
    offset,
  );
  assertEquals(lteSql, "x <= $p0");

  offset = { value: 0 };
  const [gteSql] = filterToSurrealQL(
    Filters.Gte("x", FieldValues.Int32(5)),
    offset,
  );
  assertEquals(gteSql, "x >= $p0");
});

Deno.test("filterToSurrealQL - IsNull / NotNull", () => {
  const offset = { value: 0 };
  const [isNullSql, isNullBinds] = filterToSurrealQL(
    Filters.IsNull("email"),
    offset,
  );
  assertEquals(isNullSql, "email IS NULL");
  assertEquals(isNullBinds.length, 0);

  const [notNullSql, notNullBinds] = filterToSurrealQL(
    Filters.NotNull("email"),
    offset,
  );
  assertEquals(notNullSql, "email IS NOT NULL");
  assertEquals(notNullBinds.length, 0);
});

Deno.test("filterToSurrealQL - In", () => {
  const offset = { value: 0 };
  const [sql, binds] = filterToSurrealQL(
    Filters.In("id", [
      FieldValues.Int64(1),
      FieldValues.Int64(2),
      FieldValues.Int64(3),
    ]),
    offset,
  );
  assertEquals(sql, "id IN $p0");
  assertEquals(binds.length, 1);
  assertEquals(binds[0][1], [1, 2, 3]);
});

Deno.test("filterToSurrealQL - empty In", () => {
  const offset = { value: 0 };
  const [sql, binds] = filterToSurrealQL(Filters.In("id", []), offset);
  assertEquals(sql, "false");
  assertEquals(binds.length, 0);
});

Deno.test("filterToSurrealQL - And compound", () => {
  const offset = { value: 0 };
  const [sql, binds] = filterToSurrealQL(
    Filters.And([
      Filters.Eq("name", FieldValues.Utf8("Alice")),
      Filters.Gt("age", FieldValues.Int32(21)),
    ]),
    offset,
  );
  assertEquals(sql, "(name = $p0 AND age > $p1)");
  assertEquals(binds.length, 2);
});

Deno.test("filterToSurrealQL - Or compound", () => {
  const offset = { value: 0 };
  const [sql, binds] = filterToSurrealQL(
    Filters.Or([
      Filters.Eq("status", FieldValues.Utf8("active")),
      Filters.Eq("status", FieldValues.Utf8("pending")),
    ]),
    offset,
  );
  assertEquals(sql, "(status = $p0 OR status = $p1)");
  assertEquals(binds.length, 2);
});

Deno.test("filterToSurrealQL - empty And / Or", () => {
  const offset = { value: 0 };
  const [andSql] = filterToSurrealQL(Filters.And([]), offset);
  assertEquals(andSql, "true");

  const [orSql] = filterToSurrealQL(Filters.Or([]), offset);
  assertEquals(orSql, "false");
});

Deno.test("filterToSurrealQL - Raw", () => {
  const offset = { value: 0 };
  const [sql, binds] = filterToSurrealQL(Filters.Raw("custom_fn()"), offset, {
    allowRaw: true,
  });
  assertEquals(sql, "custom_fn()");
  assertEquals(binds.length, 0);
});

// ---------------------------------------------------------------------------
// jsonRowToRecord
// ---------------------------------------------------------------------------

Deno.test("jsonRowToRecord - basic types", () => {
  const row = {
    id: "table:abc123",
    name: "test",
    count: 42,
    active: true,
    score: 0.95,
  };
  const record = jsonRowToRecord(row);
  assertEquals(record.length, 5);

  const idField = record.find(([n]) => n === "id");
  assertEquals(idField?.[1].kind, "Utf8");
  assertEquals(idField?.[1].value, "table:abc123");

  const nameField = record.find(([n]) => n === "name");
  assertEquals(nameField?.[1].kind, "Utf8");

  const countField = record.find(([n]) => n === "count");
  assertEquals(countField?.[1].kind, "Int64");
  assertEquals(countField?.[1].value, 42);

  const activeField = record.find(([n]) => n === "active");
  assertEquals(activeField?.[1].kind, "Boolean");
  assertEquals(activeField?.[1].value, true);
});

Deno.test("jsonRowToRecord - null values", () => {
  const record = jsonRowToRecord({ name: null });
  assertEquals(record.length, 1);
  assertEquals(record[0][1].kind, "Utf8");
  assertEquals(record[0][1].value, null);
});

Deno.test("jsonRowToRecord - vector arrays", () => {
  const record = jsonRowToRecord({ embedding: [1.0, 2.0, 3.0] });
  assertEquals(record.length, 1);
  assertEquals(record[0][1].kind, "Vector");
  assertEquals(record[0][1].value, [1, 2, 3]);
});

Deno.test("jsonRowToRecord - id as object", () => {
  const record = jsonRowToRecord({ id: { tb: "x", id: "123" } });
  const idField = record.find(([n]) => n === "id");
  assertEquals(idField?.[1].kind, "Utf8");
  // Should be a JSON-stringified object
  assertEquals(typeof idField?.[1].value, "string");
});

Deno.test("jsonRowToRecord - float detection", () => {
  const record = jsonRowToRecord({ val: 3.14 });
  assertEquals(record[0][1].kind, "Float64");
  assertEquals(record[0][1].value, 3.14);
});
